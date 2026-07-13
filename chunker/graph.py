from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

try:
    from utils.logging import debug as _debug
except Exception:  # pragma: no cover - fallback if utils import path is unavailable
    def _debug(msg: str, category: str = "") -> None:
        pass

# Configurable debug category. Enable via [logging] categories=graph in config.cfg
GRAPH_DEBUG_CATEGORY = "graph"


def _log(msg: str) -> None:
    _debug(msg, GRAPH_DEBUG_CATEGORY)


@dataclass
class TextUnit:
    unit_id: str
    document_id: str
    text: str
    unit_type: str = "sentence"
    metadata: Dict[str, Any] = field(default_factory=dict)
    index: int = 0

    def content_hash(self) -> str:
        normalized = " ".join(self.text.split()).encode("utf-8", errors="ignore")
        return hashlib.sha256(normalized).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "document_id": self.document_id,
            "text": self.text,
            "unit_type": self.unit_type,
            "metadata": self.metadata,
            "index": self.index,
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "TextUnit":
        return TextUnit(
            unit_id=data.get("unit_id", ""),
            document_id=data.get("document_id", ""),
            text=data.get("text", ""),
            unit_type=data.get("unit_type", "sentence"),
            metadata=data.get("metadata", {}),
            index=data.get("index", 0),
        )


@dataclass
class GraphEdge:
    source: str
    target: str
    edge_type: str = "structural"
    weight: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "edge_type": self.edge_type,
            "weight": self.weight,
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "GraphEdge":
        return GraphEdge(
            source=data.get("source", ""),
            target=data.get("target", ""),
            edge_type=data.get("edge_type", "structural"),
            weight=float(data.get("weight", 1.0)),
        )


class ChunkGraph:
    def __init__(self) -> None:
        self.edges: List[GraphEdge] = []
        self.units: Dict[str, TextUnit] = {}

    def add_unit(self, unit: TextUnit) -> None:
        _log("add_unit: unit_id=%s document_id=%s units=%d" % (unit.unit_id, unit.document_id, len(self.units)))
        self.units[unit.unit_id] = unit

    def add_edge(self, edge: GraphEdge) -> None:
        _log("add_edge: source=%s target=%s type=%s weight=%.4f edges=%d" % (edge.source, edge.target, edge.edge_type, edge.weight, len(self.edges)))
        self.edges.append(edge)

    def neighbors(self, unit_id: str, max_hops: int = 1) -> List[str]:
        _log("neighbors: unit_id=%s max_hops=%d total_units=%d total_edges=%d" % (unit_id, max_hops, len(self.units), len(self.edges)))
        adjacency: Dict[str, List[Tuple[str, float]]] = {uid: [] for uid in self.units}
        for edge in self.edges:
            adjacency.setdefault(edge.source, []).append((edge.target, edge.weight))
            adjacency.setdefault(edge.target, []).append((edge.source, edge.weight))

        visited = {unit_id}
        frontier = [unit_id]
        for _ in range(max_hops):
            next_frontier = []
            for current in frontier:
                for neighbor, _ in adjacency.get(current, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        next_frontier.append(neighbor)
            frontier = next_frontier
        result = [nid for nid in visited if nid != unit_id]
        _log("neighbors: unit_id=%s found=%d" % (unit_id, len(result)))
        return result

    def build_structural_edges(self, units: List[TextUnit]) -> None:
        _log("build_structural_edges: units=%d" % len(units))
        ordered = sorted(units, key=lambda u: u.index)
        for i in range(len(ordered) - 1):
            source_id = ordered[i].unit_id
            target_id = ordered[i + 1].unit_id
            self.edges.append(GraphEdge(source=source_id, target=target_id, edge_type="structural", weight=1.0))
            self.edges.append(GraphEdge(source=target_id, target=source_id, edge_type="structural", weight=1.0))
        _log("build_structural_edges: added=%d edges=%d" % ((len(ordered) - 1) * 2, len(self.edges)))

    def build_hierarchical_edges(self, units: List[TextUnit]) -> None:
        _log("build_hierarchical_edges: units=%d" % len(units))
        by_section: Dict[str, List[TextUnit]] = {}
        for unit in units:
            section = str(unit.metadata.get("chapter") or unit.metadata.get("section") or unit.metadata.get("book") or "root")
            by_section.setdefault(section, []).append(unit)

        for section, section_units in by_section.items():
            ordered = sorted(section_units, key=lambda u: u.index)
            for i in range(len(ordered) - 1):
                source_id = ordered[i].unit_id
                target_id = ordered[i + 1].unit_id
                self.edges.append(GraphEdge(source=source_id, target=target_id, edge_type="hierarchical", weight=0.5))
        _log("build_hierarchical_edges: sections=%d edges=%d" % (len(by_section), len(self.edges)))

    def build_semantic_edges(self, units: List[TextUnit], threshold: float = 0.75) -> None:
        _log("build_semantic_edges: units=%d threshold=%.4f" % (len(units), threshold))
        vectors: Dict[str, List[float]] = {}
        for unit in units:
            embedding = unit.metadata.get("embedding")
            if embedding is None:
                continue
            if isinstance(embedding, str):
                try:
                    embedding = json.loads(embedding)
                except json.JSONDecodeError:
                    continue
            if not isinstance(embedding, list) or len(embedding) == 0:
                continue
            vectors[unit.unit_id] = [float(v) for v in embedding]

        _log("build_semantic_edges: units_with_embedding=%d" % len(vectors))
        unit_ids = list(vectors.keys())
        semantic_edges_before = len(self.edges)
        for i in range(len(unit_ids)):
            for j in range(i + 1, len(unit_ids)):
                similarity = _cosine_similarity(vectors[unit_ids[i]], vectors[unit_ids[j]])
                if similarity >= threshold:
                    self.edges.append(GraphEdge(
                        source=unit_ids[i],
                        target=unit_ids[j],
                        edge_type="semantic",
                        weight=similarity,
                    ))
        _log("build_semantic_edges: added=%d edges=%d" % (len(self.edges) - semantic_edges_before, len(self.edges)))

    def build_keyword_edges(self, units: List[TextUnit], min_shared: int = 3, max_per_unit: int = 5) -> None:
        """Connect units that share significant keywords/topics.

        This links semantically related (but non-adjacent) units so that
        community detection groups topical content together and retrieval
        expansion can surface related chunks even when they are far apart
        in the document. Purely lexical, so it works without embeddings.
        """
        _log("build_keyword_edges: units=%d min_shared=%d max_per_unit=%d" % (len(units), min_shared, max_per_unit))
        keywords: Dict[str, set] = {}
        for unit in units:
            keywords[unit.unit_id] = _extract_keywords(unit.text)

        # Inverted index (keyword -> unit ids) so we only compare units that
        # actually share a keyword, instead of all O(n^2) unit pairs.
        keyword_to_units: Dict[str, List[str]] = {}
        for unit_id, kws in keywords.items():
            for kw in kws:
                keyword_to_units.setdefault(kw, []).append(unit_id)

        keyword_edges_before = len(self.edges)
        for ui, ki in keywords.items():
            if not ki:
                continue
            # Count shared keywords only against candidates reachable via the index.
            shared_counts: Dict[str, int] = {}
            for kw in ki:
                for uj in keyword_to_units.get(kw, ()):
                    if uj == ui:
                        continue
                    shared_counts[uj] = shared_counts.get(uj, 0) + 1
            scored: List[Tuple[int, str]] = [
                (shared, uj) for uj, shared in shared_counts.items() if shared >= min_shared
            ]
            scored.sort(reverse=True)
            for shared, uj in scored[:max_per_unit]:
                weight = min(0.6, 0.3 + 0.1 * shared)
                self.edges.append(GraphEdge(source=ui, target=uj, edge_type="keyword", weight=weight))
        _log("build_keyword_edges: added=%d edges=%d" % (len(self.edges) - keyword_edges_before, len(self.edges)))

    def detect_communities(self, resolution: float = 1.0) -> Dict[str, int]:
        if not self.units:
            return {}
        _log("detect_communities: units=%d edges=%d resolution=%.4f" % (len(self.units), len(self.edges), resolution))
        unit_list = list(self.units.keys())
        unit_to_idx = {uid: idx for idx, uid in enumerate(unit_list)}
        comms = _greedy_modularity(unit_list, self.edges, unit_to_idx, resolution)
        _log("detect_communities: communities=%d" % len(comms))
        return {unit_id: label for label, group in enumerate(comms) for unit_id in group}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "units": [unit.to_dict() for unit in self.units.values()],
            "edges": [edge.to_dict() for edge in self.edges],
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "ChunkGraph":
        graph = ChunkGraph()
        for unit_data in data.get("units", []):
            unit = TextUnit.from_dict(unit_data)
            graph.add_unit(unit)
        for edge_data in data.get("edges", []):
            graph.add_edge(GraphEdge.from_dict(edge_data))
        return graph


_KEYWORD_STOPWORDS = {
    "this", "that", "with", "from", "they", "them", "were", "been", "have", "has",
    "will", "would", "could", "should", "their", "there", "here", "what", "when",
    "where", "which", "while", "about", "after", "before", "because", "been",
    "being", "into", "than", "then", "over", "also", "some", "such", "only",
    "very", "just", "like", "more", "most", "other", "these", "those", "them",
}


def _extract_keywords(text: str, top_k: int = 12) -> set:
    """Return a small set of significant keyword stems from text (lexical only)."""
    if not text:
        return set()
    words = re.findall(r"[a-zA-Z0-9à-ỹÀ-Ỹ]{4,}", text.lower())
    counts: Dict[str, int] = {}
    for word in words:
        if word in _KEYWORD_STOPWORDS:
            continue
        counts[word] = counts.get(word, 0) + 1
    if not counts:
        return set()
    top = sorted(counts.items(), key=lambda kv: (kv[1], len(kv[0])), reverse=True)[:top_k]
    return {word for word, _ in top}


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        _log("_cosine_similarity: mismatched/empty vectors -> 0.0")
        return 0.0
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b):
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    denom = math.sqrt(norm_a) * math.sqrt(norm_b)
    if denom == 0.0:
        return 0.0
    return dot / denom


def _greedy_modularity(
    unit_ids: List[str],
    edges: List[GraphEdge],
    unit_to_idx: Dict[str, int],
    resolution: float,
) -> List[List[str]]:
    n = len(unit_ids)
    _log("_greedy_modularity: n=%d edges=%d resolution=%.4f" % (n, len(edges), resolution))
    if n <= 1:
        return [unit_ids[:]] if unit_ids else []

    adjacency: Dict[int, List[Tuple[int, float]]] = {i: [] for i in range(n)}
    degrees = [0.0] * n
    for edge in edges:
        src = unit_to_idx.get(edge.source)
        tgt = unit_to_idx.get(edge.target)
        if src is None or tgt is None:
            continue
        adjacency[src].append((tgt, edge.weight))
        adjacency[tgt].append((src, edge.weight))
        degrees[src] += edge.weight
        degrees[tgt] += edge.weight

    total = sum(degrees)  # = 2 * (sum of edge weights), i.e. 2m
    if total == 0.0:
        return [[uid] for uid in unit_ids]

    # Louvain first phase: greedily move nodes to maximize modularity.
    # Each accepted move strictly increases modularity, so the loop terminates.
    com = list(range(n))
    com_tot = list(degrees)  # total degree per community

    max_passes = 100
    for _pass in range(max_passes):
        improved = False
        for i in range(n):
            ci = com[i]
            ki_in_ci = sum(w for nb, w in adjacency[i] if com[nb] == ci and nb != i)
            # Temporarily remove i from its current community before scoring.
            com_tot[ci] -= degrees[i]
            best_com = ci
            best_gain = (2.0 * ki_in_ci / total) - (resolution * degrees[i] * com_tot[ci] / (total * total))
            seen = set()
            for nb, _w in adjacency[i]:
                cj = com[nb]
                if cj == ci or cj in seen:
                    continue
                seen.add(cj)
                ki_in_cj = sum(w2 for nb2, w2 in adjacency[i] if com[nb2] == cj)
                gain = (2.0 * ki_in_cj / total) - (resolution * degrees[i] * com_tot[cj] / (total * total))
                if gain > best_gain:
                    best_gain = gain
                    best_com = cj
            com[i] = best_com
            com_tot[best_com] += degrees[i]
            if best_com != ci:
                improved = True
        if not improved:
            break

    result: Dict[int, List[str]] = {}
    for i, uid in enumerate(unit_ids):
        result.setdefault(com[i], []).append(uid)
    communities = list(result.values())
    _log("_greedy_modularity: produced=%d communities" % len(communities))
    return communities
