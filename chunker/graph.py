from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


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
        self.units[unit.unit_id] = unit

    def add_edge(self, edge: GraphEdge) -> None:
        self.edges.append(edge)

    def neighbors(self, unit_id: str, max_hops: int = 1) -> List[str]:
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
        return [nid for nid in visited if nid != unit_id]

    def build_structural_edges(self, units: List[TextUnit]) -> None:
        ordered = sorted(units, key=lambda u: u.index)
        for i in range(len(ordered) - 1):
            source_id = ordered[i].unit_id
            target_id = ordered[i + 1].unit_id
            self.edges.append(GraphEdge(source=source_id, target=target_id, edge_type="structural", weight=1.0))
            self.edges.append(GraphEdge(source=target_id, target=source_id, edge_type="structural", weight=1.0))

    def build_hierarchical_edges(self, units: List[TextUnit]) -> None:
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

    def build_semantic_edges(self, units: List[TextUnit], threshold: float = 0.75) -> None:
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

        unit_ids = list(vectors.keys())
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

    def detect_communities(self, resolution: float = 1.0) -> Dict[str, int]:
        if not self.units:
            return {}
        unit_list = list(self.units.keys())
        unit_to_idx = {uid: idx for idx, uid in enumerate(unit_list)}
        comms = _greedy_modularity(unit_list, self.edges, unit_to_idx, resolution)
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


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
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
    if n <= 1:
        return [unit_ids[:]] if unit_ids else []

    adjacency_builder: Dict[int, Dict[int, float]] = {i: {} for i in range(n)}
    total_weight = 0.0
    for edge in edges:
        src = unit_to_idx.get(edge.source)
        tgt = unit_to_idx.get(edge.target)
        if src is None or tgt is None:
            continue
        key = (min(src, tgt), max(src, tgt))
        adjacency_builder[key[0]][key[1]] = adjacency_builder[key[0]].get(key[1], 0.0) + edge.weight
        adjacency_builder[key[1]][key[0]] = adjacency_builder[key[1]].get(key[0], 0.0) + edge.weight

    adjacency: Dict[int, List[Tuple[int, float]]] = {i: [] for i in range(n)}
    for src, targets in adjacency_builder.items():
        for tgt, weight in targets.items():
            adjacency[src].append((tgt, weight))
            total_weight += weight

    if total_weight == 0.0:
        return [[uid] for uid in unit_ids]

    m_inv = 1.0 / total_weight
    communities = [[unit_ids[i]] for i in range(n)]
    com_idx = list(range(n))
    degrees = [sum(w for _, w in adjacency[i]) for i in range(n)]
    internal = [0.0 for _ in range(n)]
    for edge in edges:
        src = unit_to_idx.get(edge.source)
        tgt = unit_to_idx.get(edge.target)
        if src is None or tgt is None:
            continue
        key = (min(src, tgt), max(src, tgt))
        weight = adjacency_builder[key[0]].get(key[1], 0.0)
        if src == tgt:
            internal[src] += weight * 2.0

    def _community_degree(ci: int) -> float:
        return sum(degrees[i] for i in range(n) if com_idx[i] == ci)

    def _internal_weight(ci: int) -> float:
        return sum(
            sum(w for n, w in adjacency[i] if com_idx[n] == ci)
            for i in range(n) if com_idx[i] == ci
        )

    improved = True
    while improved:
        improved = False
        for i in range(n):
            ci = com_idx[i]
            ki_in = sum(w for n, w in adjacency[i] if com_idx[n] == ci and n != i)
            sigma_in = internal[ci]
            sigma_total = _community_degree(ci)

            best_delta = 0.0
            best_ci = ci
            seen = set()
            for neighbor, weight in adjacency[i]:
                cj = com_idx[neighbor]
                if cj in seen:
                    continue
                seen.add(cj)
                if cj == ci:
                    continue
                ki_out = sum(w for n, w in adjacency[i] if com_idx[n] == cj)
                delta = 2.0 * weight * m_inv - (sigma_total + degrees[i]) * _community_degree(cj) * m_inv * m_inv + sigma_in * m_inv
                if delta > best_delta:
                    best_delta = delta
                    best_ci = cj

            if best_ci != ci and best_delta > 0.0:
                internal[ci] -= 2.0 * ki_in
                internal[best_ci] += 2.0 * ki_in
                com_idx[i] = best_ci
                improved = True

    result: Dict[int, List[str]] = {}
    for i, uid in enumerate(unit_ids):
        label = com_idx[i]
        result.setdefault(label, []).append(uid)
    return list(result.values())
