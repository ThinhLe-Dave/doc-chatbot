from __future__ import annotations

import json
from dataclasses import dataclass, field
import os
import re
from functools import lru_cache
from typing import Any, Dict, Iterable, Iterator, List, Optional, Set, Tuple

from chunker.document import compute_content_hash, Document
from chunker.keywords import extract_keywords as _extract_keywords
from chunker.graph import ChunkGraph, TextUnit

_SENTENCE_END_RE = re.compile(r'(?<=[.!?])\s+')


def get_chunk_file_path(input_file: str) -> str:
    base, _ = os.path.splitext(input_file)
    return f"{base}_chunks.json"


def get_embedding_ids_path(chunk_file: str) -> str:
    base, _ = os.path.splitext(chunk_file)
    return f"{base}_chunk_ids.json"


def get_embedding_file_path(chunk_file: str) -> str:
    base, _ = os.path.splitext(chunk_file)
    return f"{base}_embeddings.npz"


def get_embedding_matrix_path(chunk_file: str) -> str:
    base, _ = os.path.splitext(chunk_file)
    return f"{base}_embeddings.npy"


def get_model_meta_path(chunk_file: str) -> str:
    base, _ = os.path.splitext(chunk_file)
    return f"{base}_embeddings_meta.json"


@lru_cache(maxsize=128)
def _load_chunk_ids_cached(chunk_file: str) -> Optional[Tuple[str, ...]]:
    ids_file = get_embedding_ids_path(chunk_file)
    if not os.path.exists(ids_file):
        return None
    try:
        with open(ids_file, "r", encoding="utf-8") as f:
            return tuple(str(value) for value in json.load(f))
    except Exception:
        return None


@lru_cache(maxsize=128)
def _count_chunks_cached(chunk_file: str) -> int:
    if not os.path.exists(chunk_file):
        return 0
    count = 0
    with open(chunk_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


_JSON_FORMAT_CACHE: Dict[str, bool] = {}


def _is_json_array(path: str) -> bool:
    if path in _JSON_FORMAT_CACHE:
        return _JSON_FORMAT_CACHE[path]
    try:
        with open(path, "r", encoding="utf-8") as f:
            first_char = f.read(1)
    except Exception:
        _JSON_FORMAT_CACHE[path] = False
        return False
    result = first_char == "["
    _JSON_FORMAT_CACHE[path] = result
    return result


def save_chunks(chunks: Iterable["Chunk"], output_file: str) -> None:
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk.to_dict(), ensure_ascii=False))
            f.write("\n")


def load_chunks_from_json(path: str) -> List["Chunk"]:
    if not os.path.exists(path):
        return []

    if _is_json_array(path):
        with open(path, "r", encoding="utf-8") as f:
            raw_chunks = json.load(f)
        return [Chunk.from_dict(item) for item in raw_chunks]

    chunks: List[Chunk] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            chunks.append(Chunk.from_dict(json.loads(line)))
    return chunks


def iter_chunks_from_json(path: str) -> Iterator["Chunk"]:
    if not os.path.exists(path):
        return

    if _is_json_array(path):
        with open(path, "r", encoding="utf-8") as f:
            raw_chunks = json.load(f)
        for item in raw_chunks:
            yield Chunk.from_dict(item)
        return

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield Chunk.from_dict(json.loads(line))


def iter_chunk_batches(path: str, batch_size: int) -> Iterator[List["Chunk"]]:
    batch: List[Chunk] = []
    for chunk in iter_chunks_from_json(path):
        batch.append(chunk)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def iter_chunk_batches_by_indices(path: str, indices: Set[int], batch_size: int) -> Iterator[Tuple[int, "Chunk"]]:
    try:
        from chunker.chunk_index import get_chunk_offsets_path, load_chunk_offsets
        offsets_file = get_chunk_offsets_path(path)
        if os.path.exists(offsets_file) and not _is_json_array(path):
            offsets = load_chunk_offsets(path)
            if offsets is not None:
                sorted_indices = sorted(indices)
                with open(path, "r", encoding="utf-8") as f:
                    batch: List[Tuple[int, Chunk]] = []
                    for idx in sorted_indices:
                        if idx < len(offsets):
                            offset, length = offsets[idx]
                            f.seek(offset)
                            line = f.readline().strip()
                            if line:
                                batch.append((idx, Chunk.from_dict(json.loads(line))))
                                if len(batch) >= batch_size:
                                    yield from batch
                                    batch = []
                    if batch:
                        yield from batch
                return
    except Exception:
        pass

    current_index = 0
    batch: List[Tuple[int, Chunk]] = []
    for chunk in iter_chunks_from_json(path):
        if current_index in indices:
            batch.append((current_index, chunk))
            if len(batch) >= batch_size:
                yield from batch
                batch = []
        current_index += 1
    if batch:
        yield from batch


def count_chunks_in_json(path: str) -> int:
    return _count_chunks_cached(path)


def get_chunk_count(chunk_file: str) -> int:
    return _count_chunks_cached(chunk_file)


def get_chunk_ids_from_json(path: str) -> List[str]:
    return [chunk.id for chunk in iter_chunks_from_json(path)]


def load_chunk_ids_from_cache(chunk_file: str) -> Optional[List[str]]:
    cached = _load_chunk_ids_cached(chunk_file)
    if cached is not None:
        return list(cached)
    return None


def cache_chunk_ids(chunk_file: str, chunk_ids: List[str]) -> None:
    _load_chunk_ids_cached.cache_clear()


def write_chunks_to_file(documents: List["Document"], output_path: str, graph_mode: bool = False, semantic_threshold: float = 0.75, resolution: float = 1.0) -> int:
    """Write document chunks to JSONL file. Returns total chunk count."""
    import os
    import json
    from chunker.document import _log_memory_error

    chunker = Chunker()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    total_chunks = 0

    with open(output_path, "w", encoding="utf-8") as f:
        for document in documents:
            try:
                if graph_mode:
                    chunks = chunker.create_graph_chunks(document, semantic_threshold=semantic_threshold, resolution=resolution)[0]
                else:
                    chunks = chunker.create_chunks_from_document(document)
                for chunk in chunks:
                    f.write(json.dumps(chunk.to_dict(), ensure_ascii=False))
                    f.write("\n")
                    total_chunks += 1
            except MemoryError:
                _log_memory_error(
                    "Chunk creation",
                    "The input document is too large to process in memory.",
                )
                raise MemoryError("Chunk creation failed: insufficient memory")

    return total_chunks


def compute_keyword_scores(
    chunk_file: str,
    candidate_indices: Set[int],
    query_terms: Set[str],
) -> Dict[int, float]:
    """Compute keyword match scores for candidates."""
    keyword_scores: Dict[int, float] = {}
    chunk_iter = iter_chunk_batches_by_indices(chunk_file, candidate_indices, batch_size=128)
    for index, chunk in chunk_iter:
        text = (chunk.content + " " + chunk.metadata.get("title", "")).lower()
        matched = sum(1 for term in query_terms if term in text)
        keyword_scores[index] = matched / len(query_terms) if query_terms else 0.0
    return keyword_scores


def gather_candidate_chunks(
    chunk_file: str,
    candidate_indices: Set[int],
    scores: "np.ndarray",
    query_terms: Set[str],
    hybrid: bool,
    hybrid_weight: float,
    min_score: float,
    chunk_k: int,
) -> Dict[str, Dict]:
    """Gather and group chunks by document."""
    import numpy as np
    from chunker.document import build_document_entry, update_document_entry

    document_chunks: Dict[str, Dict] = {}
    keyword_scores: Dict[int, float] = {}

    if hybrid and query_terms:
        keyword_scores = compute_keyword_scores(chunk_file, candidate_indices, query_terms)

    chunk_iter = iter_chunk_batches_by_indices(chunk_file, candidate_indices, batch_size=128)
    for index, chunk in chunk_iter:
        score_value = float(scores[index])
        if hybrid and index in keyword_scores:
            score_value = (1.0 - hybrid_weight) * score_value + hybrid_weight * keyword_scores[index]

        if score_value < min_score:
            continue

        document_id = chunk.document_id
        entry = document_chunks.get(document_id)

        if entry is None:
            entry = build_document_entry(
                chunk_id=chunk.id,
                document_id=chunk.document_id,
                content=chunk.content,
                path=chunk.path,
                metadata=chunk.metadata,
                score_value=score_value,
                chunk_k=chunk_k,
            )
            document_chunks[document_id] = entry
        else:
            update_document_entry(entry, chunk.content, score_value, chunk_k)

    return document_chunks


@dataclass
class Chunk:
    id: str
    document_id: str
    content: str
    path: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    unit_ids: List[str] = field(default_factory=list)
    graph_id: Optional[str] = None
    parent_chunk_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "document_id": self.document_id,
            "content": self.content,
            "path": self.path,
            "metadata": self.metadata,
            "unit_ids": self.unit_ids,
            "graph_id": self.graph_id,
            "parent_chunk_id": self.parent_chunk_id,
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "Chunk":
        meta = dict(data.get("metadata", {}))
        if "categories" not in meta:
            headers = meta.get("headers", [])
            if isinstance(headers, list):
                derived = [h.strip() for h in headers if isinstance(h, str) and h.strip()]
            else:
                derived = []
            for key in ("book", "chapter", "section", "verse", "source", "title"):
                value = meta.get(key)
                if value and str(value) not in derived:
                    derived.append(str(value))
            meta["categories"] = derived
        content = data.get("content", "")
        meta.setdefault("source_hash", compute_content_hash(meta.get("source_hash") or content))
        meta.setdefault("document_hash", compute_content_hash(meta.get("document_hash") or content))
        return Chunk(
            id=data.get("id", ""),
            document_id=data.get("document_id", ""),
            content=content,
            path=data.get("path", []),
            metadata=meta,
            unit_ids=data.get("unit_ids", []),
            graph_id=data.get("graph_id"),
            parent_chunk_id=data.get("parent_chunk_id"),
        )


class Chunker:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def _overlap_word_count(self) -> int:
        words = max(1, self.chunk_overlap // 4)
        return min(words, 20)

    def _force_split_text(self, text: str) -> List[str]:
        chunks: List[str] = []
        start = 0
        length = len(text)

        while start < length:
            end = min(start + self.chunk_size, length)
            if end < length:
                boundary = text.rfind(' ', start, end)
                if boundary > start:
                    end = boundary

            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(chunk_text)

            if end >= length:
                break

            start = max(end - self.chunk_overlap, 0)
            if start >= length:
                break

        return chunks

    def _split_text(self, text: str) -> List[str]:
        """Split long text into overlapping chunks while preserving paragraph and sentence boundaries."""
        if not text:
            return []

        text = text.strip()
        if len(text) <= self.chunk_size:
            return [text]

        separator = "\n\n" if "\n\n" in text else "\n"
        paragraphs = [p.strip() for p in text.split(separator) if p.strip()]
        if len(paragraphs) > 1:
            chunks: List[str] = []
            for paragraph in paragraphs:
                if len(paragraph) <= self.chunk_size:
                    chunks.append(paragraph)
                else:
                    chunks.extend(self._split_text(paragraph))
            return chunks

        sentences = [s.strip() for s in _SENTENCE_END_RE.split(text) if s.strip()]
        if not sentences:
            return self._force_split_text(text)

        chunks: List[str] = []
        current_parts: List[str] = []
        current_len = 0
        oversized = False

        for sentence in sentences:
            sentence_len = len(sentence)
            add_space = 1 if current_parts else 0
            projected = current_len + add_space + sentence_len
            if sentence_len > self.chunk_size:
                oversized = True

            if current_parts and projected > self.chunk_size:
                chunk_text = ' '.join(current_parts)
                chunks.append(chunk_text)
                overlap_count = self._overlap_word_count()
                current_parts = current_parts[-overlap_count:] + [sentence]
                current_len = sum(len(w) for w in current_parts) + len(current_parts) - 1
            else:
                current_parts.append(sentence)
                current_len = projected

        if current_parts:
            chunk_text = ' '.join(current_parts)
            if oversized and len(chunk_text) > self.chunk_size:
                chunks.extend(self._force_split_text(chunk_text))
            else:
                chunks.append(chunk_text)

        return chunks

    def _build_chunk_path(self, metadata: Dict[str, Any]) -> List[str]:
        path: List[str] = []
        title = metadata.get("title")
        if title:
            path.append(str(title))

        book = metadata.get("book")
        if book and book not in path:
            path.append(str(book))

        chapter = metadata.get("chapter")
        if chapter and chapter not in path:
            path.append(str(chapter))

        verse = metadata.get("verse")
        if verse and verse not in path:
            path.append(str(verse))

        headers = metadata.get("headers")
        if isinstance(headers, list):
            for header in headers:
                if isinstance(header, str) and header.strip() and header.strip() not in path:
                    path.append(header.strip())
        return path

    def _build_categories(self, metadata: Dict[str, Any], content: str = "") -> List[str]:
        categories: List[str] = []
        for key in ("book", "chapter", "section", "verse"):
            value = metadata.get(key)
            if value:
                categories.append(str(value))
        headers = metadata.get("headers")
        if isinstance(headers, list):
            for header in headers:
                if isinstance(header, str) and header.strip():
                    categories.append(header.strip())
        
        if not categories and content.strip():
            for keyword in _extract_keywords(content):
                if keyword not in categories:
                    categories.append(keyword)
        return categories

    def create_chunks(self, document_id: str, content: str, metadata: Dict[str, Any] = None) -> List["Chunk"]:
        """Create metadata-rich Chunk objects from a document's content."""
        metadata = metadata or {}
        chunk_texts = self._split_text(content)
        path = self._build_chunk_path(metadata)
        source_hash = metadata.get("source_hash") or compute_content_hash(content)
        document_hash = compute_content_hash(metadata.get("title", "") + "\n" + content)
        bases = {
            "source": metadata.get("source", ""),
            "document_id": document_id,
            "title": metadata.get("title", ""),
            "source_hash": source_hash,
            "document_hash": document_hash,
            "book": metadata.get("book"),
            "chapter": metadata.get("chapter"),
            "page": metadata.get("page"),
            "total_pages": metadata.get("total_pages"),
            "extraction_method": metadata.get("extraction_method"),
            "ocr_confidence": metadata.get("ocr_confidence"),
        }
        return [
            Chunk(
                id=f"{document_id}_chunk_{index}",
                document_id=document_id,
                content=chunk_text,
                path=path,
                metadata={**metadata, "chunk_index": index, "source_hash": source_hash, "document_hash": document_hash, "categories": self._build_categories(metadata, chunk_text), **{k: v for k, v in bases.items() if v}},
            )
            for index, chunk_text in enumerate(chunk_texts)
        ]

    def create_chunks_from_document(self, document) -> List["Chunk"]:
        """Create chunks directly from a Document object."""
        return self.create_chunks(
            document_id=document.id,
            content=document.content,
            metadata={
                "source": document.source,
                "title": document.title,
                **(document.metadata or {}),
            },
        )

    def create_graph_chunks(self, document, semantic_threshold: float = 0.75, resolution: float = 1.0) -> Tuple[List["Chunk"], ChunkGraph]:
        """Create graph-aware chunks from a Document.

        Returns a list of Chunk objects with graph metadata and the
        underlying ChunkGraph for persistence.
        """
        document_id = document.id
        base_metadata = {
            "source": document.source,
            "title": document.title,
            **(document.metadata or {}),
        }
        base_metadata.setdefault("source_hash", compute_content_hash(document.content))
        base_metadata.setdefault("document_hash", compute_content_hash(document.title + "\n" + document.content))

        units = _extract_text_units(document)
        if not units:
            return [], ChunkGraph()

        graph = ChunkGraph()
        for unit in units:
            graph.add_unit(unit)
        graph.build_structural_edges(units)
        graph.build_hierarchical_edges(units)
        graph.build_semantic_edges(units, threshold=semantic_threshold)

        communities = graph.detect_communities(resolution=resolution)

        doc_units_by_community: Dict[int, List[TextUnit]] = {}
        for unit in units:
            label = communities.get(unit.unit_id, 0)
            doc_units_by_community.setdefault(label, []).append(unit)

        result_chunks: List["Chunk"] = []
        chunk_index = 0
        for label in sorted(doc_units_by_community.keys()):
            group = sorted(doc_units_by_community[label], key=lambda u: u.index)
            combined = " ".join(u.text for u in group)
            unit_ids = [u.unit_id for u in group]
            metadata = {
                **base_metadata,
                "chunk_index": chunk_index,
                "categories": self._build_categories(base_metadata, combined),
                "unit_ids": unit_ids,
                "graph_id": f"{document_id}_graph_{chunk_index}",
                "source_hash": base_metadata.get("source_hash"),
                "document_hash": base_metadata.get("document_hash"),
            }
            result_chunks.append(Chunk(
                id=f"{document_id}_graph_{chunk_index}",
                document_id=document_id,
                content=combined,
                path=self._build_chunk_path(metadata),
                metadata=metadata,
                unit_ids=unit_ids,
                graph_id=f"{document_id}_graph_{chunk_index}",
                parent_chunk_id=None,
            ))
            chunk_index += 1

        return result_chunks, graph


def _extract_text_units(document) -> List[TextUnit]:
    units: List[TextUnit] = []
    source_hash = compute_content_hash(document.content)
    text_segments = [seg.strip() for seg in document.content.split("\n\n") if seg.strip()]
    if not text_segments:
        text_segments = [document.content.strip()] if document.content.strip() else []

    base_metadata = {
        "source": document.source,
        "title": document.title,
        "source_hash": source_hash,
    }
    if document.metadata:
        base_metadata.update(document.metadata)

    idx = 0
    for segment in text_segments:
        unit_id = f"{document.id}_unit_{idx}"
        units.append(TextUnit(
            unit_id=unit_id,
            document_id=document.id,
            text=segment,
            unit_type="paragraph",
            metadata={**base_metadata, "index": idx},
            index=idx,
        ))
        idx += 1

    if idx == 0:
        unit_id = f"{document.id}_unit_0"
        units.append(TextUnit(
            unit_id=unit_id,
            document_id=document.id,
            text=document.content,
            unit_type="paragraph",
            metadata={**base_metadata, "index": 0},
            index=0,
        ))

    return units
