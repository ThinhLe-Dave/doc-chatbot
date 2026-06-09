import json
import os
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple
from uuid import uuid4

try:
    import resource
except ImportError:
    resource = None

import typer


def _format_bytes(value: int) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if value < 1024 or unit == "TB":
            return f"{value:.1f}{unit}"
        value /= 1024
    return f"{value:.1f}PB"


def _get_memory_usage() -> Optional[str]:
    if resource is None:
        return None
    try:
        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return _format_bytes(int(usage))
    except Exception:
        return None


def _log_memory_error(operation: str, details: str = "") -> None:
    memory_info = _get_memory_usage()
    extra = f" Current memory usage: {memory_info}." if memory_info else ""
    typer.secho(
        f"{operation} failed due to insufficient memory.{extra} {details}".strip(),
        fg=typer.colors.RED,
        err=True,
    )


@dataclass
class Document:
    id: str
    source: str
    title: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def create(source: str, title: str, content: str, metadata: Dict[str, Any] = None) -> "Document":
        """Create a Document with a stable unique ID and optional metadata."""
        return Document(
            id=str(uuid4()),
            source=source,
            title=title,
            content=content,
            metadata=metadata or {},
        )

    @property
    def word_count(self) -> int:
        return len(self.content.split())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "source": self.source,
            "title": self.title,
            "content": self.content,
            "metadata": self.metadata,
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "Document":
        metadata = dict(data.get("metadata", {}))
        headers = [h.strip() for h in data.get("headers", []) if isinstance(h, str) and h.strip()]
        if headers:
            metadata.setdefault("headers", headers)
            metadata.setdefault("book", headers[0])
            if len(headers) > 1:
                metadata.setdefault("chapter", headers[1])
            if len(headers) > 2:
                metadata.setdefault("verse", headers[2])

        return Document(
            id=data.get("id", str(uuid4())),
            source=data.get("source", data.get("url", "")),
            title=data.get("title", ""),
            content=data.get("content", data.get("body", "")),
            metadata=metadata,
        )


def _iter_json_items(path: str) -> Iterator[dict]:
    with open(path, "r", encoding="utf-8") as f:
        first_char = f.read(1)
        if not first_char:
            return
        f.seek(0)
        if first_char == "[":
            data = json.load(f)
            yield from data
        else:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)


def _is_text_document(item: dict) -> bool:
    body = str(item.get("content", item.get("body", "")))
    if not body.strip():
        return False

    if body.startswith("%PDF") or body.startswith("\x89PNG") or body.startswith("GIF8"):
        return False

    body_len = len(body)
    if body_len == 0:
        return False

    non_printable = 0
    for ch in body:
        o = ord(ch)
        if o < 32 and o not in (10, 13, 9):
            non_printable += 1
            if non_printable / body_len > 0.05:
                return False
    return True


def load_documents_from_json(path: str) -> List["Document"]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input file not found: {path}")

    try:
        documents: List[Document] = []
        skipped = 0
        for item in _iter_json_items(path):
            if not _is_text_document(item):
                title = item.get("title") or item.get("url", "<unknown>")
                typer.secho(
                    f"Skipping non-text document during load: {title}",
                    fg=typer.colors.YELLOW,
                )
                skipped += 1
                continue
            documents.append(Document.from_dict(item))

        if not documents:
            raise ValueError("No valid text documents were found in the input file.")

        if skipped:
            typer.secho(f"Skipped {skipped} non-text document(s).", fg=typer.colors.YELLOW)

        return documents
    except MemoryError:
        _log_memory_error(
            "Loading document JSON",
            "Try using a smaller dataset or increase available memory.",
        )
        raise MemoryError("Failed to load document JSON into memory")


def _normalize(text: str) -> str:
    return unicodedata.normalize("NFKD", text).lower().strip()


def build_document_entry(
    chunk_id: str,
    document_id: str,
    content: str,
    path: List[str],
    metadata: Dict[str, Any],
    score_value: float,
    chunk_k: int,
) -> dict:
    """Create a new document entry from chunk data."""
    source = metadata.get("source", "")
    title = metadata.get("title", "")
    book = metadata.get("book")
    chapter = metadata.get("chapter")
    verse = metadata.get("verse")
    section = metadata.get("section")
    headers = [h for h in metadata.get("headers", []) if isinstance(h, str) and h.strip()]

    if not book and headers:
        book = headers[0]
    if not chapter and len(headers) > 1:
        chapter = headers[1]
    if not verse and len(headers) > 2:
        verse = headers[2]

    normalized_path = [_normalize(p) for p in path if p]

    return {
        "id": document_id,
        "source": source,
        "title": title,
        "book": book,
        "chapter": chapter,
        "verse": verse,
        "section": section,
        "path": path,
        "location": {k: v for k, v in {"book": book, "chapter": chapter, "verse": verse}.items() if v},
        "metadata": {k: v for k, v in metadata.items() if k not in {"source", "title", "chunk_index", "book", "chapter", "verse", "section"}},
        "score": score_value,
        "best_chunk": content,
        "chunks": [(score_value, content)],
    }


def update_document_entry(
    entry: dict,
    content: str,
    score_value: float,
    chunk_k: int,
) -> None:
    """Update an existing document entry with a new chunk."""
    if len(entry["chunks"]) < chunk_k:
        entry["chunks"].append((score_value, content))
    if score_value > entry["score"]:
        entry["score"] = score_value
        entry["best_chunk"] = content


def deduplicate_chunks(chunks: List[Tuple[float, str]]) -> List[dict]:
    """Remove duplicate chunks and convert to dict format."""
    chunks = sorted(chunks, key=lambda x: x[0], reverse=True)
    seen: Set[str] = set()
    deduped: List[Tuple[float, str]] = []
    for score_value, content in chunks:
        normalized = " ".join(content.lower().split())
        if normalized not in seen:
            seen.add(normalized)
            deduped.append((score_value, content))
    return [{"score": sv, "text": c} for sv, c in deduped]
    