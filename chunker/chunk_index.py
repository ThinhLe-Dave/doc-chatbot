from __future__ import annotations

import json
import os
from typing import Dict, Iterator, List, Optional, Set, Tuple


def get_chunk_index_path(chunk_file: str) -> str:
    base, _ = os.path.splitext(chunk_file)
    return f"{base}_index.jsonl"


def get_chunk_offsets_path(chunk_file: str) -> str:
    base, _ = os.path.splitext(chunk_file)
    return f"{base}_offsets.jsonl"


def _is_json_array(path: str) -> bool:
    try:
        with open(path, "r", encoding="utf-8") as f:
            first_char = f.read(1)
    except Exception:
        return False
    return first_char == "["


def _iter_chunk_offsets(path: str) -> Iterator[Tuple[int, int]]:
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split(",")
                if len(parts) == 2:
                    yield int(parts[0]), int(parts[1])


def build_chunk_offsets(chunk_file: str) -> str:
    offsets_file = get_chunk_offsets_path(chunk_file)
    os.makedirs(os.path.dirname(offsets_file) or ".", exist_ok=True)

    if _is_json_array(chunk_file):
        with open(offsets_file, "w", encoding="utf-8") as f:
            f.write("0,-1\n")
        return offsets_file

    with open(chunk_file, "r", encoding="utf-8") as f:
        with open(offsets_file, "w", encoding="utf-8") as out:
            offset = 0
            for line in f:
                stripped = line.strip()
                if stripped:
                    out.write(f"{offset},{len(line.encode('utf-8'))}\n")
                offset += len(line.encode("utf-8"))

    return offsets_file


def load_chunk_offsets(chunk_file: str) -> Optional[List[Tuple[int, int]]]:
    offsets_file = get_chunk_offsets_path(chunk_file)
    if not os.path.exists(offsets_file):
        return None
    try:
        return list(_iter_chunk_offsets(offsets_file))
    except Exception:
        return None


def build_chunk_index(chunk_file: str) -> str:
    """Create a line-delimited metadata index for fast candidate reconstruction.

    Each line contains::

        {"index": <0-based order>, "id": "...", "document_id": "...", "path": [...], "metadata": {...}}
    """
    index_file = get_chunk_index_path(chunk_file)
    os.makedirs(os.path.dirname(index_file) or ".", exist_ok=True)

    from chunker.chunker import iter_chunks_from_json

    with open(index_file, "w", encoding="utf-8") as f:
        for index, chunk in enumerate(iter_chunks_from_json(chunk_file)):
            f.write(
                json.dumps(
                    {
                        "index": index,
                        "id": chunk.id,
                        "document_id": chunk.document_id,
                        "path": chunk.path,
                        "metadata": chunk.metadata,
                    },
                    ensure_ascii=False,
                )
            )
            f.write("\n")

    return index_file


def iter_chunk_index(path: str) -> Iterator[dict]:
    if not os.path.exists(path):
        return

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_chunk_index(path: str) -> List[dict]:
    try:
        return list(iter_chunk_index(path))
    except Exception:
        return []


def build_metadata_index(index_records: List[dict]) -> Dict[str, dict]:
    index: Dict[str, dict] = {}
    for record in index_records:
        index[record["id"]] = record
    return index
