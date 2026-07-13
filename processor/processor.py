import base64
import hashlib
import json
import os
import re
import threading
from typing import Any, Dict, Iterator, List, Mapping, Optional, Set, Tuple, Union

import numpy as np
import typer

from chunker.document import deduplicate_chunks
from chunker.chunker import Chunk
from embedding.embedding import embed_texts, get_embedding_model
from utils.logging import debug, error
from vector_store.db_store import PostgresVectorStore, PostgresVectorStoreError
from vector_store.db_config import DatabaseConfig
from vector_store.store import VectorStore
from utils.db_utils import get_chunks_by_ids

_CACHED_CHUNK_STORE: Optional[Union[PostgresVectorStore, VectorStore]] = None
_CACHED_STORE_TYPE: Optional[str] = None
_CACHED_STORE_KEY: Optional[str] = None
# Serializes access to the shared retrieval store / DB connection so requests
# offloaded to worker threads by the web layer do not use it concurrently.
_STORE_LOCK = threading.Lock()
from utils.config import (
    get_search_top_k,
    get_search_chunk_k,
    get_search_min_score,
    get_search_hybrid,
    get_search_hybrid_weight,
    get_graph_expansion_hops,
    get_graph_decay,
)


_NUMERIC_TAIL_RE = re.compile(r"/(\d+)$")
_CONTEXT_WINDOW = 1
_CHAPTER_ID_PREFIX = "chapter:"
_SOURCE_PAGE_RE = re.compile(r"#page=\d+$", re.IGNORECASE)
_TITLE_PAGE_RE = re.compile(r"\s*\(page\s*\d+\)$", re.IGNORECASE)


def _build_chapter_id(meta: dict) -> Optional[str]:
    book = meta.get("book")
    chapter = meta.get("chapter")
    if not book or not chapter:
        return None
    source_hash = meta.get("source_hash") or ""
    source = _SOURCE_PAGE_RE.sub('', meta.get("source") or "")
    payload = json.dumps({"h": source_hash, "s": source, "b": book, "c": chapter}).encode()
    return _CHAPTER_ID_PREFIX + base64.urlsafe_b64encode(payload).decode()


def _get_document_id_for_row(row: Tuple[Any, ...]) -> Optional[str]:
    document_id = row[1] if len(row) > 1 else None
    if document_id:
        return str(document_id)
    metadata = row[4] if len(row) > 4 else {}
    return _get_document_id(metadata)


def _get_document_id(metadata: Any) -> Optional[str]:
    if not isinstance(metadata, Mapping):
        return None

    for key in ("document_id", "id"):
        value = metadata.get(key)
        if value:
            return str(value)

    headers = metadata.get("headers")
    if isinstance(headers, list) and headers:
        normalized_headers = [str(header).strip() for header in headers if str(header).strip()]
        if normalized_headers:
            return "/".join(normalized_headers)

    for key in ("book", "chapter", "verse", "section"):
        value = metadata.get(key)
        if value:
            return str(value)

    return None


def _get_ordering_key(row: Tuple[Any, ...]) -> Tuple[Any, ...]:
    chunk_id = row[0]
    metadata = row[4] if len(row) > 4 else {}
    if not isinstance(metadata, Mapping):
        return (float("inf"), chunk_id)

    meta = metadata
    page = meta.get("page")
    chunk_index = meta.get("chunk_index")

    if page is not None:
        try:
            page_int = int(page)
            if chunk_index is not None:
                try:
                    return (page_int, int(chunk_index), chunk_id)
                except (TypeError, ValueError):
                    pass
            return (page_int, chunk_id)
        except (TypeError, ValueError):
            pass

    if chunk_index is not None:
        try:
            return (int(chunk_index), chunk_id)
        except (TypeError, ValueError):
            pass

    for key in ("verse", "section"):
        value = meta.get(key)
        if value is not None:
            try:
                return (int(value), chunk_id)
            except (TypeError, ValueError):
                continue

    path = metadata.get("path")
    if isinstance(path, list):
        last = path[-1] if path else ""
        match = _NUMERIC_TAIL_RE.search(f"/{last}")
        if match:
            try:
                return (int(match.group(1)), chunk_id)
            except (TypeError, ValueError):
                pass

    return (float("inf"), chunk_id)


def _connections_from_meta(metadata: Any) -> List[Tuple[str, float]]:
    if not isinstance(metadata, Mapping):
        return []
    conns = metadata.get("connected_chunk_ids")
    if not isinstance(conns, list):
        return []
    out: List[Tuple[str, float]] = []
    for entry in conns:
        if isinstance(entry, dict):
            cid = entry.get("chunk_id")
            weight = entry.get("weight", 1.0)
        elif isinstance(entry, str):
            cid = entry
            weight = 1.0
        else:
            continue
        if cid:
            out.append((str(cid), float(weight)))
    return out


def _bfs_graph_expand(
    seed_ids: Set[str],
    fetch_connections,
    expansion_hops: int,
    decay: float,
) -> Set[str]:
    """Breadth-first expand seed chunk ids along graph edges.

    `fetch_connections(chunk_id)` returns a list of (neighbor_id, weight) pairs.
    Edges with weight below `decay` are pruned so only meaningful connections
    propagate across hops.
    """
    visited: Set[str] = set(seed_ids)
    frontier: List[str] = list(seed_ids)
    for _ in range(max(0, expansion_hops)):
        if not frontier:
            break
        next_frontier: List[str] = []
        for cid in frontier:
            for neighbor_id, weight in fetch_connections(cid):
                if not neighbor_id or neighbor_id in visited:
                    continue
                if weight < decay:
                    continue
                visited.add(neighbor_id)
                next_frontier.append(neighbor_id)
        frontier = next_frontier
    return visited


_file_chunk_index_cache: Dict[str, Tuple[float, Dict[str, Any]]] = {}


def _get_file_chunk_index(chunk_file: str) -> Dict[str, Any]:
    """Return an id -> chunk map for a file store, cached by path and mtime.

    Avoids reparsing the whole chunk JSON on every retrieval request.
    """
    try:
        mtime = os.path.getmtime(chunk_file)
    except OSError:
        return {}
    cached = _file_chunk_index_cache.get(chunk_file)
    if cached and cached[0] == mtime:
        return cached[1]
    from chunker.chunker import load_chunks_from_json
    by_id = {c.id: c for c in load_chunks_from_json(chunk_file)}
    _file_chunk_index_cache[chunk_file] = (mtime, by_id)
    return by_id


def _get_chunk_file_from_store(store: VectorStore) -> Optional[str]:
    config = getattr(store, "config", None)
    if config is None:
        return None
    chunk_file = getattr(config, "chunk_file", None)
    if chunk_file and os.path.exists(chunk_file):
        return str(chunk_file)
    return None


def _expand_candidate_chunks(
    store: PostgresVectorStore,
    candidate_ids: Set[str],
) -> Set[str]:
    if not candidate_ids:
        return candidate_ids

    expansion_hops = get_graph_expansion_hops()
    decay = get_graph_decay()

    expanded = set(candidate_ids)
    try:
        with store._get_connection().cursor() as cur:
            rows = get_chunks_by_ids(cur, candidate_ids)
    except Exception as exc:
        debug(f"context expansion fetch failed: {exc}", "processor")
        return expanded

    doc_chunks: Dict[str, List[Tuple[Any, str]]] = {}
    for row in rows:
        chunk_id = row[0]
        doc_id = _get_document_id_for_row(row)
        if not doc_id:
            continue
        ordering = _get_ordering_key(row)
        doc_chunks.setdefault(doc_id, []).append((ordering, chunk_id))

    neighbor_ids: List[str] = []
    for doc_id, items in doc_chunks.items():
        items.sort()
        ids = [cid for _, cid in items]
        for position, cid in enumerate(ids):
            start = max(0, position - _CONTEXT_WINDOW)
            end = min(len(ids), position + _CONTEXT_WINDOW + 1)
            neighbor_ids.extend(ids[start:end])

    if neighbor_ids:
        expanded.update(neighbor_ids)

    try:
        conn = store._get_connection()
        with conn.cursor() as gcur:
            # Breadth-first graph expansion with one bulk query per hop
            # (instead of one query per neighbor) to avoid N+1 round-trips.
            visited: Set[str] = set(candidate_ids)
            frontier: List[str] = list(candidate_ids)
            for _ in range(max(0, expansion_hops)):
                if not frontier:
                    break
                next_frontier: List[str] = []
                for row in get_chunks_by_ids(gcur, frontier):
                    for neighbor_id, weight in _connections_from_meta(row[4]):
                        if not neighbor_id or neighbor_id in visited or weight < decay:
                            continue
                        visited.add(neighbor_id)
                        next_frontier.append(neighbor_id)
                frontier = next_frontier
        expanded.update(visited)
        debug(
            "graph expansion: seeds=%d hops=%d decay=%.2f -> %d chunks"
            % (len(candidate_ids), expansion_hops, decay, len(expanded)),
            "processor",
        )
    except Exception as exc:
        debug(f"graph context expansion failed: {exc}", "processor")

    return expanded


def _expand_candidate_chunks_file(
    chunk_ids_list: List[str],
    store: VectorStore,
) -> Set[str]:
    expanded = set(chunk_ids_list)
    try:
        chunk_ids_map = {cid: i for i, cid in enumerate(store._chunk_ids)}
        chunks_by_doc: Dict[str, List[Tuple[Any, str]]] = {}

        for cid in chunk_ids_list:
            idx = chunk_ids_map.get(cid)
            if idx is not None:
                doc_id = store._chunk_ids[idx].split("/")[0] if "/" in store._chunk_ids[idx] else store._chunk_ids[idx]
                ordering = (idx, cid)
                chunks_by_doc.setdefault(doc_id, []).append((ordering, cid))

        neighbor_ids: List[str] = []
        for items in chunks_by_doc.values():
            items.sort()
            ids = [cid for _, cid in items]
            for position, cid in enumerate(ids):
                start = max(0, position - _CONTEXT_WINDOW)
                end = min(len(ids), position + _CONTEXT_WINDOW + 1)
                neighbor_ids.extend(ids[start:end])

        if neighbor_ids:
            expanded.update(neighbor_ids)

        expansion_hops = get_graph_expansion_hops()
        decay = get_graph_decay()
        chunk_file = getattr(getattr(store, "config", None), "chunk_file", None)
        by_id = _get_file_chunk_index(chunk_file) if chunk_file else {}

        def fetch_connections(chunk_id: str) -> List[Tuple[str, float]]:
            chunk = by_id.get(chunk_id)
            if chunk is None:
                return []
            return _connections_from_meta(chunk.metadata)

        graph_expanded = _bfs_graph_expand(set(chunk_ids_list), fetch_connections, expansion_hops, decay)
        expanded.update(graph_expanded)
        debug(
            "graph (file) expansion: seeds=%d hops=%d decay=%.2f -> %d chunks"
            % (len(chunk_ids_list), expansion_hops, decay, len(expanded)),
            "processor",
        )
    except Exception as exc:
        debug(f"context expansion failed: {exc}", "processor")
    return expanded


def _get_chunks_by_ids_file(
    chunk_ids: Set[str],
    store: VectorStore,
) -> List["Chunk"]:
    result: List["Chunk"] = []
    chunk_file = _get_chunk_file_from_store(store)
    if not chunk_file:
        debug("file store chunk file not found", "processor")
        return result

    try:
        from chunker.chunker import load_chunks_from_json
        for chunk in load_chunks_from_json(chunk_file):
            if chunk.id in chunk_ids:
                result.append(chunk)
    except Exception as exc:
        debug(f"file chunk load failed: {exc}", "processor")
    return result


def _truncate_preview(text: str, max_len: int = 220) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len - 3] + "..."


def _format_single_result(index: int, item: dict) -> None:
    title = item.get("title") or "Untitled document"
    source = item.get("source", "")
    book = item.get("book")
    chapter = item.get("chapter")
    verse = item.get("verse")
    section = item.get("section")
    path = item.get("path") or []
    score = item.get("score", 0.0)
    chunks = item.get("chunks") or []
    primary = _extract_primary_text(chunks, item)
    metadata = item.get("metadata", {})
    categories = metadata.get("categories", []) or []

    typer.secho(f"{index}. {title}", fg=typer.colors.CYAN, bold=True)
    if source:
        typer.echo(f"   source: {source}")
    if book:
        typer.echo(f"   book: {book}")
    if chapter:
        typer.echo(f"   chapter: {chapter}")
    if verse:
        typer.echo(f"   verse: {verse}")
    if section:
        typer.echo(f"   section: {section}")
    if path:
        typer.echo(f"   path: {' > '.join(path)}")
    if categories:
        typer.echo(f"   categories: {', '.join(categories)}")
    typer.echo(f"   score: {score:.4f}")
    if metadata:
        typer.echo(f"   metadata: {metadata}")
    if primary:
        typer.echo(f"   excerpt: {_truncate_preview(primary)}")
    if len(chunks) > 1:
        typer.secho(f"   +{len(chunks) - 1} more matched chunks", fg=typer.colors.BLACK)
    debug(f"formatted result index={index} title={title!r} score={score:.4f} primary_excerpt={primary[:50]!r}", "processor")
    typer.echo("")


def _extract_primary_text(chunks: list, item: dict) -> str:
    if chunks and isinstance(chunks[0], dict):
        return chunks[0].get("text", "")
    return (item.get("best_chunk") or "").strip()


def display_results(results: List[dict], as_json: bool = False) -> None:
    import json
    if as_json:
        typer.echo(json.dumps(results, ensure_ascii=False, indent=2))
        return

    if not results:
        typer.secho("No relevant documents found.", fg=typer.colors.YELLOW)
        return

    for index, item in enumerate(results, start=1):
        _format_single_result(index, item)


def _resolve_chunk_store():
    global _CACHED_CHUNK_STORE, _CACHED_STORE_TYPE, _CACHED_STORE_KEY

    from pathlib import Path
    db_config = DatabaseConfig.from_config_file()
    if db_config.is_configured():
        store_key = f"postgres:{db_config.get_connection_string()}"
        if _CACHED_CHUNK_STORE is not None and _CACHED_STORE_KEY == store_key:
            return _CACHED_CHUNK_STORE, _CACHED_STORE_TYPE

        debug("loading from PostgresVectorStore", "db.store")
        store = PostgresVectorStore(config=db_config)
        store.load()
        _CACHED_CHUNK_STORE = store
        _CACHED_STORE_TYPE = "postgres"
        _CACHED_STORE_KEY = store_key
        return store, "postgres"

    chunk_file = Path(__file__).resolve().parent.parent / "database" / "pdf_data_chunks.json"
    store_key = f"file:{chunk_file}"
    if _CACHED_CHUNK_STORE is not None and _CACHED_STORE_KEY == store_key:
        return _CACHED_CHUNK_STORE, _CACHED_STORE_TYPE

    if chunk_file.exists():
        debug("loading from file-based VectorStore", "db.store")
        store = VectorStore(str(chunk_file))
        store.load()
        _CACHED_CHUNK_STORE = store
        _CACHED_STORE_TYPE = "file"
        _CACHED_STORE_KEY = store_key
        return store, "file"
    raise PostgresVectorStoreError("No database or embedding cache available")


def _encode_query(query: str) -> np.ndarray:
    debug("encoding query embedding", "embedding.encode")
    return embed_texts(get_embedding_model(), [query])[0].astype(np.float32)


def _search_and_score(
    store,
    query_embedding: np.ndarray,
    top_k: int,
    min_score: float,
    categories: Optional[List[str]],
    store_type: str = "postgres",
) -> tuple:
    top_k_chunks = min(max(top_k * 12, top_k), store.chunk_count)
    if store_type == "file":
        search_results = store.search(query_embedding, top_k=top_k_chunks, min_score=0.0)
    else:
        search_results = store.search(query_embedding, top_k=top_k_chunks, min_score=0.0, categories=categories)

    chunk_ids = {r.chunk_id for r in search_results}
    scores = {r.chunk_id: r.score for r in search_results}
    debug(f"candidate_indices={len(chunk_ids)}", "processor")
    return chunk_ids, scores


def _rank_results(
    candidate_ids: set,
    scores: dict,
    query: str,
    hybrid: bool,
    hybrid_weight: float,
    min_score: float,
    chunk_k: int,
    top_k: int,
    categories: Optional[List[str]],
    store=None,
    store_type: str = "postgres",
    expand: bool = True,
) -> List[dict]:
    query_terms = _extract_query_terms(query)
    debug(f"query_terms={sorted(query_terms)} hybrid={hybrid} hybrid_weight={hybrid_weight}", "processor")

    normalized_categories = {c.lower() for c in categories or []}
    document_chunks: dict = {}

    def _add_chunk_to_document(
        chunk_id: str,
        document_id: str,
        content: str,
        path: list,
        metadata: dict,
        score_value: float,
    ) -> None:
        entry = document_chunks.get(document_id)
        if entry is None:
            document_chunks[document_id] = _build_document_entry(
                chunk_id=chunk_id,
                document_id=document_id,
                content=content,
                path=path,
                metadata=metadata,
                score_value=score_value,
                chunk_k=chunk_k,
            )
            return

        entry["chunks"].append((score_value, content))
        if score_value <= entry["score"]:
            return

        entry["score"] = score_value
        entry["best_chunk"] = content
        entry["title"] = metadata.get("title", entry.get("title", ""))
        entry["source"] = metadata.get("source", entry.get("source", ""))
        entry["book"] = metadata.get("book", entry.get("book"))
        entry["chapter"] = metadata.get("chapter", entry.get("chapter"))
        entry["verse"] = metadata.get("verse", entry.get("verse"))
        entry["section"] = metadata.get("section", entry.get("section"))
        entry["path"] = path or entry.get("path", [])
        entry["location"] = {
            k: v
            for k, v in {
                "book": entry.get("book"),
                "chapter": entry.get("chapter"),
                "verse": entry.get("verse"),
                "section": entry.get("section"),
            }.items()
            if v
        }

    original_ids = set(candidate_ids)
    if store_type == "file":
        if expand:
            expanded_candidate_ids = _expand_candidate_chunks_file(list(candidate_ids), store)
        else:
            expanded_candidate_ids = set(original_ids)
        chunks = _get_chunks_by_ids_file(expanded_candidate_ids, store)
    else:
        if expand:
            expanded_candidate_ids = _expand_candidate_chunks(store, original_ids)
        else:
            expanded_candidate_ids = set(original_ids)
        with store._get_connection().cursor() as cur:
            rows = get_chunks_by_ids(cur, expanded_candidate_ids)
            chunks = [
                Chunk(
                    id=row[0],
                    document_id=row[1],
                    content=row[2],
                    path=row[3] if row[3] else [],
                    metadata=row[4] if row[4] else {},
                )
                for row in rows
            ]

    # Keyword (hybrid) scores are computed in-memory from the chunks we already
    # fetched, instead of a second store/connection and one query per chunk.
    keyword_scores = _keyword_scores_from_chunks(chunks, query_terms) if hybrid and query_terms else {}


    for chunk in chunks:
        chunk_meta_categories = [str(c).lower() for c in (chunk.metadata.get("categories") or [])]
        if normalized_categories and not normalized_categories.intersection(chunk_meta_categories):
            continue

        score_value = scores.get(chunk.id, 0.0)
        if hybrid and chunk.id in keyword_scores:
            score_value = (1.0 - hybrid_weight) * score_value + hybrid_weight * keyword_scores[chunk.id]

        if chunk.id in original_ids and score_value < min_score:
            continue

        meta = chunk.metadata or {}
        document_id = _build_chapter_id(meta) or chunk.document_id
        _add_chunk_to_document(
            chunk_id=chunk.id,
            document_id=document_id,
            content=chunk.content,
            path=chunk.path,
            metadata=chunk.metadata,
            score_value=score_value,
        )

    for doc in document_chunks.values():
        doc["chunks"] = deduplicate_chunks(doc["chunks"])
        if chunk_k and len(doc["chunks"]) > chunk_k:
            doc["chunks"] = sorted(doc["chunks"], key=lambda item: item.get("score", 0), reverse=True)[:chunk_k]
        if doc["chunks"]:
            doc["score"] = max(item.get("score", 0) for item in doc["chunks"])
            doc["best_chunk"] = clean_content(doc["chunks"][0].get("text", ""))

        if doc.get("id", "").startswith(_CHAPTER_ID_PREFIX):
            source = doc.get("source", "")
            if source:
                doc["source"] = _SOURCE_PAGE_RE.sub('', source)
            title = doc.get("title", "")
            if title:
                doc["title"] = _TITLE_PAGE_RE.sub('', title).strip()

    results = sorted(document_chunks.values(), key=lambda item: item["score"], reverse=True)
    final = [item for item in results if item["score"] >= min_score][:top_k]
    debug(f"results final count={len(final)}", "processor")
    return final


def recommend_documents(
    query: str,
    top_k: int = None,
    chunk_k: int = None,
    min_score: float = None,
    hybrid: bool = None,
    hybrid_weight: float = None,
    categories: Optional[List[str]] = None,
    expand: bool = True,
) -> List[dict]:
    top_k = top_k if top_k is not None else get_search_top_k()
    chunk_k = chunk_k if chunk_k is not None else get_search_chunk_k()
    min_score = min_score if min_score is not None else get_search_min_score()
    hybrid = hybrid if hybrid is not None else get_search_hybrid()
    hybrid_weight = hybrid_weight if hybrid_weight is not None else get_search_hybrid_weight()
    hybrid_weight = max(0.0, min(1.0, hybrid_weight))

    debug(f"recommend query={query!r} top_k={top_k} chunk_k={chunk_k} min_score={min_score} hybrid={hybrid} expand={expand} categories={categories}", "processor")

    # The retrieval store keeps a single shared DB connection, which is not
    # safe for concurrent use. Serialize retrieval so the FastAPI layer can
    # offload requests to worker threads without corrupting that connection.
    with _STORE_LOCK:
        store, store_type = _resolve_chunk_store()
        if store.chunk_count == 0:
            raise ValueError("Could not load any chunks from the database.")

        query_embedding = _encode_query(query)
        candidate_ids, scores = _search_and_score(store, query_embedding, top_k, min_score, categories, store_type)

        return _rank_results(
            candidate_ids, scores, query, hybrid, hybrid_weight, min_score, chunk_k, top_k, categories, store, store_type, expand
        )


def _extract_query_terms(query: str) -> set[str]:
    return {
        term
        for term in re.findall(r'\w+', query.lower())
        if len(term) > 2
    }


_OCR_SPACE_RE = re.compile(r'\b([a-z])\s+(\w+)', re.IGNORECASE)


def _normalize_ocr(text: str) -> str:
    return _OCR_SPACE_RE.sub(r'\1\2', text)


def _keyword_scores_from_chunks(chunks: List["Chunk"], query_terms: set) -> dict:
    """Compute keyword-overlap scores in-memory from already-fetched chunks.

    Avoids a second vector store, extra connection, and one query per chunk
    (the old N+1 path) on every hybrid chat request.
    """
    keyword_scores: dict = {}
    if not query_terms:
        return keyword_scores

    term_count = len(query_terms)
    # Precompile term patterns once instead of per chunk.
    term_patterns = [re.compile(rf"\b{re.escape(term)}\b") for term in query_terms]
    for chunk in chunks:
        metadata = chunk.metadata if isinstance(chunk.metadata, dict) else {}
        title = metadata.get("title") or ""
        normalized_text = _normalize_ocr((chunk.content + " " + title).lower())
        matched = sum(1 for pattern in term_patterns if pattern.search(normalized_text))
        keyword_scores[chunk.id] = matched / term_count
    return keyword_scores



def _build_document_entry(chunk_id, document_id, content, path, metadata, score_value, chunk_k):
    from chunker.document import build_document_entry
    return build_document_entry(
        chunk_id=chunk_id,
        document_id=document_id,
        content=content,
        path=path,
        metadata=metadata,
        score_value=score_value,
        chunk_k=chunk_k,
    )


def _update_document_entry(entry, content, score_value, chunk_k):
    from chunker.document import update_document_entry
    update_document_entry(entry, content, score_value, chunk_k)


def _strip_leading_ref(text: str, location: str) -> str:
    import re
    if not location:
        return text
    pattern = rf"^{re.escape(location)}\s*[-–—]?\s*"
    return re.sub(pattern, "", text, flags=re.IGNORECASE)


def _strip_tags(text: str) -> str:
    import re
    cleaned = re.sub(r"<environment_details>.*?</environment_details>", "", text, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"<environment_details>.*", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"<thinking>.*?</thinking>", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"<thinking>.*", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"<reasoning>.*?</reasoning>", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"<reasoning>.*", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    return cleaned


def clean_content(text: str, location: str = "") -> str:
    """Remove thinking/reasoning/environment tags and strip leading location reference."""
    if not text:
        return ""
    cleaned = _strip_tags(text)
    if location:
        cleaned = _strip_leading_ref(cleaned, location)
    return cleaned.strip()


def _get_generator():
    from generator.generator import generate_answer as _generate_answer
    return _generate_answer


def _extract_cited_sources(answer_text: str, sources: list, max_sources: int = 5) -> list:
    cited = set()
    source_refs = {}

    for i, s in enumerate(sources):
        book = s.get("book") or ""
        chapter = s.get("chapter") or ""
        verse = s.get("verse") or ""
        if book:
            ref = book
            if chapter:
                ref += f" {chapter}"
                if verse:
                    ref += f":{verse}"
            source_refs[ref] = i

        for chunk in s.get("chunks", []):
            chunk_text = chunk.get("text", "") if isinstance(chunk, dict) else ""
            chunk_ref_match = re.match(r"^([A-Za-z]+\s*\d+(?::\d+)?)", chunk_text)
            if chunk_ref_match:
                chunk_ref = chunk_ref_match.group(1)
                source_refs[chunk_ref] = i

    for match in re.findall(r"\[([A-Za-z]+\s*\d+(?::\d+)?)\]", answer_text):
        if match in source_refs:
            cited.add(source_refs[match])
        if match in source_refs:
            cited.add(source_refs[match])

    if not cited:
        return sources[:max_sources]
    return [sources[i] for i in sorted(cited)[:max_sources]]


def ask_question(
    query: str,
    top_k: int = None,
    chunk_k: int = None,
    min_score: float = None,
    hybrid: bool = None,
    hybrid_weight: float = None,
    categories: Optional[List[str]] = None,
    max_new_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    stream: bool = False,
    history: Optional[List[Dict[str, str]]] = None,
) -> dict:
    """
    Ask a question using RAG: retrieve documents and generate an answer.
    
    Args:
        query: The question to ask
        top_k: Number of documents to retrieve (default from config)
        chunk_k: Max chunks per document (default from config)
        min_score: Minimum score threshold (default from config)
        hybrid: Use hybrid scoring (default from config)
        hybrid_weight: Weight for hybrid scoring (default from config)
        categories: Filter by category (default None)
        max_new_tokens: Max response tokens (default from config)
        temperature: Sampling temperature (default from config)
        top_p: Top-p sampling (default from config)
        stream: Return streaming generator
        
    Returns:
        dict with 'query', 'answer', 'sources', and optionally 'stream'
    """
    from generator.generator import format_context

    results = recommend_documents(
        query=query,
        top_k=top_k,
        chunk_k=chunk_k,
        min_score=min_score,
        hybrid=hybrid,
        hybrid_weight=hybrid_weight,
        categories=categories,
    )
    
    context = format_context(results)
    generate_answer = _get_generator()
    
    effective_top_k = top_k or get_search_top_k()
    
    if stream:
        return {
            "query": query,
            "stream": generate_answer(
                query=query,
                context=context,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                stream=True,
                history=history,
            ),
            "sources": results[:effective_top_k],
        }
    
    answer = generate_answer(
        query=query,
        context=context,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        stream=False,
        history=history,
    )

    cited_sources = _extract_cited_sources(answer, results[:effective_top_k], max_sources=10)

    return {
        "query": query,
        "answer": answer,
        "sources": cited_sources,
    }