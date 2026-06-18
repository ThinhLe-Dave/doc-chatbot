import re
from typing import Any, Dict, Iterator, List, Mapping, Optional, Set, Tuple, Union

import numpy as np
import typer

from chunker.document import deduplicate_chunks
from chunker.chunker import Chunk
from embedding.embedding import embed_texts, get_embedding_model
from utils.logging import debug, error
from vector_store.db_store import PostgresVectorStore, PostgresVectorStoreError
from vector_store.db_config import DatabaseConfig
from utils.db_utils import SQL_GET_CHUNKS_BY_IDS, SQL_GET_CHUNK_CONTENT
from utils.config import (
    get_search_top_k,
    get_search_chunk_k,
    get_search_min_score,
    get_search_hybrid,
    get_search_hybrid_weight,
)


_NUMERIC_TAIL_RE = re.compile(r"/(\d+)$")
_CONTEXT_WINDOW = 3


def _get_document_id_for_row(row: Tuple[Any, ...]) -> Optional[str]:
    document_id = row[1] if len(row) > 1 else None
    if document_id:
        return str(document_id)
    metadata = row[4] if len(row) > 4 else {}
    return _get_document_id(metadata)


def _get_ordering_key(row: Tuple[Any, ...]) -> Tuple[Any, ...]:
    chunk_id = row[0]
    metadata = row[4] if len(row) > 4 else {}
    if not isinstance(metadata, Mapping):
        return (chunk_id,)

    meta = metadata
    chunk_index = meta.get("chunk_index")
    if chunk_index is not None:
        try:
            return (int(chunk_index), chunk_id)
        except (TypeError, ValueError):
            pass

    for key in ("page", "verse", "section"):
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

    return (chunk_id,)


def _expand_candidate_chunks(
    store: PostgresVectorStore,
    candidate_ids: Set[str],
) -> Set[str]:
    if not candidate_ids:
        return candidate_ids

    expanded = set(candidate_ids)
    try:
        store.load()
        with store._conn.cursor() as cur:
            cur.execute(
                SQL_GET_CHUNKS_BY_IDS,
                (list(candidate_ids),),
            )
            rows = cur.fetchall()
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

    return expanded


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
    db_config = DatabaseConfig.from_config_file()
    if not db_config.is_configured():
        raise PostgresVectorStoreError("Database not configured. Add [database] section to config.cfg")
    debug("loading from PostgresVectorStore", "db.store")
    store = PostgresVectorStore(config=db_config)
    store.load()
    return store


def _encode_query(query: str) -> np.ndarray:
    debug("encoding query embedding", "embedding.encode")
    return embed_texts(get_embedding_model(), [query])[0].astype(np.float32)


def _search_and_score(
    store: PostgresVectorStore,
    query_embedding: np.ndarray,
    top_k: int,
    min_score: float,
    categories: Optional[List[str]],
) -> tuple:
    top_k_chunks = min(max(top_k * 24, top_k), store.chunk_count)
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
) -> List[dict]:
    query_terms = _extract_query_terms(query)
    debug(f"query_terms={sorted(query_terms)} hybrid={hybrid} hybrid_weight={hybrid_weight}", "processor")

    normalized_categories = {c.lower() for c in categories or []}
    document_chunks: dict = {}

    if hybrid and query_terms:
        keyword_scores = _compute_keyword_scores(candidate_ids, query_terms)
    else:
        keyword_scores = {}

    db_config = DatabaseConfig.from_config_file()
    store = PostgresVectorStore(config=db_config)
    try:
        store.load()
        original_ids = set(candidate_ids)
        expanded_candidate_ids = _expand_candidate_chunks(store, original_ids)
        expanded_ids = expanded_candidate_ids - original_ids
        with store._conn.cursor() as cur:
            cur.execute(
                SQL_GET_CHUNKS_BY_IDS,
                (list(expanded_candidate_ids),)
            )
            for row in cur.fetchall():
                chunk = Chunk(
                    id=row[0],
                    document_id=row[1],
                    content=row[2],
                    path=row[3] if row[3] else [],
                    metadata=row[4] if row[4] else {},
                )

                chunk_meta_categories = [str(c).lower() for c in (chunk.metadata.get("categories") or [])]
                if normalized_categories and not normalized_categories.intersection(chunk_meta_categories):
                    continue

                score_value = scores.get(chunk.id, 0.0)
                if hybrid and chunk.id in keyword_scores:
                    score_value = (1.0 - hybrid_weight) * score_value + hybrid_weight * keyword_scores[chunk.id]

                if chunk.id in original_ids and score_value < min_score:
                    continue

                document_id = chunk.document_id
                entry = document_chunks.get(document_id)

                if entry is None:
                    entry = _build_document_entry(
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
                    entry["chunks"].append((score_value, chunk.content))
                    if score_value > entry["score"]:
                        entry["score"] = score_value
                        entry["best_chunk"] = chunk.content
                        meta = chunk.metadata or {}
                        entry["title"] = meta.get("title", entry.get("title", ""))
                        entry["source"] = meta.get("source", entry.get("source", ""))
                        entry["book"] = meta.get("book", entry.get("book"))
                        entry["chapter"] = meta.get("chapter", entry.get("chapter"))
                        entry["verse"] = meta.get("verse", entry.get("verse"))
                        entry["section"] = meta.get("section", entry.get("section"))
                        entry["path"] = chunk.path or entry.get("path", [])
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
    finally:
        store.close()

    for doc in document_chunks.values():
        doc["chunks"] = deduplicate_chunks(doc["chunks"])
        if chunk_k and len(doc["chunks"]) > chunk_k:
            doc["chunks"] = sorted(doc["chunks"], key=lambda item: item.get("score", 0), reverse=True)[:chunk_k]
        if doc["chunks"]:
            doc["score"] = max(item.get("score", 0) for item in doc["chunks"])
            doc["best_chunk"] = doc["chunks"][0].get("text", "")

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
) -> List[dict]:
    top_k = top_k if top_k is not None else get_search_top_k()
    chunk_k = chunk_k if chunk_k is not None else get_search_chunk_k()
    min_score = min_score if min_score is not None else get_search_min_score()
    hybrid = hybrid if hybrid is not None else get_search_hybrid()
    hybrid_weight = hybrid_weight if hybrid_weight is not None else get_search_hybrid_weight()
    hybrid_weight = max(0.0, min(1.0, hybrid_weight))

    db_config = DatabaseConfig.from_config_file()
    if not db_config.is_configured():
        raise PostgresVectorStoreError("Database not configured. Add [database] section to config.cfg")

    debug(f"recommend query={query!r} top_k={top_k} chunk_k={chunk_k} min_score={min_score} hybrid={hybrid} categories={categories}", "processor")

    store = _resolve_chunk_store()
    if store.chunk_count == 0:
        raise ValueError("Could not load any chunks from the database.")

    query_embedding = _encode_query(query)
    candidate_ids, scores = _search_and_score(store, query_embedding, top_k, min_score, categories)

    return _rank_results(
        candidate_ids, scores, query, hybrid, hybrid_weight, min_score, chunk_k, top_k, categories
    )


_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "he", "in",
    "is", "it", "its", "of", "on", "or", "should", "that", "the", "their", "this", "to", "use", "used", "was",
    "were", "will", "with", "how",
}


def _extract_query_terms(query: str) -> set[str]:
    return {
        term
        for term in re.findall(r'\w+', query.lower())
        if len(term) > 2 and term not in _STOPWORDS
    }


_OCR_SPACE_RE = re.compile(r'\b([a-z])\s+(\w+)', re.IGNORECASE)


def _normalize_ocr(text: str) -> str:
    return _OCR_SPACE_RE.sub(r'\1\2', text)


def _compute_keyword_scores(chunk_ids: set, query_terms: set) -> dict:
    keyword_scores: dict = {}
    if not query_terms:
        return keyword_scores

    db_config = DatabaseConfig.from_config_file()
    if db_config.is_configured():
        store = PostgresVectorStore(config=db_config)
        try:
            store.load()
            with store._conn.cursor() as cur:
                for chunk_id in chunk_ids:
                    cur.execute(SQL_GET_CHUNK_CONTENT, (chunk_id,))
                    row = cur.fetchone()
                    if row:
                        normalized_text = _normalize_ocr(row[0].lower())
                        matched = sum(
                            1
                            for term in query_terms
                            if re.search(rf"\b{re.escape(term)}\b", normalized_text)
                        )
                        keyword_scores[chunk_id] = matched / len(query_terms)
        finally:
            store.close()
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


def _get_generator():
    from generator.generator import generate_answer as _generate_answer
    return _generate_answer


def _extract_cited_sources(answer_text: str, sources: list, max_sources: int = 5) -> list:
    cited = set()
    for match in re.findall(r"\[(\d+)\]", answer_text):
        try:
            index = int(match) - 1
        except ValueError:
            continue
        if 0 <= index < len(sources):
            cited.add(index)
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
    )
    
    return {
        "query": query,
        "answer": answer,
        "sources": results[:effective_top_k],
    }