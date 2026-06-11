import re
from typing import List, Optional

import numpy as np
import typer

from chunker.document import deduplicate_chunks
from chunker.chunker import Chunk
from embedding.embedding import embed_texts, get_embedding_model
from utils.logging import debug
from vector_store.db_store import PostgresVectorStore, PostgresVectorStoreError
from vector_store.db_config import DatabaseConfig
from utils.db_utils import SQL_GET_CHUNKS_BY_IDS, SQL_GET_CHUNK_CONTENT

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
) -> tuple:
    top_k_chunks = min(max(top_k * 24, top_k), store.chunk_count)
    search_results = store.search(query_embedding, top_k=top_k_chunks, min_score=0.0)

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
    query_terms = set(re.findall(r'\w+', query.lower()))
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
        with store._conn.cursor() as cur:
            cur.execute(
                SQL_GET_CHUNKS_BY_IDS,
                (list(candidate_ids),)
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

                if score_value < min_score:
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
                    _update_document_entry(entry, chunk.content, score_value, chunk_k)
    finally:
        store.close()

    for doc in document_chunks.values():
        doc["chunks"] = deduplicate_chunks(doc["chunks"])

    results = sorted(document_chunks.values(), key=lambda item: item["score"], reverse=True)
    final = [item for item in results if item["score"] >= min_score][:top_k]
    debug(f"results final count={len(final)}", "processor")
    return final


def recommend_documents(
    query: str,
    top_k: int = 10,
    chunk_k: int = 3,
    min_score: float = 0.01,
    hybrid: bool = True,
    hybrid_weight: float = 0.4,
    categories: Optional[List[str]] = None,
) -> List[dict]:
    hybrid_weight = max(0.0, min(1.0, hybrid_weight))

    db_config = DatabaseConfig.from_config_file()
    if not db_config.is_configured():
        raise PostgresVectorStoreError("Database not configured. Add [database] section to config.cfg")

    debug(f"recommend query={query!r} top_k={top_k} chunk_k={chunk_k} min_score={min_score} hybrid={hybrid} categories={categories}", "processor")

    store = _resolve_chunk_store()
    if store.chunk_count == 0:
        raise ValueError("Could not load any chunks from the database.")

    query_embedding = _encode_query(query)
    candidate_ids, scores = _search_and_score(store, query_embedding, top_k, min_score)

    return _rank_results(
        candidate_ids, scores, query, hybrid, hybrid_weight, min_score, chunk_k, top_k, categories
    )


_OCR_SPACE_RE = re.compile(r'\b([a-z])\s+(\w+)', re.IGNORECASE)


def _normalize_ocr(text: str) -> str:
    return _OCR_SPACE_RE.sub(r'\1\2', text)


def _compute_keyword_scores(chunk_ids: set, query_terms: set) -> dict:
    keyword_scores: dict = {}
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
                        matched = sum(1 for term in query_terms if term in normalized_text)
                        keyword_scores[chunk_id] = matched / len(query_terms) if query_terms else 0.0
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