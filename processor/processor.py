import json
import os
import re
from typing import List, Optional

import numpy as np
import typer

from chunker.document import load_documents_from_json, deduplicate_chunks
from chunker.chunker import get_chunk_file_path
from embedding.embedding import embed_texts, get_embedding_model
from utils.logging import debug
from vector_store.store import StaleCacheError, VectorStore


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
        typer.secho(f"   +{len(chunks) - 1} more matched chunks", fg=typer.colors.DIM)
    typer.echo("")


def _extract_primary_text(chunks: list, item: dict) -> str:
    if chunks and isinstance(chunks[0], dict):
        return chunks[0].get("text", "")
    return (item.get("best_chunk") or "").strip()


def display_results(results: List[dict], as_json: bool = False) -> None:
    if as_json:
        typer.echo(json.dumps(results, ensure_ascii=False, indent=2))
        return

    if not results:
        typer.secho("No relevant documents found.", fg=typer.colors.YELLOW)
        return

    for index, item in enumerate(results, start=1):
        _format_single_result(index, item)


def build_chunk_cache(input_file: str, output_file: Optional[str] = None) -> tuple:
    from chunker.chunker import write_chunks_to_file

    debug(f"build_chunk_cache input={input_file} output={output_file}", "processor")
    documents = load_documents_from_json(input_file)
    debug(f"loaded {len(documents)} documents", "processor")
    if not documents:
        raise ValueError("No documents found in the input file.")

    output_path = output_file or get_chunk_file_path(input_file)
    total_chunks = write_chunks_to_file(documents, output_path)

    if total_chunks == 0:
        raise ValueError("No chunks were created from the loaded documents.")

    debug("loading embedding model", "embedding.model")
    model = get_embedding_model()
    debug("building embedding cache via VectorStore", "vector.store")
    store = VectorStore(output_path)
    store.build(model)

    return total_chunks, output_path


def recommend_documents(
    query: str,
    input_file: str,
    top_k: int = 10,
    chunk_k: int = 3,
    min_score: float = 0.01,
    hybrid: bool = True,
    hybrid_weight: float = 0.4,
) -> List[dict]:
    hybrid_weight = max(0.0, min(1.0, hybrid_weight))
    debug(f"recommend query={query!r} input={input_file} top_k={top_k} chunk_k={chunk_k} min_score={min_score} hybrid={hybrid}", "processor")
    chunk_file = get_chunk_file_path(input_file)
    debug(f"chunk_file={chunk_file} exists={os.path.exists(chunk_file)}", "processor")
    if not os.path.exists(chunk_file):
        raise FileNotFoundError(f"Chunk file not found: {chunk_file}. Run 'build-chunks' first.")

    store = VectorStore(chunk_file)
    try:
        store.load()
    except (FileNotFoundError, StaleCacheError):
        debug("rebuilding stale cache", "processor")
        chunk_count, _ = build_chunk_cache(input_file, chunk_file)
        typer.secho(f"Rebuilt {chunk_count} chunks in {chunk_file}", fg=typer.colors.GREEN)
        store = VectorStore(chunk_file)
        store.load()

    if store.chunk_count == 0:
        raise ValueError("Could not load any chunks from the cache.")

    debug("encoding query embedding", "embedding.encode")
    query_embedding = embed_texts(get_embedding_model(), [query])[0].astype(np.float32)

    top_k_chunks = min(max(top_k * chunk_k * 8, top_k * chunk_k), store.chunk_count)
    search_results = store.search(query_embedding, top_k=top_k_chunks, min_score=0.0)

    candidate_indices = {r.chunk_index for r in search_results}
    scores = np.zeros(store.chunk_count, dtype=np.float32)
    for r in search_results:
        scores[r.chunk_index] = r.score

    query_terms = set(re.findall(r'\w+', query.lower()))
    debug(f"query_terms={sorted(query_terms)} hybrid={hybrid} hybrid_weight={hybrid_weight}", "processor")
    debug(f"candidate_indices={len(candidate_indices)}", "processor")

    document_chunks = _gather_candidate_chunks(
        chunk_file, candidate_indices, scores, query_terms, hybrid, hybrid_weight, min_score, chunk_k
    )

    for doc in document_chunks.values():
        debug(f"dedupe doc_id={doc.get('id')} chunks_before={len(doc['chunks'])}", "processor")
        doc["chunks"] = deduplicate_chunks(doc["chunks"])

    results = sorted(document_chunks.values(), key=lambda item: item["score"], reverse=True)
    final = [item for item in results if item["score"] >= min_score][:top_k]
    debug(f"results final count={len(final)}", "processor")
    return final


def _compute_keyword_scores(
    chunk_file: str,
    candidate_indices: set,
    query_terms: set,
) -> dict:
    from chunker.chunker import iter_chunk_batches_by_indices

    keyword_scores: dict = {}
    batch_iter = iter_chunk_batches_by_indices(chunk_file, candidate_indices, batch_size=128)
    for index, chunk in batch_iter:
        text = (chunk.content + " " + chunk.metadata.get("title", "")).lower()
        matched = sum(1 for term in query_terms if term in text)
        keyword_scores[index] = matched / len(query_terms) if query_terms else 0.0
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


def _gather_candidate_chunks(
    chunk_file: str,
    candidate_indices: set,
    scores: np.ndarray,
    query_terms: set,
    hybrid: bool,
    hybrid_weight: float,
    min_score: float,
    chunk_k: int,
) -> dict:
    from chunker.chunker import iter_chunk_batches_by_indices

    document_chunks: dict = {}
    keyword_scores: dict = {}

    if hybrid and query_terms:
        keyword_scores = _compute_keyword_scores(chunk_file, candidate_indices, query_terms)

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

    return document_chunks
