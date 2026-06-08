import json
import os
import re
from typing import Optional, Tuple, List

import numpy as np
import typer

from chunker.document import (
    load_documents_from_json,
    deduplicate_chunks,
)
from chunker.chunker import (
    count_chunks_in_json,
    gather_candidate_chunks,
    get_chunk_file_path,
    write_chunks_to_file,
)
from embedding.embedding import (
    embed_texts,
    load_or_build_embeddings,
    save_embeddings,
    build_embedding_cache,
    get_embedding_model,
)


def _debug(msg: str) -> None:
    if os.environ.get("DOC_DEBUG") == "1":
        typer.secho(f"[DEBUG] {msg}", fg=typer.colors.YELLOW, dim=True)


def build_chunk_cache(input_file: str, output_file: Optional[str] = None) -> Tuple[int, str]:
    _debug(f"build_chunk_cache input_file={input_file} output_file={output_file}")
    documents = load_documents_from_json(input_file)
    _debug(f"loaded {len(documents)} documents")
    if not documents:
        raise ValueError("No documents found in the input file.")

    output_path = output_file or get_chunk_file_path(input_file)
    total_chunks = write_chunks_to_file(documents, output_path)

    if total_chunks == 0:
        raise ValueError("No chunks were created from the loaded documents.")

    _debug("loading embedding model")
    model = get_embedding_model()
    _debug("building embedding cache")
    chunk_embeddings, chunk_ids = build_embedding_cache(output_path, model)
    _debug(f"saving embeddings chunk_ids={len(chunk_ids)} embeddings_shape={getattr(chunk_embeddings, 'shape', None)}")
    save_embeddings(output_path, chunk_embeddings, chunk_ids)

    return total_chunks, output_path


def _truncate_preview(text: str, max_len: int = 220) -> str:
    """Truncate text to max_len with ellipsis if needed."""
    if len(text) <= max_len:
        return text
    return text[:max_len - 3] + "..."


def _format_single_result(index: int, item: dict) -> None:
    """Format and print a single search result."""
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
    """Extract primary text from chunks or best_chunk field."""
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
    _debug(f"recommend_documents query={query!r} input={input_file} top_k={top_k} chunk_k={chunk_k} min_score={min_score} hybrid={hybrid}")
    chunk_file = get_chunk_file_path(input_file)
    _debug(f"chunk_file={chunk_file} exists={os.path.exists(chunk_file)}")
    if not os.path.exists(chunk_file) or count_chunks_in_json(chunk_file) == 0:
        chunk_count, saved_path = build_chunk_cache(input_file, chunk_file)
        typer.secho(f"Saved {chunk_count} chunks to {saved_path}", fg=typer.colors.GREEN)

    if not os.path.exists(chunk_file) or count_chunks_in_json(chunk_file) == 0:
        raise ValueError("Could not generate any chunks from the loaded documents.")

    embeddings, chunk_ids = load_or_build_embeddings(chunk_file)

    _debug("encoding query embedding")
    query_embedding = embed_texts(get_embedding_model(), [query])[0].astype(np.float32)

    top_k_chunks = min(max(top_k * chunk_k * 8, top_k * chunk_k), embeddings.shape[0])
    scores = (embeddings @ query_embedding).astype(np.float32)

    candidate_indices = set(np.argpartition(-scores, top_k_chunks - 1)[:top_k_chunks].tolist())

    query_terms = set(re.findall(r'\w+', query.lower()))
    _debug(f"query={query!r} query_terms={sorted(query_terms)} hybrid={hybrid} hybrid_weight={hybrid_weight}")
    _debug(f"candidate_indices={len(candidate_indices)}")

    document_chunks = gather_candidate_chunks(
        chunk_file, candidate_indices, scores, query_terms, hybrid, hybrid_weight, min_score, chunk_k
    )

    for doc in document_chunks.values():
        _debug(f"dedupe doc_id={doc.get('id')} chunks_before={len(doc['chunks'])}")
        doc["chunks"] = deduplicate_chunks(doc["chunks"])

    results = sorted(document_chunks.values(), key=lambda item: item["score"], reverse=True)
    final = [item for item in results if item["score"] >= min_score][:top_k]
    _debug(f"results final count={len(final)}")
    return final