import json
import os
from typing import List, Optional, Tuple

import numpy as np
try:
    import torch
except ImportError:
    torch = None

import typer
from sentence_transformers import SentenceTransformer

from chunker.chunker import (
    cache_chunk_ids,
    count_chunks_in_json,
    get_embedding_ids_path,
    get_embedding_matrix_path,
    get_model_meta_path,
    iter_chunk_batches,
    load_chunk_ids_from_cache,
    get_chunk_ids_from_json,
)
from chunker.document import _log_memory_error

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

MODEL: Optional[SentenceTransformer] = None
_MODEL_DIMENSION: Optional[int] = None


def _debug(msg: str) -> None:
    if os.environ.get("DOC_DEBUG") == "1":
        typer.secho(f"[DEBUG] {msg}", fg=typer.colors.YELLOW, dim=True)


def get_embedding_model() -> SentenceTransformer:
    global MODEL
    if MODEL is not None:
        _debug("embedding model cache hit")
        return MODEL

    device = "cpu"
    if torch is not None and torch.cuda.is_available():
        device = "cuda"

    try:
        _debug(f"loading embedding model ({MODEL_NAME}) on {device}")
        MODEL = SentenceTransformer(MODEL_NAME, device=device)
        return MODEL
    except Exception:
        _log_memory_error(
            f"Failed to load the embedding model ({MODEL_NAME}). Ensure sentence-transformers is installed and internet access is available.",
        )
        raise RuntimeError(f"Failed to load embedding model: {MODEL_NAME}")


def _get_cached_dimension(model: SentenceTransformer) -> int:
    global _MODEL_DIMENSION
    if _MODEL_DIMENSION is None:
        _debug("retrieving model embedding dimension")
        _MODEL_DIMENSION = model.get_embedding_dimension()
    return _MODEL_DIMENSION


def embed_texts(model: SentenceTransformer, texts: List[str], batch_size: int = 32) -> np.ndarray:
    if not texts:
        _debug("empty text list passed to embed_texts")
        return np.zeros((0, _get_cached_dimension(model)), dtype=np.float32)

    current_batch_size = batch_size
    while current_batch_size >= 1:
        try:
            _debug(f"encode batch_size={current_batch_size} text_count={len(texts)}")
            if torch is not None:
                with torch.no_grad():
                    embeddings = model.encode(
                        texts,
                        batch_size=current_batch_size,
                        convert_to_numpy=True,
                        normalize_embeddings=True,
                    )
            else:
                embeddings = model.encode(
                    texts,
                    batch_size=current_batch_size,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                )
            result = np.asarray(embeddings, dtype=np.float32)
            _debug(f"encoded shape={result.shape}")
            return result
        except MemoryError:
            if current_batch_size <= 1:
                break
            _debug(f"MemoryError at batch_size={current_batch_size}; halving")
            current_batch_size //= 2

    _log_memory_error(
        "Encoding texts ran out of memory",
        "Try a smaller batch size or reduce number of chunks.",
    )
    raise MemoryError("Embedding encoding ran out of memory")


def save_embeddings(chunk_file: str, embeddings: np.ndarray, chunk_ids: List[str]) -> None:
    _debug(f"save_embeddings chunk_ids={len(chunk_ids)}")
    matrix_file = get_embedding_matrix_path(chunk_file)
    ids_file = get_embedding_ids_path(chunk_file)
    meta_file = get_model_meta_path(chunk_file)
    os.makedirs(os.path.dirname(matrix_file) or ".", exist_ok=True)
    with open(ids_file, "w", encoding="utf-8") as f:
        json.dump(chunk_ids, f, ensure_ascii=False)
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump({"model": MODEL_NAME, "dimension": int(embeddings.shape[1])}, f)

    cache_chunk_ids(chunk_file, chunk_ids)


def load_embeddings(chunk_file: str) -> Tuple[Optional[np.ndarray], Optional[List[str]]]:
    _debug(f"load_embeddings chunk_file={chunk_file}")
    matrix_file = get_embedding_matrix_path(chunk_file)
    ids_file = get_embedding_ids_path(chunk_file)
    meta_file = get_model_meta_path(chunk_file)

    if not os.path.exists(meta_file):
        _debug("load_embeddings missing meta_file")
        return None, None

    try:
        with open(meta_file, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if meta.get("model") != MODEL_NAME:
            _debug("load_embeddings model mismatch")
            return None, None
        expected_dim = meta.get("dimension")
    except Exception as exc:
        _debug(f"load_embeddings meta read failed: {exc}")
        return None, None

    if not os.path.exists(matrix_file) or not os.path.exists(ids_file):
        _debug("load_embeddings missing matrix/ids")
        return None, None

    try:
        raw = np.memmap(matrix_file, dtype=np.float32, mode="r")
        if raw.size > 0 and expected_dim and raw.size % expected_dim == 0:
            embeddings = raw.reshape(-1, expected_dim)
            with open(ids_file, "r", encoding="utf-8") as f:
                chunk_ids = json.load(f)
            if embeddings.shape[0] == len(chunk_ids):
                _debug(f"load_embeddings loaded from cache shape={embeddings.shape}")
                return embeddings, chunk_ids
    except Exception as exc:
        _debug(f"load_embeddings mmap/interp failed: {exc}")

    return None, None


def build_embedding_cache(chunk_file: str, model: SentenceTransformer, batch_size: int = 64) -> Tuple[np.ndarray, List[str]]:
    total_chunks = count_chunks_in_json(chunk_file)
    _debug(f"build_embedding_cache total_chunks={total_chunks}")
    if total_chunks <= 0:
        raise ValueError("No chunks were found in the chunk cache to encode.")

    embedding_dim = _get_cached_dimension(model)
    matrix_file = get_embedding_matrix_path(chunk_file)
    ids_file = get_embedding_ids_path(chunk_file)
    meta_file = get_model_meta_path(chunk_file)
    os.makedirs(os.path.dirname(matrix_file) or ".", exist_ok=True)

    if os.path.exists(matrix_file):
        os.remove(matrix_file)

    try:
        _debug(f"allocating memmap shape=({total_chunks}, {embedding_dim})")
        embeddings = np.memmap(
            matrix_file,
            dtype=np.float32,
            mode="w+",
            shape=(total_chunks, embedding_dim),
        )
    except (MemoryError, OSError) as exc:
        _log_memory_error("Allocating embedding storage", str(exc))
        raise MemoryError(f"Failed to allocate embedding storage: {exc}") from exc

    chunk_ids: List[str] = []
    offset = 0
    for batch_index, batch in enumerate(iter_chunk_batches(chunk_file, batch_size), start=1):
        batch_texts = [chunk.content for chunk in batch]
        try:
            _debug(f"building cache batch {batch_index} size={len(batch_texts)}")
            batch_embeddings = embed_texts(model, batch_texts, batch_size=batch_size)
        except MemoryError:
            _log_memory_error(
                "Encoding chunk batch",
                "Try a smaller batch size or reduce number of chunks.",
            )
            raise

        embeddings[offset : offset + len(batch)] = batch_embeddings
        chunk_ids.extend(chunk.id for chunk in batch)
        offset += len(batch)

        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

    embeddings.flush()
    del embeddings

    _debug(f"build_embedding_cache chunk_ids={len(chunk_ids)}")
    with open(ids_file, "w", encoding="utf-8") as f:
        json.dump(chunk_ids, f, ensure_ascii=False)

    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump({"model": MODEL_NAME, "dimension": int(embedding_dim)}, f)

    result = np.memmap(matrix_file, dtype=np.float32, mode="r").reshape(-1, embedding_dim)
    _debug(f"build_embedding_cache result_shape={result.shape}")
    return result, chunk_ids


def load_or_build_embeddings(chunk_file: str) -> Tuple[np.ndarray, List[str]]:
    """Load embeddings from cache or rebuild if needed."""
    _debug("loading chunk ids and embeddings")
    chunk_ids = load_chunk_ids_from_cache(chunk_file)
    embeddings, saved_chunk_ids = load_embeddings(chunk_file)

    if chunk_ids is None:
        chunk_ids = get_chunk_ids_from_json(chunk_file)

    if embeddings is None or saved_chunk_ids != chunk_ids or embeddings.shape[0] != len(chunk_ids):
        model = get_embedding_model()
        chunk_embeddings, chunk_ids = build_embedding_cache(chunk_file, model)
        save_embeddings(chunk_file, chunk_embeddings, chunk_ids)
        embeddings = chunk_embeddings
    else:
        _debug("reusing cached embeddings")

    return embeddings, chunk_ids
