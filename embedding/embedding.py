import json
import os
from typing import List, Optional

import numpy as np

try:
    import torch
except ImportError:
    torch = None

from sentence_transformers import SentenceTransformer

from chunker.chunker import count_chunks_in_json, get_chunk_file_path
from utils.config import get_hf_token

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

MODEL: Optional[SentenceTransformer] = None
_MODEL_DIMENSION: Optional[int] = None


def _get_hf_token() -> Optional[str]:
    token = get_hf_token()
    return token if token else None


def get_embedding_model() -> SentenceTransformer:
    global MODEL
    if MODEL is not None:
        print("[debug] model cache hit", flush=True)
        return MODEL

    device = "cpu"
    if torch is not None and torch.cuda.is_available():
        device = "cuda"

    try:
        print(f"[debug] loading model ({MODEL_NAME}) on {device}...", flush=True)
        MODEL = SentenceTransformer(MODEL_NAME, device=device)
        print("[debug] model loaded successfully", flush=True)
        return MODEL
    except Exception as e:
        print(f"[error] Failed to load model: {e}", flush=True)
        from chunker.document import _log_memory_error
        _log_memory_error(
            f"Failed to load the embedding model ({MODEL_NAME}). Ensure sentence-transformers is installed and internet access is available.",
        )
        raise RuntimeError(f"Failed to load embedding model: {MODEL_NAME}")


def _get_cached_dimension(model: SentenceTransformer) -> int:
    global _MODEL_DIMENSION
    if _MODEL_DIMENSION is None:
        print("[debug] retrieving model embedding dims", flush=True)
        _MODEL_DIMENSION = model.get_embedding_dimension()
    return _MODEL_DIMENSION


def embed_texts(model: SentenceTransformer, texts: List[str], batch_size: int = 32) -> np.ndarray:
    if not texts:
        print("[debug] empty text list passed to embed_texts", flush=True)
        return np.zeros((0, _get_cached_dimension(model)), dtype=np.float32)

    current_batch_size = batch_size
    while current_batch_size >= 1:
        try:
            print(f"[debug] encoding batch_size={current_batch_size} text_count={len(texts)}", flush=True)
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
            print(f"[debug] encoded shape={result.shape}", flush=True)
            return result
        except MemoryError:
            if current_batch_size <= 1:
                break
            print(f"[debug] MemoryError at batch_size={current_batch_size}; halving", flush=True)
            current_batch_size //= 2
        except Exception as e:
            print(f"[error] embed_texts error: {type(e).__name__}: {e}", flush=True)
            raise

    from chunker.document import _log_memory_error
    _log_memory_error(
        "Encoding texts ran out of memory",
        "Try a smaller batch size or reduce number of chunks.",
    )
    raise MemoryError("Embedding encoding ran out of memory")


def load_or_build_embeddings(chunk_file: str) -> tuple[np.ndarray, List[str]]:
    from vector_store.store import StaleCacheError, VectorStore

    print("[debug] loading cache via VectorStore", flush=True)
    store = VectorStore(chunk_file)
    try:
        store.load()
    except (FileNotFoundError, StaleCacheError):
        print("[debug] cache miss, rebuilding", flush=True)
        model = get_embedding_model()
        store.build(model)
    embeddings = store._index.embeddings if store._index else np.zeros((0, 0), dtype=np.float32)
    print(f"[debug] loaded cache shape={embeddings.shape} ids={len(store._chunk_ids)}", flush=True)
    return embeddings, store._chunk_ids
