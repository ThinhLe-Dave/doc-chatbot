from __future__ import annotations

import json
import os
from typing import List, Optional

import numpy as np

from utils.logging import debug, split_stats, text_sample
from vector_store.index import SearchResult, VectorIndex, VectorStoreConfig


class StaleCacheError(Exception):
    pass


class VectorStore:
    def __init__(self, chunk_file: str):
        self.config = VectorStoreConfig(
            model_name="paraphrase-multilingual-MiniLM-L12-v2",
            chunk_file=chunk_file,
        )
        self._index: Optional[VectorIndex] = None
        self._chunk_ids: List[str] = []

    def load(self) -> "VectorStore":
        meta_path = self.config.meta_path
        debug(f"load chunk_file={self.config.chunk_file}", "vector.store")
        if not os.path.exists(meta_path):
            debug("meta missing", "vector.store")
            raise FileNotFoundError(f"Embedding metadata not found: {meta_path}")

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        if meta.get("model") != self.config.model_name:
            debug(f"model mismatch: expected={self.config.model_name} cached={meta.get('model')}", "vector.store")
            raise StaleCacheError(
                f"Model mismatch: expected {self.config.model_name}, cached {meta.get('model')}"
            )

        expected_dim = meta.get("dimension")
        memmap_path = self.config.memmap_path
        ids_path = self.config.ids_path

        if not os.path.exists(memmap_path) or not os.path.exists(ids_path):
            debug("missing memmap or ids file", "vector.store")
            raise FileNotFoundError("Embedding files missing")

        try:
            raw = np.memmap(memmap_path, dtype=np.float32, mode="r")
            if raw.size > 0 and expected_dim and raw.size % expected_dim == 0:
                embeddings = raw.reshape(-1, expected_dim)
                with open(ids_path, "r", encoding="utf-8") as f:
                    chunk_ids = json.load(f)
                if embeddings.shape[0] == len(chunk_ids):
                    debug(f"success shape={embeddings.shape} ids={len(chunk_ids)}", "vector.store")
                    self._index = VectorIndex(embeddings, int(expected_dim))
                    self._chunk_ids = list(chunk_ids)
                    return self
        except Exception as exc:
            raise StaleCacheError(f"Failed to load embedding cache: {exc}") from exc

        debug("invalid cache", "vector.store")
        raise StaleCacheError("Embedding cache is invalid")

    def build(self, model, batch_size: int = 64) -> "VectorStore":
        from chunker.chunker import count_chunks_in_json, get_chunk_file_path, iter_chunk_batches
        from embedding.embedding import embed_texts, get_embedding_model

        chunk_file = self.config.chunk_file
        debug(f"build chunk_file={chunk_file} batch_size={batch_size}", "vector.store")
        if not os.path.exists(chunk_file):
            raise FileNotFoundError(f"Chunk file not found: {chunk_file}")

        total_chunks = count_chunks_in_json(chunk_file)
        debug(f"build total_chunks={total_chunks}", "vector.store")
        if total_chunks <= 0:
            raise ValueError("No chunks found to build embeddings")


        embedding_dim = model.get_embedding_dimension()
        debug(f"build embedding_dim={embedding_dim}", "vector.store")
        memmap_path = self.config.memmap_path
        ids_path = self.config.ids_path
        meta_path = self.config.meta_path

        os.makedirs(os.path.dirname(memmap_path) or ".", exist_ok=True)

        if os.path.exists(memmap_path):
            os.remove(memmap_path)

        try:
            embeddings = np.memmap(
                memmap_path,
                dtype=np.float32,
                mode="w+",
                shape=(total_chunks, embedding_dim),
            )
        except (MemoryError, OSError) as exc:
            raise MemoryError(f"Failed to allocate embedding storage: {exc}") from exc

        chunk_ids: List[str] = []
        offset = 0
        for batch_index, batch in enumerate(iter_chunk_batches(chunk_file, batch_size), start=1):
            batch_texts = [chunk.content for chunk in batch]
            print(f"[debug] vector_store.build: batch {batch_index} size={len(batch)} offset_before={offset}", flush=True)
            for index, chunk in enumerate(batch[:5], start=1):
                print(
                    f"[debug] vector_store.build: batch_chunk {index}/{len(batch)} id={chunk.id} "
                    f"chars={len(chunk.content)} {split_stats(chunk.content)} "
                    f"preview={text_sample(chunk.content)!r}",
                    flush=True,
                )
            batch_embeddings = embed_texts(model, batch_texts, batch_size=batch_size)
            embeddings[offset : offset + len(batch)] = batch_embeddings
            chunk_ids.extend(chunk.id for chunk in batch)
            offset += len(batch)
            debug(f"build batch {batch_index} offset={offset}", "vector.store")

            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

        embeddings.flush()
        del embeddings

        with open(ids_path, "w", encoding="utf-8") as f:
            json.dump(chunk_ids, f, ensure_ascii=False)

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({"model": self.config.model_name, "dimension": int(embedding_dim)}, f)

        return self.load()

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        min_score: float = 0.0,
    ) -> List[SearchResult]:
        if self._index is None:
            raise RuntimeError("VectorStore not loaded")

        results = self._index.query(query_embedding, top_k, min_score)
        for result in results:
            if result.chunk_index < len(self._chunk_ids):
                result.chunk_id = self._chunk_ids[result.chunk_index]
        return results

    def get_chunk_id(self, chunk_index: int) -> str:
        if chunk_index < len(self._chunk_ids):
            return self._chunk_ids[chunk_index]
        return ""

    @property
    def chunk_count(self) -> int:
        if self._index is None:
            return 0
        return int(self._index.embeddings.shape[0])
