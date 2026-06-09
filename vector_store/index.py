from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

from utils.logging import debug


def _get_embedding_ids_path(chunk_file: str) -> str:
    base, _ = os.path.splitext(chunk_file)
    return f"{base}_chunk_ids.json"


def _get_embedding_matrix_path(chunk_file: str) -> str:
    base, _ = os.path.splitext(chunk_file)
    return f"{base}_embeddings.npy"


def _get_model_meta_path(chunk_file: str) -> str:
    base, _ = os.path.splitext(chunk_file)
    return f"{base}_embeddings_meta.json"


@dataclass
class SearchResult:
    chunk_index: int
    chunk_id: str
    score: float


@dataclass
class VectorStoreConfig:
    model_name: str
    chunk_file: str
    memmap_path: str = field(init=False)
    ids_path: str = field(init=False)
    meta_path: str = field(init=False)

    def __post_init__(self) -> None:
        self.memmap_path = _get_embedding_matrix_path(self.chunk_file)
        self.ids_path = _get_embedding_ids_path(self.chunk_file)
        self.meta_path = _get_model_meta_path(self.chunk_file)


class VectorIndex:
    def __init__(self, embeddings: np.memmap, dimension: int):
        self.embeddings = embeddings
        self.dimension = dimension

    def query(
        self,
        query_vec: np.ndarray,
        top_k: int,
        min_score: float = 0.0,
    ) -> List[SearchResult]:
        debug(f"query vec_shape={query_vec.shape} top_k={top_k} min_score={min_score}", "vector.index")
        if self.embeddings.shape[0] == 0:
            debug("empty embeddings", "vector.index")
            return []

        scores = (self.embeddings @ query_vec).astype(np.float32)

        candidate_count = min(max(top_k * 4, top_k), scores.shape[0])
        if candidate_count >= scores.shape[0]:
            candidate_indices = np.arange(scores.shape[0])
        else:
            candidate_indices = np.argpartition(-scores, candidate_count - 1)[:candidate_count]

        results: List[SearchResult] = []
        for idx in candidate_indices:
            score = float(scores[idx])
            if score >= min_score:
                results.append(SearchResult(chunk_index=int(idx), chunk_id="", score=score))

        results.sort(key=lambda r: r.score, reverse=True)
        debug(f"query results={len(results)}", "vector.index")
        return results[:top_k]
