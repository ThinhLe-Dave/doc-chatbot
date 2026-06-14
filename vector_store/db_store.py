from __future__ import annotations

import json
import os
from typing import List, Optional

import numpy as np

from utils.logging import debug
from vector_store.db_config import DatabaseConfig
from vector_store.index import SearchResult
from utils.db_utils import (
    SQL_CREATE_DOCUMENTS_TABLE,
    SQL_CREATE_CHUNKS_TABLE,
    SQL_CREATE_EMBEDDINGS_TABLE_TEMPLATE,
    SQL_INSERT_DOCUMENT,
    SQL_INSERT_CHUNK,
    SQL_INSERT_EMBEDDING,
    SQL_SEARCH_SIMILAR,
    SQL_SEARCH_SIMILAR_WITH_CATEGORIES,
    SQL_GET_CHUNK_BY_ID,
    SQL_COUNT_CHUNKS,
)


class PostgresVectorStoreError(Exception):
    """Error for PostgreSQL vector store operations."""
    pass


class PostgresVectorStore:
    """PostgreSQL-backed vector store using pgvector for similarity search."""

    def __init__(self, config: Optional[DatabaseConfig] = None, chunk_file: Optional[str] = None):
        self.config = config or DatabaseConfig()
        self._conn = None
        self._chunk_file = chunk_file
        self._embedding_dimension = self.config.embedding_dimension

    def _get_connection(self):
        """Get or create database connection."""
        if self._conn is None:
            try:
                import psycopg
                self._conn = psycopg.connect(self.config.get_connection_string())
            except ImportError:
                raise PostgresVectorStoreError(
                    "psycopg not installed. Run: pip install psycopg[binary,pool]"
                )
        return self._conn

    def _ensure_tables(self) -> None:
        """Create tables if they don't exist."""
        conn = self._get_connection()
        with conn.cursor() as cur:
            cur.execute(SQL_CREATE_DOCUMENTS_TABLE)
            cur.execute(SQL_CREATE_CHUNKS_TABLE)
            cur.execute(SQL_CREATE_EMBEDDINGS_TABLE_TEMPLATE.format(dim=self._embedding_dimension))
            conn.commit()

    def load(self) -> "PostgresVectorStore":
        """Load vector index from database (no-op for DB, always ready)."""
        self._ensure_tables()
        return self

    def build(self, model, batch_size: int = 64) -> "PostgresVectorStore":
        """Build embeddings and store in database."""
        from chunker.chunker import count_chunks_in_json, iter_chunk_batches
        from embedding.embedding import embed_texts

        chunk_file = self._chunk_file
        if not chunk_file:
            raise PostgresVectorStoreError("No chunk file configured")

        if not os.path.exists(chunk_file):
            raise FileNotFoundError(f"Chunk file not found: {chunk_file}")

        total_chunks = count_chunks_in_json(chunk_file)
        debug(f"build total_chunks={total_chunks}", "db.store")
        if total_chunks <= 0:
            raise ValueError("No chunks found to build embeddings")

        self._ensure_tables()
        conn = self._get_connection()

        seen_docs = set()
        with conn.cursor() as cur:
            for batch_index, batch in enumerate(iter_chunk_batches(chunk_file, batch_size), start=1):
                batch_texts = [chunk.content for chunk in batch]
                batch_embeddings = embed_texts(model, batch_texts)

                for chunk, embedding in zip(batch, batch_embeddings):
                    if chunk.document_id not in seen_docs:
                        cur.execute(
                            SQL_INSERT_DOCUMENT,
                            (chunk.document_id, chunk.metadata.get("source", ""), chunk.metadata.get("title", ""), json.dumps(chunk.path), json.dumps(chunk.metadata)),
                        )
                        seen_docs.add(chunk.document_id)
                    cur.execute(
                        SQL_INSERT_CHUNK,
                        (chunk.id, chunk.document_id, chunk.content, json.dumps(chunk.path), json.dumps(chunk.metadata)),
                    )
                    cur.execute(
                        SQL_INSERT_EMBEDDING,
                        (chunk.id, embedding.tolist())
                    )

                conn.commit()
                debug(f"build batch {batch_index} processed", "db.store")

        return self

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        min_score: float = 0.0,
        categories: Optional[List[str]] = None,
    ) -> List[SearchResult]:
        """Search for similar chunks using cosine similarity."""
        conn = self._get_connection()
        normalized_categories = [str(category).lower() for category in categories or []]

        with conn.cursor() as cur:
            if normalized_categories:
                cur.execute(
                    SQL_SEARCH_SIMILAR_WITH_CATEGORIES,
                    (query_embedding.tolist(), normalized_categories, top_k * 4)
                )
            else:
                cur.execute(
                    SQL_SEARCH_SIMILAR,
                    (query_embedding.tolist(), top_k * 4)
                )
            results = []
            for row in cur.fetchall():
                chunk_id, document_id, distance = row
                score = float(1.0 - distance)
                if score >= min_score:
                    results.append(SearchResult(
                        chunk_index=0,
                        chunk_id=chunk_id,
                        score=score
                    ))
            results.sort(key=lambda r: r.score, reverse=True)
            return results[:top_k]

    def save_chunks_from_file(self, chunk_file: str) -> int:
        """Import chunks from JSONL file into database."""
        from chunker.chunker import iter_chunks_from_json

        conn = self._get_connection()
        count = 0
        seen_docs = set()

        with conn.cursor() as cur:
            for chunk in iter_chunks_from_json(chunk_file):
                if chunk.document_id not in seen_docs:
                    cur.execute(
                        SQL_INSERT_DOCUMENT,
                        (chunk.document_id, chunk.metadata.get("source", ""), chunk.metadata.get("title", ""), json.dumps(chunk.path), json.dumps(chunk.metadata)),
                    )
                    seen_docs.add(chunk.document_id)
                cur.execute(
                    SQL_INSERT_CHUNK,
                    (chunk.id, chunk.document_id, chunk.content, json.dumps(chunk.path), json.dumps(chunk.metadata)),
                )
                count += 1
            conn.commit()

        return count

    def get_chunks_by_ids(self, chunk_ids: set) -> List[dict]:
        """Retrieve multiple chunks by their IDs."""
        conn = self._get_connection()
        with conn.cursor() as cur:
            chunks = []
            for chunk_id in chunk_ids:
                cur.execute(
                    SQL_GET_CHUNK_BY_ID,
                    (chunk_id,)
                )
                row = cur.fetchone()
                if row:
                    chunks.append({
                        "id": row[0],
                        "document_id": row[1],
                        "content": row[2],
                        "path": row[3] if row[3] else [],
                        "metadata": row[4] if row[4] else {},
                    })
            return chunks

    def get_chunk(self, chunk_id: str) -> Optional[dict]:
        """Retrieve chunk data by ID."""
        for chunk in self.get_chunks_by_ids({chunk_id}):
            return chunk
        return None

    @property
    def chunk_count(self) -> int:
        """Return total number of chunks in database."""
        conn = self._get_connection()
        with conn.cursor() as cur:
            cur.execute(SQL_COUNT_CHUNKS)
            return int(cur.fetchone()[0])

    def close(self) -> None:
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None