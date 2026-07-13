from __future__ import annotations

import json
import os
from typing import List, Optional

import numpy as np

from utils.logging import debug
from vector_store.db_config import DatabaseConfig
from vector_store.index import SearchResult
from utils.db_utils import (
    create_tables,
    insert_document,
    insert_chunk,
    insert_embedding,
    store_chunk_batch,
    search_similar,
    search_similar_with_categories,
    get_chunks_by_ids,
    get_chunk_by_id,
    count_chunks,
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
            self._connect()
            return self._conn

        closed = getattr(self._conn, "closed", None)
        if isinstance(closed, (bool, int)):
            if closed:
                debug("Connection closed, reconnecting...", "db.store")
                self._connect()
            return self._conn

        return self._conn

    def _connect(self) -> None:
        """Create a new database connection."""
        try:
            import psycopg
            self._conn = psycopg.connect(self.config.get_connection_string())
        except ImportError:
            raise PostgresVectorStoreError(
                "psycopg not installed. Run: pip install psycopg[binary,pool]"
            )

    def _ensure_tables(self) -> None:
        """Create tables if they don't exist."""
        conn = self._get_connection()
        create_tables(conn, self._embedding_dimension)

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
                        seen_docs.add(chunk.document_id)
                        insert_document(cur, chunk.document_id, chunk.metadata.get("source", ""), chunk.metadata.get("title", ""), chunk.path, chunk.metadata)
                    insert_chunk(cur, chunk.id, chunk.document_id, chunk.content, chunk.path, chunk.metadata)
                    insert_embedding(cur, chunk.id, embedding)

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
        """Search for similar chunks using cosine distance and return similarity scores."""
        conn = self._get_connection()
        normalized_categories = [str(category).lower() for category in categories or []]

        with conn.cursor() as cur:
            if normalized_categories:
                rows = search_similar_with_categories(cur, query_embedding, normalized_categories, top_k)
            else:
                rows = search_similar(cur, query_embedding, top_k)
            results = []
            for row in rows:
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
                    seen_docs.add(chunk.document_id)
                    insert_document(cur, chunk.document_id, chunk.metadata.get("source", ""), chunk.metadata.get("title", ""), chunk.path, chunk.metadata)
                insert_chunk(cur, chunk.id, chunk.document_id, chunk.content, chunk.path, chunk.metadata)
                count += 1
            conn.commit()

        return count

    def get_chunks_by_ids(self, chunk_ids: set) -> List[dict]:
        """Retrieve multiple chunks by their IDs."""
        conn = self._get_connection()
        with conn.cursor() as cur:
            rows = get_chunks_by_ids(cur, chunk_ids)
            return [
                {
                    "id": row[0],
                    "document_id": row[1],
                    "content": row[2],
                    "path": row[3] if row[3] else [],
                    "metadata": row[4] if row[4] else {},
                }
                for row in rows
            ]

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
            return count_chunks(cur)

    def close(self) -> None:
        """Close database connection."""
        if self._conn:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None