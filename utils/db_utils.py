import hashlib
import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vector_store.db_store import PostgresVectorStore
    from chunker.chunker import Chunk



SQL_CREATE_DOCUMENTS_TABLE = """
    CREATE TABLE IF NOT EXISTS documents (
        id TEXT PRIMARY KEY,
        source TEXT,
        title TEXT,
        path JSONB,
        metadata JSONB
    )
"""

SQL_CREATE_CHUNKS_TABLE = """
    CREATE TABLE IF NOT EXISTS chunks (
        id TEXT PRIMARY KEY,
        document_id TEXT REFERENCES documents(id),
        content TEXT,
        path JSONB,
        metadata JSONB
    )
"""

SQL_CREATE_EMBEDDINGS_TABLE_TEMPLATE = "CREATE TABLE IF NOT EXISTS embeddings (chunk_id TEXT PRIMARY KEY REFERENCES chunks(id), embedding VECTOR({dim}))"

SQL_UPSERT_DOCUMENT = """
    INSERT INTO documents (id, source, title, path, metadata)
    VALUES (%s, %s, %s, %s, %s)
    ON CONFLICT (id) DO UPDATE SET
        source = EXCLUDED.source,
        title = EXCLUDED.title,
        path = EXCLUDED.path,
        metadata = EXCLUDED.metadata
"""

SQL_UPSERT_DOCUMENT_METADATA = """
    UPDATE documents
    SET metadata = jsonb_set(
        COALESCE(metadata, '{}'),
        %s,
        to_jsonb(%s),
        true
    )
    WHERE id = %s
"""

SQL_INSERT_DOCUMENT = SQL_UPSERT_DOCUMENT

SQL_INSERT_CHUNK = """
    INSERT INTO chunks (id, document_id, content, path, metadata)
    VALUES (%s, %s, %s, %s, %s)
    ON CONFLICT (id) DO UPDATE SET
        content = EXCLUDED.content,
        path = EXCLUDED.path,
        metadata = EXCLUDED.metadata
"""

SQL_INSERT_EMBEDDING = """
    INSERT INTO embeddings (chunk_id, embedding)
    VALUES (%s, %s)
    ON CONFLICT (chunk_id) DO UPDATE SET
        embedding = EXCLUDED.embedding
"""

SQL_DROP_TABLES = "DROP TABLE IF EXISTS embeddings, chunks, documents CASCADE"

SQL_CREATE_JOBS_TABLE = """
    CREATE TABLE IF NOT EXISTS jobs (
        id TEXT PRIMARY KEY,
        job_type TEXT,
        status TEXT,
        progress INTEGER,
        message TEXT,
        error TEXT,
        result JSONB,
        created_at TIMESTAMPTZ DEFAULT now(),
        updated_at TIMESTAMPTZ DEFAULT now(),
        started_at TIMESTAMPTZ,
        finished_at TIMESTAMPTZ,
        input_payload JSONB
    )
"""

SQL_UPSERT_JOB = """
    INSERT INTO jobs (id, job_type, status, progress, message, error, result, started_at, finished_at, input_payload)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT (id) DO UPDATE SET
        status = EXCLUDED.status,
        progress = EXCLUDED.progress,
        message = EXCLUDED.message,
        error = EXCLUDED.error,
        result = EXCLUDED.result,
        started_at = COALESCE(jobs.started_at, EXCLUDED.started_at),
        finished_at = EXCLUDED.finished_at,
        updated_at = now(),
        input_payload = EXCLUDED.input_payload
"""

SQL_GET_JOB = "SELECT id, job_type, status, progress, message, error, result, created_at, updated_at, started_at, finished_at, input_payload FROM jobs WHERE id = %s"

SQL_RECENT_JOBS = "SELECT id, job_type, status, progress, message, error, result, started_at, finished_at FROM jobs ORDER BY created_at DESC LIMIT %s"

SQL_COUNT_CHUNKS = "SELECT COUNT(*) FROM chunks"

SQL_GET_CHUNK_BY_ID = "SELECT id, document_id, content, path, metadata FROM chunks WHERE id = %s"

SQL_GET_CHUNKS_BY_IDS = "SELECT id, document_id, content, path, metadata FROM chunks WHERE id = ANY(%s)"

SQL_GET_CHUNK_CONTENT = "SELECT content FROM chunks WHERE id = %s"

SQL_SEARCH_SIMILAR = """
    SELECT c.id, c.document_id, embedding <#> %s::vector AS score
    FROM embeddings e
    JOIN chunks c ON e.chunk_id = c.id
    ORDER BY score ASC
    LIMIT %s
"""

SQL_SEARCH_SIMILAR_WITH_CATEGORIES = """
    SELECT c.id, c.document_id, embedding <#> %s::vector AS score
    FROM embeddings e
    JOIN chunks c ON e.chunk_id = c.id
    WHERE EXISTS (
        SELECT 1
        FROM jsonb_array_elements_text(c.metadata->'categories') AS cat
        WHERE lower(cat) = ANY(%s::text[])
    )
    ORDER BY score ASC
    LIMIT %s
"""


def create_tables(conn, embedding_dim: int) -> None:
    """Create tables if they don't exist."""
    with conn.cursor() as cur:
        cur.execute(SQL_CREATE_DOCUMENTS_TABLE)
        cur.execute(SQL_CREATE_CHUNKS_TABLE)
        cur.execute(SQL_CREATE_EMBEDDINGS_TABLE_TEMPLATE.format(dim=embedding_dim))
        conn.commit()


def search_similar(cur, query_embedding, top_k: int):
    """Search for similar chunks without category filtering."""
    cur.execute(
        SQL_SEARCH_SIMILAR,
        (query_embedding.tolist(), top_k * 4)
    )
    return cur.fetchall()


def search_similar_with_categories(cur, query_embedding, categories, top_k: int):
    """Search for similar chunks with category filtering."""
    cur.execute(
        SQL_SEARCH_SIMILAR_WITH_CATEGORIES,
        (query_embedding.tolist(), categories, top_k * 4)
    )
    return cur.fetchall()


def get_chunks_by_ids(cur, chunk_ids):
    """Retrieve multiple chunks by their IDs."""
    cur.execute(SQL_GET_CHUNKS_BY_IDS, (list(chunk_ids),))
    return cur.fetchall()


def get_chunk_content(cur, chunk_id: str):
    """Retrieve content for a single chunk."""
    cur.execute(SQL_GET_CHUNK_CONTENT, (chunk_id,))
    return cur.fetchone()


def get_chunk_by_id(cur, chunk_id: str):
    """Retrieve chunk data by ID."""
    cur.execute(SQL_GET_CHUNK_BY_ID, (chunk_id,))
    return cur.fetchone()


def get_document_by_id(cur, document_id: str):
    """Retrieve document data by ID."""
    cur.execute(
        "SELECT id, source, title, path, metadata FROM documents WHERE id = %s",
        (document_id,)
    )
    return cur.fetchone()


def get_chunks_for_document(cur, document_id: str):
    """Retrieve chunks for a document."""
    cur.execute(
        "SELECT id, document_id, content, path, metadata FROM chunks WHERE document_id = %s ORDER BY id",
        (document_id,)
    )
    return cur.fetchall()


def get_chunks_for_chapter(cur, source_hash: str, book: str, chapter: str):
    """Retrieve chunks for a chapter."""
    cur.execute(
        """
        SELECT id, document_id, content, path, metadata 
        FROM chunks 
        WHERE metadata->>'source_hash' = %s 
          AND metadata->>'book' = %s 
          AND metadata->>'chapter' = %s 
        ORDER BY (metadata->>'page')::int, id
        """,
        (source_hash, book, chapter)
    )
    return cur.fetchall()


def count_chunks(cur) -> int:
    """Return total number of chunks in database."""
    cur.execute(SQL_COUNT_CHUNKS)
    return int(cur.fetchone()[0])


def drop_tables(cur) -> None:
    """Drop all tables."""
    cur.execute(SQL_DROP_TABLES)


def store_chunk_batch(conn, chunks, model) -> None:
    """Store a batch of chunks and embeddings."""
    import sys
    from embedding.embedding import embed_texts

    if not chunks:
        return
    embeddings = embed_texts(model, [c.content for c in chunks])
    if embeddings.shape[0] != len(chunks):
        raise ValueError(f"Embedding row count mismatch: rows={embeddings.shape[0]} chunks={len(chunks)}")

    with conn.cursor() as cur:
        cur.executemany(
            SQL_INSERT_CHUNK,
            [
                (
                    chunk.id,
                    chunk.document_id,
                    chunk.content,
                    json.dumps(chunk.path),
                    json.dumps(chunk.metadata),
                )
                for chunk in chunks
            ],
        )
        embedding_rows = [
            (chunk.id, embedding.tolist())
            for chunk, embedding in zip(chunks, embeddings)
        ]
        cur.executemany(
            SQL_INSERT_EMBEDDING,
            embedding_rows,
        )


def compute_content_hash(content: str) -> str:
    normalized = " ".join((content or "").split()).encode("utf-8", errors="ignore")
    return hashlib.sha256(normalized).hexdigest()