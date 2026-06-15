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
    SELECT c.id, c.document_id, embedding <=> %s::vector AS score
    FROM embeddings e
    JOIN chunks c ON e.chunk_id = c.id
    ORDER BY score ASC
    LIMIT %s
"""

SQL_SEARCH_SIMILAR_WITH_CATEGORIES = """
    SELECT c.id, c.document_id, embedding <=> %s::vector AS score
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


def insert_document(cur, doc_id: str, source: str, title: str, path: list, metadata: dict) -> None:
    """Upsert a document record."""
    cur.execute(SQL_UPSERT_DOCUMENT, (doc_id, source, title, json.dumps(path), json.dumps(metadata)))


def upsert_document(cur, doc_id: str, source: str, title: str, path: list, metadata: dict) -> None:
    """Alias for insert_document with upsert behavior."""
    insert_document(cur, doc_id, source, title, path, metadata)


def update_document_metadata(cur, doc_id: str, keys: list, values: list) -> None:
    cursor = cur
    if not isinstance(keys, (list, tuple)) or len(keys) != 1:
        raise ValueError("update_document_metadata currently supports a single dotted key.")
    cursor.execute(SQL_UPSERT_DOCUMENT_METADATA, (keys[0], values[0], doc_id))


def insert_chunk(cur, chunk_id: str, document_id: str, content: str, path: list, metadata: dict) -> None:
    """Insert a chunk record."""
    cur.execute(SQL_INSERT_CHUNK, (chunk_id, document_id, content, json.dumps(path), json.dumps(metadata)))


def insert_embedding(cur, chunk_id: str, embedding) -> None:
    """Insert an embedding record."""
    cur.execute(SQL_INSERT_EMBEDDING, (chunk_id, embedding.tolist()))


def store_chunk_with_embedding(conn, chunk: "Chunk", embedding) -> None:
    """Store a chunk and its embedding in a single transaction."""
    with conn.cursor() as cur:
        cur.execute(SQL_INSERT_CHUNK, (chunk.id, chunk.document_id, chunk.content, json.dumps(chunk.path), json.dumps(chunk.metadata)))
        cur.execute(SQL_INSERT_EMBEDDING, (chunk.id, embedding.tolist()))


def store_chunk_batch(conn, chunks: list, model) -> None:
    """Store a batch of chunks and embeddings."""
    import sys
    from embedding.embedding import embed_texts

    if not chunks:
        return
    print(f"[debug] store_chunk_batch: starting with {len(chunks)} chunks", flush=True)
    embeddings = embed_texts(model, [c.content for c in chunks])
    print(f"[debug] store_chunk_batch: got embeddings shape={embeddings.shape}", flush=True)
    
    print(f"[debug] store_chunk_batch: starting DB insert", flush=True)
    with conn.cursor() as cur:
        print(f"[debug] store_chunk_batch: executing SQL_INSERT_CHUNK for {len(chunks)} rows", flush=True)
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
        print(f"[debug] store_chunk_batch: SQL_INSERT_CHUNK done, executing SQL_INSERT_EMBEDDING", flush=True)
        cur.executemany(
            SQL_INSERT_EMBEDDING,
            [
                (chunk.id, embedding.tolist())
                for chunk, embedding in zip(chunks, embeddings)
            ],
        )
        print(f"[debug] store_chunk_batch: SQL_INSERT_EMBEDDING done", flush=True)


def compute_content_hash(content: str) -> str:
    normalized = " ".join((content or "").split()).encode("utf-8", errors="ignore")
    return hashlib.sha256(normalized).hexdigest()