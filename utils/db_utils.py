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

SQL_INSERT_DOCUMENT = """
    INSERT INTO documents (id, source, title, path, metadata)
    VALUES (%s, %s, %s, %s, %s)
    ON CONFLICT (id) DO NOTHING
"""

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


def insert_document(cur, doc_id: str, source: str, title: str, path: list, metadata: dict) -> None:
    """Insert a document record."""
    cur.execute(SQL_INSERT_DOCUMENT, (doc_id, source, title, json.dumps(path), json.dumps(metadata)))


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
    from embedding.embedding import embed_texts
    embeddings = embed_texts(model, [c.content for c in chunks])
    with conn.cursor() as cur:
        for chunk, embedding in zip(chunks, embeddings):
            cur.execute(SQL_INSERT_CHUNK, (chunk.id, chunk.document_id, chunk.content, json.dumps(chunk.path), json.dumps(chunk.metadata)))
            cur.execute(SQL_INSERT_EMBEDDING, (chunk.id, embedding.tolist()))