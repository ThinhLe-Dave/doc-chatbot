#!/bin/bash
set -e

# Initialize PostgreSQL database for Hugging Face Spaces
python3 -c "
from vector_store.db_config import DatabaseConfig
from utils.db_utils import SQL_CREATE_DOCUMENTS_TABLE, SQL_CREATE_CHUNKS_TABLE, SQL_CREATE_EMBEDDINGS_TABLE_TEMPLATE
import psycopg

try:
    cfg = DatabaseConfig.from_env()
    if cfg.is_configured():
        conn = psycopg.connect(cfg.get_connection_string())
        with conn.cursor() as cur:
            cur.execute('CREATE EXTENSION IF NOT EXISTS vector;')
            cur.execute(SQL_CREATE_DOCUMENTS_TABLE)
            cur.execute(SQL_CREATE_CHUNKS_TABLE)
            cur.execute(SQL_CREATE_EMBEDDINGS_TABLE_TEMPLATE.format(dim=cfg.embedding_dimension))
            conn.commit()
        conn.close()
        print('Database initialized successfully')
except Exception as e:
    print(f'Database init skipped: {e}')
"

exec "$@"