import json
import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch, call

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class MockCursor:
    def __init__(self):
        self.executed = []
        
    def execute(self, sql, params=None):
        self.executed.append((sql, params))
        
    def fetchall(self):
        return []
    
    def fetchone(self):
        return None
    
    def __enter__(self):
        return self
        
    def __exit__(self, *args):
        pass


class MockConnection:
    def __init__(self):
        self.committed = 0
        self.cursor_obj = MockCursor()
        
    def cursor(self):
        return self.cursor_obj
        
    def commit(self):
        self.committed += 1
        
    def close(self):
        pass


class DatabaseConfigTest(unittest.TestCase):
    def test_db_config_from_defaults(self):
        from vector_store.db_config import DatabaseConfig
        from utils.config import DATABASE_DEFAULTS
        
        config = DatabaseConfig()
        self.assertEqual(config.host, DATABASE_DEFAULTS["host"])
        self.assertEqual(config.port, DATABASE_DEFAULTS["port"])
        self.assertEqual(config.name, DATABASE_DEFAULTS["name"])
        self.assertEqual(config.user, DATABASE_DEFAULTS["user"])
        self.assertEqual(config.password, DATABASE_DEFAULTS["password"])
        
    def test_db_config_connection_string(self):
        from vector_store.db_config import DatabaseConfig
        
        config = DatabaseConfig(host="localhost", port=5432, name="testdb", user="testuser", password="testpass")
        conn_str = config.get_connection_string()
        self.assertEqual(conn_str, "postgresql://testuser:testpass@localhost:5432/testdb")
        
    def test_db_config_with_url(self):
        from vector_store.db_config import DatabaseConfig
        
        config = DatabaseConfig(url="postgresql://user:pass@host:5432/mydb")
        conn_str = config.get_connection_string()
        self.assertEqual(conn_str, "postgresql://user:pass@host:5432/mydb")
        
    def test_db_config_is_configured_with_url(self):
        from vector_store.db_config import DatabaseConfig
        
        self.assertTrue(DatabaseConfig(url="postgresql://...").is_configured())
        
    def test_db_config_is_configured_with_host_name_user(self):
        from vector_store.db_config import DatabaseConfig
        
        self.assertTrue(DatabaseConfig(host="localhost", name="db", user="user").is_configured())


class PostgresVectorStoreTest(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.chunk_file = os.path.join(self.tempdir.name, "test_chunks.json")
        
        payload = [
            {
                "id": "test_doc_1_chunk_0",
                "document_id": "test_doc_1",
                "content": "Test chunk content for searching.",
                "path": ["Test Book", "Chapter 1"],
                "metadata": {"source": "test://doc1", "title": "Test Doc 1"},
            }
        ]
        with open(self.chunk_file, "w", encoding="utf-8") as f:
            for item in payload:
                f.write(json.dumps(item) + "\n")
                
    def tearDown(self):
        self.tempdir.cleanup()

    def test_db_utils_insert_document(self):
        from utils.db_utils import insert_document, SQL_INSERT_DOCUMENT
        
        mock_cur = MagicMock()
        insert_document(mock_cur, "doc1", "test://doc1", "Test", ["Book 1"], {"key": "value"})
        
        mock_cur.execute.assert_called_once()
        call_args = mock_cur.execute.call_args
        self.assertIn("INSERT INTO documents", call_args[0][0])
        
    def test_db_utils_store_chunk_batch(self):
        from utils.db_utils import store_chunk_batch
        from chunker.chunker import Chunk
        
        mock_model = MagicMock()
        
        chunks = [
            Chunk(id="chunk_1", document_id="doc_1", content="Content 1", path=[], metadata={"source": "test"})
        ]
        
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cur)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        
        mock_embedding = MagicMock()
        mock_embedding.tolist.return_value = [0.1] * 384
        
        with patch("embedding.embedding.embed_texts", return_value=[mock_embedding]):
            store_chunk_batch(mock_conn, chunks, mock_model)
        
        self.assertTrue(mock_cur.execute.called)

    def test_search_result_dataclass(self):
        from vector_store.index import SearchResult
        
        result = SearchResult(chunk_index=5, chunk_id="test_chunk", score=0.95)
        self.assertEqual(result.chunk_index, 5)
        self.assertEqual(result.chunk_id, "test_chunk")
        self.assertEqual(result.score, 0.95)

    @patch("builtins.open", create=True)
    @patch("os.path.exists")
    def test_store_load_raises_file_not_found(self, mock_exists, mock_open):
        from vector_store.store import VectorStore, StaleCacheError
        
        mock_exists.return_value = False
        
        store = VectorStore(self.chunk_file)
        with self.assertRaises(FileNotFoundError):
            store.load()


class SQLQueriesTest(unittest.TestCase):
    def test_sql_statements_exist(self):
        from utils.db_utils import (
            SQL_CREATE_DOCUMENTS_TABLE,
            SQL_CREATE_CHUNKS_TABLE,
            SQL_INSERT_DOCUMENT,
            SQL_INSERT_CHUNK,
            SQL_INSERT_EMBEDDING,
            SQL_SEARCH_SIMILAR,
        )
        
        self.assertIn("CREATE TABLE", SQL_CREATE_DOCUMENTS_TABLE)
        self.assertIn("CREATE TABLE", SQL_CREATE_CHUNKS_TABLE)
        self.assertIn("INSERT INTO documents", SQL_INSERT_DOCUMENT)
        self.assertIn("INSERT INTO chunks", SQL_INSERT_CHUNK)
        self.assertIn("INSERT INTO embeddings", SQL_INSERT_EMBEDDING)
        self.assertIn("embedding", SQL_SEARCH_SIMILAR)


class VectorIndexTest(unittest.TestCase):
    def test_vector_index_query_empty(self):
        from vector_store.index import VectorIndex
        import numpy as np
        
        embeddings = np.zeros((0, 384), dtype=np.float32)
        index = VectorIndex(embeddings, 384)
        
        query = np.zeros(384, dtype=np.float32)
        results = index.query(query, top_k=10)
        
        self.assertEqual(len(results), 0)
        
    def test_vector_index_query_returns_results(self):
        from vector_store.index import VectorIndex
        import numpy as np
        
        embeddings = np.zeros((3, 384), dtype=np.float32)
        embeddings[0, 0] = 1.0
        embeddings[1, 1] = 0.5
        embeddings[2, 2] = 0.3
        index = VectorIndex(embeddings, 384)
        
        query = np.zeros(384, dtype=np.float32)
        query[0] = 1.0
        
        results = index.query(query, top_k=3)
        
        self.assertEqual(len(results), 3)
        self.assertGreaterEqual(results[0].score, 0.9)


if __name__ == "__main__":
    unittest.main()