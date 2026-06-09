import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import app


class LowMemorySmokeTest(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.input_file = os.path.join(self.tempdir.name, "test_data.json")
        self.chunk_file = os.path.join(self.tempdir.name, "test_data_chunks.json")

        payload = [
            {
                "url": "https://example.com/test",
                "title": "Low Memory Test",
                "body": "This is a test document." * 100,
            }
        ]
        with open(self.input_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)

    def tearDown(self):
        self.tempdir.cleanup()

    def test_build_embedding_cache_handles_memory_error(self):
        from embedding.embedding import get_embedding_model
        from vector_store.store import VectorStore
        model = get_embedding_model()
        with open(self.chunk_file, "w", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "id": "doc_0_chunk_0",
                        "document_id": "doc_0",
                        "content": "test chunk content",
                        "path": [],
                        "metadata": {
                            "source": "https://example.com/test",
                            "title": "Low Memory Test",
                        },
                    }
                )
                + "\n"
            )

        store = VectorStore(self.chunk_file)
        with patch("vector_store.store.np.memmap", side_effect=MemoryError("Simulated low memory")):
            with self.assertRaises(MemoryError):
                store.build(model, batch_size=1)

    def test_search_handles_low_memory_encoding(self):
        with patch("embedding.embedding.SentenceTransformer.encode", side_effect=MemoryError("Simulated low memory")):
            with self.assertRaises(MemoryError):
                app.build_chunk_cache(self.input_file, self.chunk_file)


if __name__ == "__main__":
    unittest.main()
