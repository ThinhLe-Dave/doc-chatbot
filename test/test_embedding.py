import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class EmbeddingModelTest(unittest.TestCase):
    def test_get_embedding_model_returns_cached_model(self):
        from embedding.embedding import get_embedding_model, MODEL
        
        # Clear any cached model
        import embedding.embedding as emb_module
        emb_module.MODEL = None
        
        mock_model = MagicMock()
        mock_model.get_embedding_dimension.return_value = 384
        
        with patch("embedding.embedding.SentenceTransformer", return_value=mock_model):
            model1 = get_embedding_model()
            model2 = get_embedding_model()
            
            self.assertIs(model1, model2)
            
        # Restore for other tests
        emb_module.MODEL = None


class EmbedTextsTest(unittest.TestCase):
    def test_embed_texts_empty_list(self):
        from embedding.embedding import embed_texts
        
        mock_model = MagicMock()
        mock_model.get_embedding_dimension.return_value = 384
        
        result = embed_texts(mock_model, [])
        
        self.assertEqual(result.shape, (0, 384))
        
    def test_embed_texts_normal_batch(self):
        from embedding.embedding import embed_texts
        
        mock_model = MagicMock()
        mock_model.get_embedding_dimension.return_value = 384
        mock_model.encode.return_value = np.zeros((3, 384), dtype=np.float32)
        
        texts = ["text1", "text2", "text3"]
        result = embed_texts(mock_model, texts)
        
        self.assertEqual(result.shape, (3, 384))
        mock_model.encode.assert_called_once()
        
    def test_embed_texts_halves_batch_on_memory_error(self):
        from embedding.embedding import embed_texts
        
        mock_model = MagicMock()
        mock_model.get_embedding_dimension.return_value = 384
        
        call_count = [0]
        def mock_encode(texts, batch_size, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise MemoryError("OOM")
            return np.zeros((len(texts), 384), dtype=np.float32)
        
        mock_model.encode = mock_encode
        
        texts = ["text"] * 10
        result = embed_texts(mock_model, texts, batch_size=5)
        
        self.assertEqual(result.shape[0], 10)
        self.assertEqual(call_count[0], 2)


class StaleCacheErrorTest(unittest.TestCase):
    def test_stale_cache_error_is_exception(self):
        from vector_store.store import StaleCacheError
        
        with self.assertRaises(StaleCacheError):
            raise StaleCacheError("test error")


if __name__ == "__main__":
    unittest.main()
