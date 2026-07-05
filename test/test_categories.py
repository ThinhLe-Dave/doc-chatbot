import sys
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, sys.path[0] + "/..")


class CategoryExtractionTest(unittest.TestCase):
    def test_build_categories_excludes_page_number(self):
        from chunker.chunker import Chunker
        
        chunker = Chunker()
        metadata = {"book": "GDPR", "page": 57}
        categories = chunker._build_categories(metadata, "test content")
        
        self.assertIn("GDPR", categories)
        self.assertNotIn("Page 57", categories)
        
    def test_build_categories_prioritizes_structural(self):
        from chunker.chunker import Chunker
        
        chunker = Chunker()
        metadata = {"book": "GDPR", "chapter": "Chapter 2", "section": "Section 5"}
        categories = chunker._build_categories(metadata, "test content")
        
        self.assertIn("GDPR", categories)
        self.assertIn("Chapter 2", categories)
        self.assertIn("Section 5", categories)
        self.assertEqual(len(categories), 3)
        
    def test_build_categories_falls_back_to_keywords(self):
        from chunker.chunker import Chunker
        
        chunker = Chunker()
        metadata = {}
        categories = chunker._build_categories(metadata, "privacy rights data protection")
        
        self.assertEqual(len(categories), 3)


if __name__ == "__main__":
    unittest.main()
