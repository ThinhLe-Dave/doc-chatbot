import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, sys.path[0] + "/..")


class FormatSingleResultTest(unittest.TestCase):
    def test_format_single_result_displays_categories(self):
        from processor.processor import _format_single_result
        
        item = {
            "title": "Test Document",
            "source": "test://doc",
            "categories": ["EU Law", "Privacy"],
            "score": 0.85,
            "chunks": [(0.85, "test chunk")],
            "metadata": {"categories": ["EU Law", "Privacy"]},
        }
        
        with patch("typer.secho") as mock_secho, patch("typer.echo") as mock_echo:
            _format_single_result(1, item)
            
            outputs = [call[0][0] for call in mock_echo.call_args_list]
            category_output = [o for o in outputs if "categories:" in o]
            self.assertEqual(len(category_output), 1)


class BuildDocumentEntryTest(unittest.TestCase):
    def test_build_document_entry_includes_categories(self):
        from chunker.document import build_document_entry
        
        entry = build_document_entry(
            chunk_id="chunk1",
            document_id="doc1",
            content="test content",
            path=["Book 1", "Chapter 1"],
            metadata={"source": "test://doc", "categories": ["law", "EU"]},
            score_value=0.75,
            chunk_k=3,
        )
        
        self.assertIn("categories", entry)
        self.assertEqual(entry["categories"], ["law", "EU"])


if __name__ == "__main__":
    unittest.main()
