import json
import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

sys.modules['bs4'] = MagicMock()
sys.modules['requests'] = MagicMock()

from datacollector.pdf_scanner import (
    PDFScanner,
    _clean_extracted_text,
    _looks_like_broken_pdf_text,
    _parse_page_ranges,
    compute_chapter_pages,
    preflight_chapters,
    scan_and_build_chunks,
)


class MockPage:
    def extract_text(self):
        return "This is test content from page. " * 10


class PDFScannerTest(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.pdf_path = os.path.join(self.tempdir.name, "test.pdf")
        self.output_path = os.path.join(self.tempdir.name, "output.json")
        # Create an empty file so os.path.exists returns True
        with open(self.pdf_path, "wb") as f:
            f.write(b"%PDF-1.4\n")

    def tearDown(self):
        self.tempdir.cleanup()

    @patch("datacollector.pdf_scanner.PdfReader")
    def test_scan_pdf_creates_documents(self, mock_reader_cls):
        mock_reader = MagicMock()
        mock_reader.pages = [MockPage(), MockPage()]
        mock_reader_cls.return_value = mock_reader

        scanner = PDFScanner()
        docs = scanner.scan_pdf(self.pdf_path)

        self.assertEqual(len(docs), 2)
        self.assertEqual(docs[0].metadata["page"], 1)
        self.assertEqual(docs[0].metadata["total_pages"], 2)

    def test_clean_extracted_text_joins_split_words(self):
        text = "The right t o erasure means r ight t o be f orgotten and EUR OPEAN P ARLIAMENT."

        self.assertTrue(_looks_like_broken_pdf_text(text))
        self.assertEqual(
            _clean_extracted_text(text),
            "The right to erasure means right to be forgotten and EUROPEAN PARLIAMENT.",
        )
        self.assertEqual(_clean_extracted_text("AI model under EU law."), "AI model under EU law.")
        self.assertEqual(_clean_extracted_text("REGUL A TIONS OF THE COUNCIL"), "REGULATIONS OF THE COUNCIL")

    @patch("datacollector.pdf_scanner.PdfReader")
    def test_scan_pdf_uses_ocr_when_pypdf_text_is_broken(self, mock_reader_cls):
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Right t o erasure."
        mock_reader = MagicMock()
        mock_reader.pages = [mock_page]
        mock_reader_cls.return_value = mock_reader

        scanner = PDFScanner(use_ocr=True)
        scanner._ocr_page = MagicMock(return_value=("Right to erasure.", 0.8))

        docs = scanner.scan_pdf(self.pdf_path)

        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0].content, "Right to erasure.")
        self.assertEqual(docs[0].metadata["extraction_method"], "ocr")

    @patch("datacollector.pdf_scanner.PdfReader")
    def test_export_to_json(self, mock_reader_cls):
        mock_reader = MagicMock()
        mock_reader.pages = [MockPage()]
        mock_reader_cls.return_value = mock_reader

        scanner = PDFScanner()
        scanner.scan_pdf(self.pdf_path)
        result_path = scanner.export_to_json(self.output_path)

        self.assertTrue(os.path.exists(result_path))
        with open(result_path, "r") as f:
            data = json.load(f)
        self.assertEqual(len(data), 1)

    @patch("datacollector.pdf_scanner.write_chunks_to_file")
    @patch("datacollector.pdf_scanner.PdfReader")
    def test_build_chunks(self, mock_reader_cls, mock_write_chunks):
        mock_reader = MagicMock()
        mock_reader.pages = [MockPage()]
        mock_reader_cls.return_value = mock_reader
        mock_write_chunks.return_value = 3

        scanner = PDFScanner()
        scanner.scan_pdf(self.pdf_path)
        chunk_count, saved_path = scanner.build_chunks()

        self.assertEqual(chunk_count, 3)


if __name__ == "__main__":
    unittest.main()