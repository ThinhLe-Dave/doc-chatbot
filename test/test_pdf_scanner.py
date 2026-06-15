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


class MockChapterPage:
    def __init__(self, page_num, has_chapter=None):
        self.page_num = page_num
        self.has_chapter = has_chapter

    def extract_text(self):
        if self.has_chapter == "Chapter 1":
            return f"Chapter 1 Content on page {self.page_num}."
        elif self.has_chapter == "Chapter 2":
            return f"Chapter 2 Content on page {self.page_num}."
        return f"Regular content on page {self.page_num}."


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

    @patch("datacollector.pdf_scanner.PdfReader")
    def test_scan_pdf_uses_chapter_filter_when_chapters_provided(self, mock_reader_cls):
        mock_reader = MagicMock()
        mock_reader.pages = [
            MockChapterPage(1, has_chapter="Chapter 1"),
            MockChapterPage(2, has_chapter="Chapter 1"),
            MockChapterPage(3, has_chapter="Chapter 2"),
            MockChapterPage(4),
        ]
        mock_reader_cls.return_value = mock_reader

        scanner = PDFScanner()
        docs = scanner.scan_pdf(self.pdf_path, chapters=["Chapter 1"])

        self.assertEqual(len(docs), 2)
        self.assertEqual(docs[0].metadata["page"], 1)
        self.assertEqual(docs[1].metadata["page"], 2)

    @patch("datacollector.pdf_scanner.PdfReader")
    def test_scan_pdf_uses_page_range_filter(self, mock_reader_cls):
        mock_reader = MagicMock()
        mock_reader.pages = [MockPage(), MockPage(), MockPage(), MockPage()]
        mock_reader_cls.return_value = mock_reader

        scanner = PDFScanner()
        docs = scanner.scan_pdf(self.pdf_path, page_range="1,3")

        self.assertEqual(len(docs), 2)
        self.assertEqual(docs[0].metadata["page"], 1)
        self.assertEqual(docs[1].metadata["page"], 3)

    @patch("datacollector.pdf_scanner.PdfReader")
    def test_scan_pdf_combines_chapter_and_page_range_filters(self, mock_reader_cls):
        mock_reader = MagicMock()
        mock_reader.pages = [
            MockChapterPage(1, has_chapter="Chapter 1"),
            MockChapterPage(2, has_chapter="Chapter 1"),
            MockChapterPage(3, has_chapter="Chapter 2"),
            MockChapterPage(4, has_chapter="Chapter 2"),
        ]
        mock_reader_cls.return_value = mock_reader

        scanner = PDFScanner()
        docs = scanner.scan_pdf(self.pdf_path, chapters=["Chapter 1"], page_range="1-2")

        self.assertEqual(len(docs), 2)
        self.assertEqual(docs[0].metadata["page"], 1)
        self.assertEqual(docs[1].metadata["page"], 2)

    @patch("datacollector.pdf_scanner.PdfReader")
    def test_scan_pdf_no_chapters_matched_scans_all(self, mock_reader_cls):
        mock_reader = MagicMock()
        mock_reader.pages = [MockPage()]
        mock_reader_cls.return_value = mock_reader

        scanner = PDFScanner()
        docs = scanner.scan_pdf(self.pdf_path, chapters=["Nonexistent Chapter"])

        self.assertEqual(len(docs), 1)

    def test_parse_page_ranges_parses_single_pages(self):
        result = _parse_page_ranges("1,3,5", 10)
        self.assertEqual(result, {1, 3, 5})

    def test_parse_page_ranges_parses_ranges(self):
        result = _parse_page_ranges("1-3,5-7", 10)
        self.assertEqual(result, {1, 2, 3, 5, 6, 7})

    def test_parse_page_ranges_clamps_to_total_pages(self):
        result = _parse_page_ranges("1-100", 5)
        self.assertEqual(result, {1, 2, 3, 4, 5})

    def test_parse_page_ranges_returns_none_for_empty(self):
        result = _parse_page_ranges("", 10)
        self.assertIsNone(result)

    def test_parse_page_ranges_handles_invalid_input(self):
        result = _parse_page_ranges("abc,xyz", 10)
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()