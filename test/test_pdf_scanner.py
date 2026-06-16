import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

sys.modules['bs4'] = MagicMock()
sys.modules['requests'] = MagicMock()

from datacollector.pdf_scanner import (
    PDFScanner,
    _parse_page_ranges,
    _extract_text_from_page,
    _get_pdf_reader,
    _get_allowed_pages,
    _pre_extract_pages,
    _build_page_metadata,
    _create_page_document,
    compute_chapter_pages,
    preflight_chapters,
    scan_and_build_chunks,
)


class MockPage:
    def get_text(self, mode="text"):
        return "This is test content from page. " * 10


class MockChapterPage:
    def __init__(self, page_num, has_chapter=None):
        self.page_num = page_num
        self.has_chapter = has_chapter

    def get_text(self, mode="text"):
        if self.has_chapter == "Chapter 1":
            return f"Chapter 1 Content on page {self.page_num}."
        elif self.has_chapter == "Chapter 2":
            return f"Chapter 2 Content on page {self.page_num}."
        return f"Regular content on page {self.page_num}."


class MockFitZDocument:
    def __init__(self, pages):
        self._pages = pages

    def __len__(self):
        return len(self._pages)

    def __getitem__(self, index):
        return self._pages[index]

    def close(self):
        pass


class PDFScannerTest(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.pdf_path = os.path.join(self.tempdir.name, "test.pdf")
        self.output_path = os.path.join(self.tempdir.name, "output.json")
        with open(self.pdf_path, "wb") as f:
            f.write(b"%PDF-1.4\n")

    def tearDown(self):
        self.tempdir.cleanup()

    @patch("datacollector.pdf_scanner.fitz.open")
    def test_scan_pdf_creates_documents(self, mock_fitz_open):
        mock_doc = MockFitZDocument([MockPage(), MockPage()])
        mock_fitz_open.return_value = mock_doc

        scanner = PDFScanner()
        docs = scanner.scan_pdf(self.pdf_path)

        self.assertEqual(len(docs), 2)
        self.assertEqual(docs[0].metadata["page"], 1)
        self.assertEqual(docs[0].metadata["total_pages"], 2)

    def test_copy_pdf_to_serve_dir_creates_copy(self):
        scanner = PDFScanner()

        with patch("datacollector.pdf_scanner.PDF_SERVE_DIR", Path(self.tempdir.name)):
            dest_path = scanner._copy_pdf_to_serve_dir(self.pdf_path)

        self.assertEqual(dest_path, Path(self.tempdir.name) / "test.pdf")
        self.assertTrue(dest_path.exists())

    def test_extract_text_from_page_returns_text_or_empty_string(self):
        page = MagicMock()
        page.get_text.return_value = "  Right t o erasure.  "

        text = _extract_text_from_page(page)

        self.assertEqual(text, "  Right t o erasure.  ")

    def test_extract_text_from_page_handles_exception(self):
        page = MagicMock()
        page.get_text.side_effect = Exception("extraction failed")

        text = _extract_text_from_page(page)

        self.assertEqual(text, "")

    @patch("datacollector.pdf_scanner.fitz.open")
    def test_scan_pdf_does_not_fix_broken_words(self, mock_fitz_open):
        mock_page = MagicMock()
        mock_page.get_text.return_value = "Right t o erasure. This text remains unchanged during extraction."
        mock_doc = MockFitZDocument([mock_page])
        mock_fitz_open.return_value = mock_doc

        scanner = PDFScanner()

        docs = scanner.scan_pdf(self.pdf_path)

        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0].content, "Right t o erasure. This text remains unchanged during extraction.")
        self.assertEqual(docs[0].metadata["extraction_method"], "pymupdf")

    @patch("datacollector.pdf_scanner.fitz.open")
    def test_export_to_json(self, mock_fitz_open):
        mock_doc = MockFitZDocument([MockPage()])
        mock_fitz_open.return_value = mock_doc

        scanner = PDFScanner()
        scanner.scan_pdf(self.pdf_path)
        result_path = scanner.export_to_json(self.output_path)

        self.assertTrue(os.path.exists(result_path))
        with open(result_path, "r") as f:
            data = json.load(f)
        self.assertEqual(len(data), 1)

    @patch("datacollector.pdf_scanner.write_chunks_to_file")
    @patch("datacollector.pdf_scanner.fitz.open")
    def test_build_chunks(self, mock_fitz_open, mock_write_chunks):
        mock_doc = MockFitZDocument([MockPage()])
        mock_fitz_open.return_value = mock_doc
        mock_write_chunks.return_value = 3

        scanner = PDFScanner()
        scanner.scan_pdf(self.pdf_path)
        chunk_count, saved_path = scanner.build_chunks()

        self.assertEqual(chunk_count, 3)

    @patch("datacollector.pdf_scanner.fitz.open")
    def test_scan_pdf_uses_chapter_filter_when_chapters_provided(self, mock_fitz_open):
        mock_doc = MockFitZDocument([
            MockChapterPage(1, has_chapter="Chapter 1"),
            MockChapterPage(2, has_chapter="Chapter 1"),
            MockChapterPage(3, has_chapter="Chapter 2"),
            MockChapterPage(4),
        ])
        mock_fitz_open.return_value = mock_doc

        scanner = PDFScanner()
        docs = scanner.scan_pdf(self.pdf_path, chapters=["Chapter 1"])

        self.assertEqual(len(docs), 2)
        self.assertEqual(docs[0].metadata["page"], 1)
        self.assertEqual(docs[1].metadata["page"], 2)

    @patch("datacollector.pdf_scanner.fitz.open")
    def test_scan_pdf_uses_page_range_filter(self, mock_fitz_open):
        mock_doc = MockFitZDocument([MockPage(), MockPage(), MockPage(), MockPage()])
        mock_fitz_open.return_value = mock_doc

        scanner = PDFScanner()
        docs = scanner.scan_pdf(self.pdf_path, page_range="1,3")

        self.assertEqual(len(docs), 2)
        self.assertEqual(docs[0].metadata["page"], 1)
        self.assertEqual(docs[1].metadata["page"], 3)

    @patch("datacollector.pdf_scanner.fitz.open")
    def test_scan_pdf_combines_chapter_and_page_range_filters(self, mock_fitz_open):
        mock_doc = MockFitZDocument([
            MockChapterPage(1, has_chapter="Chapter 1"),
            MockChapterPage(2, has_chapter="Chapter 1"),
            MockChapterPage(3, has_chapter="Chapter 2"),
            MockChapterPage(4, has_chapter="Chapter 2"),
        ])
        mock_fitz_open.return_value = mock_doc

        scanner = PDFScanner()
        docs = scanner.scan_pdf(self.pdf_path, chapters=["Chapter 1"], page_range="1-2")

        self.assertEqual(len(docs), 2)
        self.assertEqual(docs[0].metadata["page"], 1)
        self.assertEqual(docs[1].metadata["page"], 2)

    @patch("datacollector.pdf_scanner.fitz.open")
    def test_scan_pdf_no_chapters_matched_scans_all(self, mock_fitz_open):
        mock_doc = MockFitZDocument([MockPage()])
        mock_fitz_open.return_value = mock_doc

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

    @patch("datacollector.pdf_scanner.fitz.open")
    def test_get_pdf_reader_returns_reader(self, mock_fitz_open):
        mock_doc = MagicMock()
        mock_fitz_open.return_value = mock_doc

        reader = _get_pdf_reader(self.pdf_path)

        self.assertEqual(reader, mock_doc)

    def test_get_pdf_reader_raises_for_missing_file(self):
        with self.assertRaises(FileNotFoundError):
            _get_pdf_reader("/nonexistent/path.pdf")

    def test_pre_extract_pages_extracts_allowed_pages(self):
        mock_page1 = MagicMock()
        mock_page1.get_text.return_value = "text1"
        mock_page2 = MagicMock()
        mock_page2.get_text.return_value = "text2"
        mock_doc = MockFitZDocument([mock_page1, mock_page2])

        result = _pre_extract_pages(mock_doc, {1})

        self.assertIsNotNone(result)
        self.assertIn(1, result)

    def test_pre_extract_pages_returns_none_when_no_allowed_pages(self):
        result = _pre_extract_pages(MagicMock(), None)
        self.assertIsNone(result)

    def test_build_page_metadata_returns_correct_dict(self):
        metadata = _build_page_metadata("Test Book", 1, 5, "pymupdf", "doc_hash", "Page text")

        self.assertEqual(metadata["page"], 1)
        self.assertEqual(metadata["total_pages"], 5)
        self.assertEqual(metadata["extraction_method"], "pymupdf")
        self.assertEqual(metadata["book"], "Test Book")

    def test_create_page_document_creates_document(self):
        doc = _create_page_document("/pdfs/test.pdf", "Test Book", 1, "text", {"page": 1})

        self.assertIn("page=1", doc.source)
        self.assertEqual(doc.title, "Test Book (page 1)")
        self.assertEqual(doc.content, "text")


if __name__ == "__main__":
    unittest.main()