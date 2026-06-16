import hashlib
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy
sys.modules['bs4'] = MagicMock()
sys.modules['requests'] = MagicMock()

from chunker.document import compute_content_hash
from datacollector.crawler import Scraper
from datacollector.pdf_scanner import PDFScanner


class SitemapParserTest(unittest.TestCase):
    @patch("requests.Session.get")
    def test_discover_sitemap_from_robots(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "\n".join([
            "User-agent: *",
            "Disallow: /private/",
            "Sitemap: https://example.com/sitemap.xml",
        ])
        mock_get.return_value = mock_response

        with patch.object(Scraper, "_parse_sitemap") as mock_parse:
            mock_parse.return_value = ["https://example.com/page/1"]
            scraper = Scraper(base_url="https://example.com", obey_robots=False)
            scraper._discover_sitemap_urls()
            mock_parse.assert_any_call("https://example.com/sitemap.xml")
            urls = scraper._discover_sitemap_urls()
            self.assertIn("https://example.com/page/1", urls)

    @patch("requests.Session.get")
    def test_discover_falls_back_to_sitemap_path(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response

        with patch("datacollector.crawler.Scraper._parse_sitemap") as mock_parse:
            mock_parse.return_value = ["https://example.com/page/2"]
            scraper = Scraper(base_url="https://example.com", obey_robots=False)
            urls = scraper._discover_sitemap_urls()
            self.assertIn("https://example.com/page/2", urls)
            calls = [c.args[0] for c in mock_parse.call_args_list]
            self.assertIn("https://example.com/sitemap.xml", calls)
            self.assertIn("https://example.com/sitemap_index.xml", calls)


class CanonicalizationTest(unittest.TestCase):
    def test_canonical_removes_www(self):
        scraper = Scraper.__new__(Scraper)
        scraper.base_url_netloc_bak = "www.example.com"
        scraper.base_url = "https://www.example.com"
        self.assertEqual(scraper._canonical_url("https://www.example.com/path"), "https://example.com/path")

    def test_canonical_same_origin_when_no_expand(self):
        scraper = Scraper.__new__(Scraper)
        scraper.base_url = "https://example.com"
        self.assertEqual(scraper._canonical_url("https://example.com"), "https://example.com/")


class IncrementalSkipTest(unittest.TestCase):
    @patch("datacollector.crawler.BeautifulSoup")
    def test_skips_unchanged_page(self, mock_soup_cls):
        url = "https://example.com/page/1"
        mock_response = MagicMock()
        mock_response.status_code = 304
        mock_response.text = ""

        scraper = Scraper(base_url="https://example.com")
        scraper.session = MagicMock()
        scraper._polite_sleep = MagicMock()
        scraper.session.get.return_value = mock_response
        mock_soup_cls.return_value = MagicMock()
        scraper._process_url = MagicMock(return_value=[])

        scraper.crawl(max_pages=5)
        scraper._process_url.assert_not_called()

    def test_computes_content_hash(self):
        content = "Hello World"
        expected = hashlib.sha256(" ".join(content.split()).encode("utf-8", errors="ignore")).hexdigest()
        self.assertEqual(compute_content_hash(content), expected)


class MetricsTest(unittest.TestCase):
    def test_metrics_initial_values(self):
        scraper = Scraper(base_url="https://example.com", max_workers=1)
        self.assertEqual(scraper.metrics["discovered"], 0)


class PDFScannerHashingTest(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.pdf_path = os.path.join(self.tempdir.name, "test.pdf")
        with open(self.pdf_path, "wb") as f:
            f.write(b"%PDF-1.4\n")

    def tearDown(self):
        self.tempdir.cleanup()

    @patch("datacollector.pdf_scanner.fitz.open")
    def test_scan_pdf_records_source_hash(self, mock_fitz_open):
        class MockFitZPage:
            def get_text(self, mode="text"):
                return "Document content here for testing page hash purposes."

        class MockFitZDoc:
            def __init__(self):
                self._pages = [MockFitZPage()]

            def __len__(self):
                return len(self._pages)

            def __getitem__(self, index):
                return self._pages[index]

            def close(self):
                pass

        mock_fitz_open.return_value = MockFitZDoc()

        scanner = PDFScanner()
        docs = scanner.scan_pdf(self.pdf_path)
        self.assertEqual(len(docs), 1)
        self.assertIn("source_hash", docs[0].metadata)
        self.assertEqual(len(docs[0].metadata["source_hash"]), 64)

    @patch("datacollector.pdf_scanner.fitz.open")
    def test_scan_pdf_records_page_hash(self, mock_fitz_open):
        class MockFitZPage:
            def get_text(self, mode="text"):
                return "Page content extracted successfully for hashing."

        class MockFitZDoc:
            def __init__(self):
                self._pages = [MockFitZPage()]

            def __len__(self):
                return len(self._pages)

            def __getitem__(self, index):
                return self._pages[index]

            def close(self):
                pass

        mock_fitz_open.return_value = MockFitZDoc()

        scanner = PDFScanner()
        docs = scanner.scan_pdf(self.pdf_path)
        self.assertEqual(len(docs), 1)
        self.assertIn("page_hash", docs[0].metadata)


def load_suite():
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(SitemapParserTest))
    suite.addTests(loader.loadTestsFromTestCase(CanonicalizationTest))
    suite.addTests(loader.loadTestsFromTestCase(IncrementalSkipTest))
    suite.addTests(loader.loadTestsFromTestCase(MetricsTest))
    suite.addTests(loader.loadTestsFromTestCase(PDFScannerHashingTest))
    return suite


if __name__ == "__main__":
    unittest.main(defaultTest="load_suite")
