import json
import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

sys.modules['bs4'] = MagicMock()
sys.modules['requests'] = MagicMock()

from datacollector.crawler import Scraper, scrape_and_build_chunks


class MockResponse:
    text = "<html><head><title>Test Page</title></head><body><p>Test content</p></body></html>"


class MockSoup:
    title = type('obj', (object,), {'string': 'Test Page'})
    
    def find_all(self, *args, **kwargs):
        return []
    
    def get_text(self, separator=" ", strip=False):
        return "Test content"


class ScraperTest(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.output_path = os.path.join(self.tempdir.name, "output.json")

    def tearDown(self):
        self.tempdir.cleanup()

    @patch("datacollector.crawler.Scraper._fetch_page")
    def test_crawl_collects_pages(self, mock_fetch):
        mock_fetch.return_value = ("http://example.com", MagicMock(text=MockResponse.text), None, False)

        scraper = Scraper(base_url="http://example.com", output_file=self.output_path)
        scraper.crawl(max_pages=1)

        self.assertEqual(len(scraper.scraped_data), 1)

    @patch("datacollector.crawler.Scraper._fetch_page")
    def test_scan_collects_pages(self, mock_fetch):
        mock_fetch.return_value = ("http://example.com", MagicMock(text=MockResponse.text), None, False)

        scraper = Scraper(base_url="http://example.com", output_file=self.output_path)
        result = scraper.scan()

        self.assertEqual(len(result), 1)
        self.assertEqual(len(scraper.scraped_data), 1)

    @patch("datacollector.crawler.Scraper._fetch_page")
    def test_collect_returns_scraped_data(self, mock_fetch):
        mock_fetch.return_value = ("http://example.com", MagicMock(text=MockResponse.text), None, False)

        scraper = Scraper(base_url="http://example.com", output_file=self.output_path)
        result = scraper.collect("http://example.com")

        self.assertEqual(len(result), 1)

    @patch("datacollector.crawler.Scraper._fetch_page")
    def test_export_to_json(self, mock_fetch):
        mock_fetch.return_value = ("http://example.com", MagicMock(text=MockResponse.text), None, False)

        scraper = Scraper(base_url="http://example.com", output_file=self.output_path)
        scraper.crawl(max_pages=1)
        scraper.scraped_data = [{"url": "http://example.com", "title": "Test", "body": "Test content"}]
        result_path = scraper.export_to_json()

        self.assertTrue(os.path.exists(result_path))
        with open(result_path, "r") as f:
            data = json.load(f)
        self.assertEqual(len(data), 1)


if __name__ == "__main__":
    unittest.main()