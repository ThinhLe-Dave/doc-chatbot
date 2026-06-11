import concurrent.futures
import json
import logging
import time
import uuid
from collections import deque
from threading import Lock, local
from urllib.parse import urljoin, urlparse, urlunparse
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup

from datacollector.base import DataCollector


class Scraper(DataCollector):
    def __init__(
        self,
        base_url: str,
        output_file: str = "website_data.json",
        delay: float = 0.25,
        obey_robots: bool = True,
        max_workers: int = 6,
    ):
        super().__init__(output_file)
        self.base_url = self._normalize_url(base_url)
        self.visited_urls = set()
        self.scraped_data = []
        self.scraped_data_lock = Lock()
        self.domain = urlparse(self.base_url).netloc
        self.delay = delay
        self.obey_robots = obey_robots
        self.max_workers = max_workers
        self.last_request_time = 0.0
        self.request_lock = Lock()
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "DocChatbotCrawler/1.0 (+https://example.com)"})
        self.robots_parser = self._setup_robots_parser() if obey_robots else None

        logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
        self.logger = logging.getLogger(__name__)

    def collect(self, source: str, **kwargs) -> list:
        """Collect data from the base_url by crawling the website."""
        max_pages = kwargs.get("max_pages", 20)
        self.crawl(max_pages=max_pages)
        return self.scraped_data

    def scan(self, url: str = None, **kwargs) -> list:
        """Alias for crawl - scans the website for data."""
        max_pages = kwargs.get("max_pages", 20)
        self.crawl(current_url=url, max_pages=max_pages)
        return self.scraped_data

    def export_to_json(self, output_file: str = None) -> str:
        """Saves the scraped data into a formatted JSON file."""
        output_file = output_file or self.output_file
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(self.scraped_data, f, ensure_ascii=False, indent=4)
        self.logger.info(f"Successfully exported data to {output_file}")
        return output_file

    def _polite_sleep(self):
        with self.request_lock:
            if self.delay > 0:
                now = time.monotonic()
                wait_time = self.delay - (now - self.last_request_time)
                if wait_time > 0:
                    time.sleep(wait_time)
            self.last_request_time = time.monotonic()

    def _fetch_page(self, url: str):
        try:
            self._polite_sleep()
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            return url, response, None
        except requests.RequestException as e:
            return url, None, e

    def _process_url(self, url: str, soup: BeautifulSoup):
        new_urls = []
        for anchor in soup.find_all("a", href=True):
            normalized = self._normalize_url(urljoin(url, anchor["href"]))
            if normalized and normalized not in self.visited_urls and self._is_internal(normalized):
                new_urls.append(normalized)

        page_content = {
            "id": str(uuid.uuid4()),
            "url": url,
            "source": url,
            "title": soup.title.string.strip() if soup.title and soup.title.string else "No Title",
            "headers": [h.get_text(strip=True) for h in soup.find_all(["h1", "h2", "h3"])],
            "content": self._extract_main_content(soup),
        }

        with self.scraped_data_lock:
            self.scraped_data.append(page_content)

        return new_urls

    def crawl(self, current_url: str = None, max_pages: int = 20):
        """Crawl a website using a queue and visit only internal URLs."""
        start_url = self._normalize_url(current_url or self.base_url)
        queue = deque([start_url])

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            while queue and len(self.visited_urls) < max_pages:
                batch = []
                while queue and len(batch) < self.max_workers:
                    url = queue.popleft()
                    if not url or url in self.visited_urls:
                        continue
                    if not self._can_fetch(url):
                        self.logger.warning(f"Skipping disallowed URL by robots.txt: {url}")
                        continue
                    batch.append(url)

                if not batch:
                    break

                self.visited_urls.update(batch)
                self.logger.info(f"Processing batch: {len(batch)} url(s)")

                futures = {executor.submit(self._fetch_page, url): url for url in batch}
                pending_urls = []

                for future in concurrent.futures.as_completed(futures):
                    url, response, error = future.result()
                    if error:
                        self.logger.error(f"Failed to fetch {url}: {error}")
                        continue

                    try:
                        soup = BeautifulSoup(response.text, "html.parser")
                        new_urls = self._process_url(url, soup)
                        pending_urls.extend(new_urls)
                    except Exception as e:
                        self.logger.error(f"Failed to parse {url}: {e}")

                for url in pending_urls:
                    if url not in self.visited_urls and len(self.visited_urls) + len(queue) < max_pages:
                        queue.append(url)

    def _normalize_url(self, url: str) -> str:
        """Normalize a URL by resolving relative paths, stripping fragments, and normalizing the path."""
        if not url:
            return ""

        base = getattr(self, "base_url", url)
        absolute = urljoin(base, url)
        parsed = urlparse(absolute)

        if parsed.scheme not in {"http", "https"}:
            return ""

        scheme = parsed.scheme.lower()
        netloc = parsed.netloc.lower()
        path = parsed.path or "/"
        if path != "/" and path.endswith("/"):
            path = path.rstrip("/")

        return urlunparse((scheme, netloc, path, "", "", ""))

    def _is_internal(self, url: str) -> bool:
        """Return True when the URL belongs to the same domain being crawled."""
        parsed = urlparse(url)
        return parsed.scheme in {"http", "https"} and parsed.netloc == self.domain

    def _setup_robots_parser(self) -> RobotFileParser:
        """Load robots.txt for the current domain so the crawler can obey access rules."""
        parser = RobotFileParser()
        robots_url = urlunparse((urlparse(self.base_url).scheme, self.domain, "/robots.txt", "", "", ""))
        parser.set_url(robots_url)
        try:
            parser.read()
        except Exception:
            self.logger.warning("Unable to load robots.txt, continuing without robots rules.")
        return parser

    def _can_fetch(self, url: str) -> bool:
        """Check robots.txt rules for the requested URL."""
        if not self.obey_robots or not self.robots_parser:
            return True
        return self.robots_parser.can_fetch(self.session.headers.get("User-Agent", "*"), url)

    def _extract_main_content(self, soup: BeautifulSoup) -> str:
        """Extract the meaningful text while stripping boilerplate noise."""
        content_soup = BeautifulSoup(str(soup), "html.parser")
        for noise in content_soup(["script", "style", "nav", "footer", "header", "aside", "form", "iframe"]):
            noise.decompose()

        main_area = (
            content_soup.find("main")
            or content_soup.find("article")
            or content_soup.find("div", id="content")
        )
        target = main_area if main_area else content_soup
        return target.get_text(separator=" ", strip=True)


def scrape_and_build_chunks(url: str, output_file: str = "website_data.json", limit: int = 10000) -> tuple:
    """Convenience function to crawl website and build chunks in one call."""
    scraper = Scraper(base_url=url, output_file=output_file)
    scraper.crawl(max_pages=limit)
    scraper.export_to_json()
    from processor.processor import build_chunk_cache
    return build_chunk_cache(output_file)