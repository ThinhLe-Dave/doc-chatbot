from __future__ import annotations

import concurrent.futures
import hashlib
import json
import time
import uuid
from collections import deque
from threading import Lock, local
from urllib.parse import urljoin, urlparse, urlunparse
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup

from datacollector.base import DataCollector
from chunker.document import compute_content_hash, Document
from chunker.chunker import get_chunk_file_path, write_chunks_to_file
from utils.logging import debug, info, warning, error


class Scraper(DataCollector):
    def __init__(
        self,
        base_url: str,
        output_file: str = "website_data.json",
        delay: float = 0.25,
        obey_robots: bool = True,
        max_workers: int = 6,
        max_response_bytes: int = 10 * 1024 * 1024,
        retries: int = 3,
        sitemap_first: bool = False,
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
        self.max_response_bytes = max_response_bytes
        self.retries = retries
        self.sitemap_first = sitemap_first
        self.metrics = {
            "discovered": 0,
            "fetched": 0,
            "skipped": 0,
            "failed": 0,
            "bytes": 0,
        }

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
        info(f"Successfully exported data to {output_file}")
        return output_file

    def _polite_sleep(self):
        with self.request_lock:
            if self.delay > 0:
                now = time.monotonic()
                wait_time = self.delay - (now - self.last_request_time)
                if wait_time > 0:
                    time.sleep(wait_time)
            self.last_request_time = time.monotonic()

    def _fetch_page(self, url: str, last_modified: str = "", etag: str = ""):
        headers = {}
        if etag:
            headers["If-None-Match"] = etag
        if last_modified:
            headers["If-Modified-Since"] = last_modified

        session = self.session
        for attempt in range(1, self.retries + 1):
            try:
                self._polite_sleep()
                response = session.get(url, headers=headers, timeout=10, stream=True)
                if response.status_code == 304:
                    return url, response, None, True
                response.raise_for_status()
                content_type = response.headers.get("Content-Type", "")
                if "text/html" not in content_type and "text" not in content_type:
                    return url, response, ValueError(f"unsupported content type: {content_type}"), False
                body = b""
                for chunk in response.iter_content(8192):
                    body += chunk
                    if len(body) > self.max_response_bytes:
                        return url, None, ValueError("response exceeded max size"), False
                response._content = body
                response._content_consumed = True
                return url, response, None, False
            except requests.RequestException as e:
                if attempt == self.retries:
                    return url, None, e, False
                time.sleep(0.5 * attempt)

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

    def _get_canonical_url(self, url: str) -> str:
        if url in self.canonical_cache:
            return self.canonical_cache[url]
        canonical = self._canonical_url(url)
        self.canonical_cache[url] = canonical
        return canonical

    def crawl(self, current_url: str = None, max_pages: int = 20, force: bool = False):
        """Crawl a website using a queue and visit only internal URLs."""
        start_url = self._normalize_url(current_url or self.base_url)
        self.domain = urlparse(start_url).netloc
        queue: deque[str] = deque()

        if self.sitemap_first and not force:
            sitemap_urls = self._discover_sitemap_urls()
            for url in sitemap_urls:
                if url not in self.visited_urls:
                    queue.append(url)
            info(f"Seeded {len(sitemap_urls)} sitemap URL(s)")

        if not queue:
            queue.append(start_url)

        while queue and len(self.visited_urls) < max_pages:
            batch = []
            while queue and len(batch) < self.max_workers:
                url = queue.popleft()
                if not url or url in self.visited_urls:
                    continue
                if not self._can_fetch(url):
                    warning(f"Skipping disallowed URL by robots.txt: {url}")
                    debug("Skipping disallowed URL by robots.txt: {url}", "scraper")
                    continue
                batch.append(url)

            if not batch:
                break

            self.visited_urls.update(batch)
            self.metrics["discovered"] += len(batch)
            info(f"Processing batch: {len(batch)} url(s)")
            debug(f"Processing batch: {len(batch)} url(s)", "scraper")

            with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(batch), self.max_workers)) as executor:
                futures = {executor.submit(self._fetch_page, url): url for url in batch}
                pending_urls = []

                for future in concurrent.futures.as_completed(futures):
                    url = futures[future]
                    url, response, error, not_modified = future.result()
                    if error:
                        if isinstance(error, ValueError) and "response exceeded max size" in str(error):
                            warning(f"Skipped oversized response for {url}")
                            debug(f"Skipped oversized response for {url}", "scraper")
                            self.metrics["skipped"] += 1
                        else:
                            error(f"Failed to fetch {url}: {error}")
                            debug(f"Failed to fetch {url}: {error}", "scraper")
                            self.metrics["failed"] += 1
                        continue

                    if not_modified and not force:
                        info(f"Not modified: {url}")
                        debug(f"Not modified: {url}", "scraper")
                        self.metrics["skipped"] += 1
                        continue

                    try:
                        soup = BeautifulSoup(response.text, "html.parser")
                        new_urls = self._process_url(url, soup)
                        pending_urls.extend(new_urls)
                        self.metrics["fetched"] += 1
                        self.metrics["bytes"] += len(response.text.encode("utf-8", errors="ignore"))
                        debug(f"Fetched {url} bytes={self.metrics['bytes']}", "scraper")
                    except Exception as e:
                        error(f"Failed to parse {url}: {e}")
                        debug(f"Failed to parse {url}: {e}", "scraper")
                        self.metrics["failed"] += 1

            for url in pending_urls:
                if url not in self.visited_urls and len(self.visited_urls) + len(queue) < max_pages:
                    queue.append(url)

    def build_chunks(self, output_file: str = None) -> tuple:
        if output_file is None:
            output_file = self.output_file
        chunk_file = get_chunk_file_path(output_file)

        if not self.scraped_data:
            raise ValueError("No documents to chunk. Run crawl() first.")

        documents = [Document.from_dict(item) for item in self.scraped_data]
        total_chunks = write_chunks_to_file(documents, chunk_file)
        info(f"Created {total_chunks} chunks in {chunk_file}")
        return total_chunks, chunk_file

    def _canonical_netloc(self, netloc: str) -> str:
        return netloc.lower().removeprefix("www.")

    def _canonical_url(self, url: str) -> str:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            return ""

        base = getattr(self, "base_url", url)
        scheme = urlparse(base).scheme or parsed.scheme
        netloc = self._canonical_netloc(parsed.netloc)
        path = parsed.path or "/"
        if path != "/":
            path = path.rstrip("/")

        return urlunparse((scheme, netloc, path, "", "", ""))

    def _normalize_url(self, url: str) -> str:
        """Normalize a URL by resolving relative paths and canonicalizing scheme, host, and path."""
        if not url:
            return ""

        base = getattr(self, "base_url", url)
        absolute = urljoin(base, url)
        parsed = urlparse(absolute)

        if parsed.scheme not in {"http", "https"}:
            return ""

        return self._canonical_url(absolute)

    def _is_internal(self, url: str) -> bool:
        """Return True when the URL belongs to the same domain being crawled."""
        parsed = urlparse(url)
        return parsed.scheme in {"http", "https"} and self._canonical_netloc(parsed.netloc) == self._canonical_netloc(self.domain)

    def _setup_robots_parser(self) -> RobotFileParser:
        """Load robots.txt for the current domain so the crawler can obey access rules."""
        parser = RobotFileParser()
        robots_url = urlunparse((urlparse(self.base_url).scheme, self.domain, "/robots.txt", "", "", ""))
        parser.set_url(robots_url)
        try:
            parser.read()
        except Exception:
            warning("Unable to load robots.txt, continuing without robots rules.")
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

    def _parse_sitemap(self, sitemap_url: str) -> list[str]:
        urls = []
        try:
            response = self.session.get(sitemap_url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "xml")
            for loc in soup.find_all("loc"):
                url = loc.get_text(strip=True)
                if url:
                    urls.append(url)
        except Exception as e:
            warning(f"Failed to load sitemap {sitemap_url}: {e}")
        return urls

    def _discover_sitemap_urls(self) -> list[str]:
        urls = []
        robots_url = urlunparse((urlparse(self.base_url).scheme, self.domain, "/robots.txt", "", "", ""))
        try:
            response = self.session.get(robots_url, timeout=10)
            if response.status_code == 200:
                for line in response.text.splitlines():
                    line = line.strip()
                    if line.lower().startswith("sitemap:"):
                        sitemap_url = line.split(":", 1)[1].strip()
                        urls.extend(self._parse_sitemap(sitemap_url))
        except Exception:
            pass

        if not urls:
            for path in ["/sitemap.xml", "/sitemap_index.xml"]:
                sitemap_url = urlunparse((urlparse(self.base_url).scheme, self.domain, path, "", "", ""))
                urls.extend(self._parse_sitemap(sitemap_url))

        seen = set()
        result = []
        for url in urls:
            canonical = self._canonical_url(url)
            if canonical and canonical not in seen:
                seen.add(canonical)
                result.append(canonical)
        return result


def scrape_and_build_chunks(url: str, output_file: str = "website_data.json", limit: int = 10000) -> tuple:
    """Convenience function to crawl website and build chunks in one call."""
    scraper = Scraper(base_url=url, output_file=output_file)
    scraper.crawl(max_pages=limit)
    total_chunks, chunk_file = scraper.build_chunks()
    from processor.processor import build_chunk_cache
    return build_chunk_cache(chunk_file)
