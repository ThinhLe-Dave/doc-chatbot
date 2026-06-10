# Data utilities module - re-exports from datacollector
from datacollector.base import DataCollector
from datacollector.crawler import Scraper, scrape_and_build_chunks
from datacollector.pdf_scanner import PDFScanner, scan_and_build_chunks

__all__ = ["DataCollector", "Scraper", "scrape_and_build_chunks", "PDFScanner", "scan_and_build_chunks"]


def scrape(url: str, output: str = "research_data.json", limit: int = 10000) -> tuple:
    """CLI-style function to scrape a URL and build chunks."""
    import os
    from processor.processor import build_chunk_cache
    
    final_output_path = os.path.join("database", output)
    os.makedirs("database", exist_ok=True)
    
    scraper = Scraper(base_url=url, output_file=final_output_path)
    scraper.crawl(max_pages=limit)
    scraper.export_to_json()
    
    return build_chunk_cache(final_output_path)


def pdf_scan(path: str, output: str = "pdf_data.json") -> tuple:
    """CLI-style function to scan a PDF and build chunks."""
    import os
    
    if not os.path.exists(path) or not path.lower().endswith(".pdf"):
        raise ValueError(f"Invalid PDF path: {path}")
    
    final_output_path = os.path.join("database", output)
    os.makedirs("database", exist_ok=True)
    
    scanner = PDFScanner()
    scanner.scan_pdf(path)
    scanner.export_to_json(final_output_path)
    
    return scanner.build_chunks(final_output_path)
