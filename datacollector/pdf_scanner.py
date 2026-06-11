import json
import logging
import os
import shutil
from pathlib import Path
from typing import List, Optional

from pypdf import PdfReader

from chunker.document import Document
from chunker.chunker import Chunker, write_chunks_to_file, get_chunk_file_path
from datacollector.base import DataCollector

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


PDF_SERVE_DIR = Path(__file__).resolve().parent.parent / "pdfs"


class PDFScanner(DataCollector):
    def __init__(
        self,
        output_file: str = "pdf_data.json",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ):
        super().__init__(output_file)
        self.chunker = Chunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def collect(self, source: str, **kwargs) -> List[Document]:
        return self.scan(source, **kwargs)

    def scan(self, pdf_path: str = None, **kwargs) -> List[Document]:
        """Alias for scan_pdf."""
        source = kwargs.get("source")
        return self.scan_pdf(pdf_path, source=source)

    def scan_pdf(self, pdf_path: str, source: str = None) -> List[Document]:
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        title = Path(pdf_path).stem

        try:
            reader = PdfReader(pdf_path)
        except Exception as e:
            logger.error(f"Failed to read PDF {pdf_path}: {e}")
            return []

        PDF_SERVE_DIR.mkdir(parents=True, exist_ok=True)
        dest_path = PDF_SERVE_DIR / Path(pdf_path).name
        if not dest_path.exists():
            try:
                shutil.copy2(pdf_path, dest_path)
            except Exception as e:
                logger.warning(f"Could not copy PDF to serve directory: {e}")

        base_source = source or f"/pdfs/{Path(pdf_path).name}"

        self.documents = []
        for page_num, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text()
                if text and text.strip():
                    document = Document.create(
                        source=f"{base_source}#page={page_num}",
                        title=f"{title} (page {page_num})",
                        content=text,
                        metadata={"page": page_num, "total_pages": len(reader.pages)},
                    )
                    self.documents.append(document)
            except Exception as e:
                logger.warning(f"Failed to extract text from page {page_num}: {e}")

        logger.info(f"Extracted {len(self.documents)} pages from {pdf_path}")
        return self.documents

    def export_to_json(self, output_file: str = None) -> str:
        """Export extracted documents to JSON file."""
        output_file = output_file or self.output_file
        os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)

        data = [doc.to_dict() for doc in self.documents]
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

        logger.info(f"Exported {len(self.documents)} documents to {output_file}")
        return output_file

    def build_chunks(self, output_file: str = None) -> tuple:
        """Build chunk cache from extracted documents."""
        if output_file is None:
            output_file = self.output_file
        # Derive chunk file path from the data file path
        chunk_file = get_chunk_file_path(output_file)

        if not self.documents:
            raise ValueError("No documents to chunk. Run scan_pdf() first.")

        total_chunks = write_chunks_to_file(self.documents, chunk_file)
        logger.info(f"Created {total_chunks} chunks in {chunk_file}")
        return total_chunks, chunk_file


def scan_and_build_chunks(pdf_path: str, output_file: str = None) -> tuple:
    """Convenience function to scan PDF and build chunks in one call."""
    scanner = PDFScanner()
    scanner.scan(pdf_path)
    return scanner.build_chunks(output_file=output_file)