import json
import logging
import os
import re
import shutil
from pathlib import Path
from typing import List, Optional

import fitz

from chunker.document import Document, compute_content_hash
from chunker.chunker import Chunker, write_chunks_to_file, get_chunk_file_path
from datacollector.base import DataCollector

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


PDF_SERVE_DIR = Path(__file__).resolve().parent.parent / "pdfs"

_CHAPTER_RE = re.compile(r"(?m)^(\d+\.?\s*Chapter\s*\d+|\bChapter\s+\d+[:\s]|\bSection\s+\d+[:\s]|\bArticle\s+\d+[:\s])", re.IGNORECASE)
_PAGE_HEADING_RE = re.compile(r"(?m)^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s*(?:Chapter|Section|Article)\s*\d+)$", re.IGNORECASE)
_CHUNK_PREVIEW_MAX_PAGES = 50


def _extract_text_from_page(page) -> str:
    try:
        return page.get_text("text") or ""
    except Exception:
        return ""


def _parse_page_ranges(page_range: Optional[str], total_pages: int) -> Optional[set]:
    if not page_range or not str(page_range).strip():
        return None
    pages = set()
    for part in str(page_range).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_str, end_str = part.split("-", 1)
            try:
                start = int(start_str.strip())
            except ValueError:
                continue
            try:
                end = int(end_str.strip())
            except ValueError:
                continue
            start = max(1, min(start, total_pages))
            end = max(start, min(end, total_pages))
            pages.update(range(start, end + 1))
        else:
            try:
                page_num = int(part)
            except ValueError:
                continue
            if 1 <= page_num <= total_pages:
                pages.add(page_num)
    return pages if pages else None


def preflight_chapters(pdf_path: str, max_pages: Optional[int] = _CHUNK_PREVIEW_MAX_PAGES) -> List[dict]:
    reader = _get_pdf_reader(pdf_path)

    count = 0
    chapters: List[dict] = []
    last_chapter: Optional[str] = None
    for page_num in range(len(reader)):
        if max_pages is not None and count >= max_pages:
            break
        count += 1
        text = _extract_text_from_page(reader[page_num])
        if not text.strip():
            continue
        chapter = _extract_chapter_info(text)
        if chapter and chapter != last_chapter:
            chapters.append({"chapter": chapter, "page": page_num + 1, "pages_span": page_num + 1})
            last_chapter = chapter
        elif last_chapter is not None and chapters:
            chapters[-1]["pages_span"] = page_num + 1
    return chapters


def compute_chapter_pages(pdf_path: str, chapters: Optional[List[str]] = None) -> tuple[Optional[set], dict[str, set], int]:
    reader = _get_pdf_reader(pdf_path)

    total_pages = len(reader)
    if not chapters:
        return None, {}, total_pages

    chapter_pages: dict[str, set] = {}
    for page_num in range(total_pages):
        text = _extract_text_from_page(reader[page_num])
        chapter = _extract_chapter_info(text)
        if chapter:
            chapter_pages.setdefault(chapter, set()).add(page_num + 1)

    allowed_pages: set = set()
    for label in chapters:
        allowed_pages.update(chapter_pages.get(label, set()))

    if not allowed_pages:
        return None, chapter_pages, total_pages

    return allowed_pages, chapter_pages, total_pages


def _extract_chapter_info(text: str) -> Optional[str]:
    matches = _CHAPTER_RE.findall(text)
    if matches:
        return matches[0].strip()
    matches = _PAGE_HEADING_RE.findall(text)
    if matches:
        return matches[0].strip()
    lines = [line.strip() for line in text.split('\n')[:3] if line.strip()]
    for line in lines:
        if 3 < len(line) < 80 and line.isupper() and ' ' in line:
            words = line.split()
            if len(words) >= 2:
                return line.title()
    return None


def _get_pdf_reader(pdf_path: str) -> fitz.Document:
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    try:
        return fitz.open(pdf_path)
    except Exception as exc:
        raise RuntimeError(f"Failed to read PDF {pdf_path}: {exc}") from exc


def _get_allowed_pages(
    reader: fitz.Document,
    chapters: Optional[List[str]],
    page_range: Optional[str],
) -> tuple[Optional[set], int]:
    total_pages = len(reader)
    allowed_pages: Optional[set] = None
    if chapters:
        chapter_pages: dict[str, set] = {}
        for page_num in range(total_pages):
            text = _extract_text_from_page(reader[page_num])
            chapter = _extract_chapter_info(text)
            if chapter:
                chapter_pages.setdefault(chapter, set()).add(page_num + 1)
        allowed_pages = set()
        for label in chapters:
            allowed_pages.update(chapter_pages.get(label, set()))
        if not allowed_pages:
            logger.info(f"No chapters matched the selection {chapters}")
            allowed_pages = None
    if page_range:
        range_pages = _parse_page_ranges(page_range, total_pages)
        if range_pages:
            if allowed_pages is not None:
                allowed_pages.intersection_update(range_pages)
            else:
                allowed_pages = range_pages
        if not allowed_pages:
            logger.info(f"No pages matched the range {page_range}")
            allowed_pages = None
    return allowed_pages, total_pages


def _pre_extract_pages(
    reader: fitz.Document, allowed_pages: Optional[set]
) -> Optional[dict[int, str]]:
    if allowed_pages is None:
        return None
    pre_extracted: dict[int, str] = {}
    for page_num in range(len(reader)):
        if (page_num + 1) in allowed_pages:
            raw_text = _extract_text_from_page(reader[page_num])
            pre_extracted[page_num + 1] = raw_text
    return pre_extracted


def _build_page_metadata(
    title: str,
    page_num: int,
    total_pages: int,
    extraction_method: str,
    document_hash: str,
    text: str,
) -> dict:
    page_hash = compute_content_hash(text)
    chapter = _extract_chapter_info(text)
    return {
        "page": page_num,
        "total_pages": total_pages,
        "extraction_method": extraction_method,
        "book": re.sub(r'\s+', ' ', title.replace(".pdf", "").replace("_", " ").replace("-", " ")).strip(),
        "chapter": chapter,
        "source_hash": document_hash,
        "page_hash": page_hash,
    }


def _create_page_document(
    base_source: str, title: str, page_num: int, text: str, metadata: dict
) -> Document:
    return Document.create(
        source=f"{base_source}#page={page_num}",
        title=f"{title} (page {page_num})",
        content=text,
        metadata=metadata,
    )


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
        source = kwargs.get("source")
        chapters = kwargs.get("chapters")
        original_filename = kwargs.get("original_filename")
        return self.scan_pdf(pdf_path, source=source, chapters=chapters, original_filename=original_filename)

    def _compute_document_hash(self, pdf_path: str) -> str:
        try:
            with open(pdf_path, "rb") as f:
                return compute_content_hash(f.read().decode("latin-1", errors="ignore"))
        except Exception:
            return compute_content_hash(pdf_path)

    def scan_pdf_chapters(self, pdf_path: str, source: str = None, original_filename: Optional[str] = None, chapters: Optional[List[str]] = None) -> List[Document]:
        return self.scan_pdf(pdf_path, source=source, original_filename=original_filename, chapters=chapters)

    def _copy_pdf_to_serve_dir(self, pdf_path: str) -> Path:
        PDF_SERVE_DIR.mkdir(parents=True, exist_ok=True)
        dest_path = PDF_SERVE_DIR / Path(pdf_path).name
        if not dest_path.exists():
            try:
                shutil.copy2(pdf_path, dest_path)
            except Exception as e:
                logger.warning(f"Could not copy PDF to serve directory: {e}")
        return dest_path

    def scan_pdf(
        self,
        pdf_path: str,
        source: str = None,
        chapters: Optional[List[str]] = None,
        page_range: Optional[str] = None,
        original_filename: Optional[str] = None,
    ) -> List[Document]:

        title = Path(original_filename).stem if original_filename else Path(pdf_path).stem
        document_hash = self._compute_document_hash(pdf_path)

        try:
            reader = fitz.open(pdf_path)
        except Exception as e:
            logger.error(f"Failed to read PDF {pdf_path}: {e}")
            return []

        self._copy_pdf_to_serve_dir(pdf_path)

        base_source = source or f"/pdfs/{Path(pdf_path).name}"
        total_pages = len(reader)
        skipped = 0

        allowed_pages, total_pages = _get_allowed_pages(reader, chapters, page_range)

        pre_extracted = _pre_extract_pages(reader, allowed_pages)

        self.documents = []
        for page_num in range(total_pages):
            try:
                actual_page_num = page_num + 1
                if allowed_pages is not None and actual_page_num not in allowed_pages:
                    skipped += 1
                    continue
                if pre_extracted and pre_extracted.get(actual_page_num) is not None:
                    raw_text = pre_extracted[actual_page_num]
                    extraction_method = "pymupdf"
                else:
                    raw_text = _extract_text_from_page(reader[page_num])
                    extraction_method = "pymupdf"

                text = raw_text
                if text and text.strip():
                    metadata = _build_page_metadata(
                        title, actual_page_num, total_pages, extraction_method, document_hash, text
                    )
                    document = _create_page_document(
                        base_source, title, actual_page_num, text, metadata
                    )
                    self.documents.append(document)
            except Exception as e:
                logger.warning(f"Failed to extract text from page {page_num + 1}: {e}")

        if skipped:
            logger.info(f"Skipped {skipped} pages for {pdf_path}")
        logger.info(f"Extracted {len(self.documents)} pages from {pdf_path}")
        reader.close()
        return self.documents

    def export_to_json(self, output_file: str = None) -> str:
        output_file = output_file or self.output_file
        os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)

        data = [doc.to_dict() for doc in self.documents]
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

        logger.info(f"Exported {len(self.documents)} documents to {output_file}")
        return output_file

    def build_chunks(self, output_file: str = None) -> tuple:
        if output_file is None:
            output_file = self.output_file
        chunk_file = get_chunk_file_path(output_file)

        if not self.documents:
            raise ValueError("No documents to chunk. Run scan_pdf() first.")

        total_chunks = write_chunks_to_file(self.documents, chunk_file)
        logger.info(f"Created {total_chunks} chunks in {chunk_file}")
        return total_chunks, chunk_file


def scan_and_build_chunks(pdf_path: str, output_file: str = None) -> tuple:
    scanner = PDFScanner()
    scanner.scan(pdf_path)
    return scanner.build_chunks(output_file=output_file)
