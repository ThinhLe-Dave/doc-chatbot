import io
import json
import logging
import os
import re
import shutil
import unicodedata
from pathlib import Path
from typing import List, Optional

from pypdf import PdfReader

from chunker.document import Document
from chunker.chunker import Chunker, write_chunks_to_file, get_chunk_file_path
from datacollector.base import DataCollector

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


PDF_SERVE_DIR = Path(__file__).resolve().parent.parent / "pdfs"

_COMMON_SHORT_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "can",
    "do",
    "for",
    "from",
    "go",
    "had",
    "has",
    "he",
    "her",
    "him",
    "his",
    "if",
    "in",
    "is",
    "it",
    "me",
    "my",
    "no",
    "not",
    "of",
    "on",
    "or",
    "our",
    "she",
    "so",
    "than",
    "that",
    "the",
    "then",
    "this",
    "to",
    "up",
    "us",
    "was",
    "we",
    "were",
    "will",
    "with",
    "you",
}
_SINGLE_LETTER_WORDS = {"a", "I"}
_ACRONYMS = {"AI", "EEA", "EU", "TFEU", "UK", "US"}
_CHAPTER_RE = re.compile(r"(?m)^(\d+\.?\s*Chapter\s*\d+|\bChapter\s+\d+[:\s]|\bSection\s+\d+[:\s]|\bArticle\s+\d+[:\s])", re.IGNORECASE)
_PAGE_HEADING_RE = re.compile(r"(?m)^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s*(?:Chapter|Section|Article)\s*\d+)$", re.IGNORECASE)
_UPPERCASE_SUFFIXES = {"TION", "TIONS", "MENT", "MENTS", "SHIP", "SHIPS", "ENCE", "ENCES", "ANCE", "ANCES"}
_ALPHA_TOKEN_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ]+")
_JOIN_SPLIT_WORD_RE = re.compile(r"\b([A-Za-zÀ-ÖØ-öø-ÿ]{1,30})\s+([A-Za-zÀ-ÖØ-öø-ÿ]{1,30})\b")


def _alpha_tokens(text: str) -> List[str]:
    return _ALPHA_TOKEN_RE.findall(text or "")


def _is_suspicious_token(token: str) -> bool:
    if token in _SINGLE_LETTER_WORDS:
        return False
    if token.isupper() and len(token) <= 3:
        return False
    if len(token) == 1:
        return True
    if len(token) <= 3 and token.lower() not in _COMMON_SHORT_WORDS:
        return True
    return False


def _looks_like_broken_pdf_text(text: str) -> bool:
    tokens = _alpha_tokens(text)
    if len(tokens) < 10:
        return False

    suspicious_count = sum(1 for token in tokens if _is_suspicious_token(token))
    suspicious_ratio = suspicious_count / len(tokens)
    return suspicious_ratio > 0.06


def _should_join_split_words(left: str, right: str) -> bool:
    left_lower = left.lower()
    right_lower = right.lower()

    if right == "a" and len(left) <= 3 and left_lower not in _COMMON_SHORT_WORDS:
        return True
    if right == "A" and left.isupper() and len(left) > 3:
        return True
    if left == "A" and right[:1].isupper() and len(right) > 1:
        return False
    if left in _SINGLE_LETTER_WORDS or right in _SINGLE_LETTER_WORDS:
        return False
    if len(left) == 1 and left.isupper() and left not in {"A", "I"} and len(right) > 3:
        return True
    if left.isupper() and len(left) <= 3:
        if left_lower in _COMMON_SHORT_WORDS or left in _ACRONYMS:
            return False
        if right[:1].isupper() and len(right) > 3:
            return True
        if right[:1].islower():
            return False
    if right.isupper() and len(right) <= 3:
        if right_lower in _COMMON_SHORT_WORDS or right in _ACRONYMS:
            return False
        if left.isupper() and len(left) > 3:
            return True
        if left[:1].islower():
            return False
    if left_lower in _COMMON_SHORT_WORDS and len(left) <= 3:
        return False
    if right_lower in _COMMON_SHORT_WORDS and len(right) <= 3:
        return False
    if len(right) == 1 and right.isupper() and left.isupper():
        return True
    if right.isupper() and right in _UPPERCASE_SUFFIXES:
        return left.isupper() and len(left) > 3
    if len(left) <= 3 and len(right) <= 30 and left_lower not in _COMMON_SHORT_WORDS:
        if left.isupper() and right[:1].islower():
            return False
        return True
    if len(right) <= 3 and len(left) <= 30 and right_lower not in _COMMON_SHORT_WORDS:
        return True
    return False


def _join_split_words(text: str) -> str:
    def replace(match: re.Match) -> str:
        left, right = match.groups()
        if _should_join_split_words(left, right):
            return f"{left}{right}"
        return match.group(0)

    for _ in range(6):
        updated = _JOIN_SPLIT_WORD_RE.sub(replace, text)
        if updated == text:
            return updated
        text = updated
    return text


def _extract_chapter_info(text: str) -> Optional[str]:
    """Extract chapter/section heading from page text."""
    match = _CHAPTER_RE.search(text)
    if match:
        return match.group(1).strip()
    match = _PAGE_HEADING_RE.search(text)
    if match:
        return match.group(1).strip()
    return None


def _clean_extracted_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = text.replace("\x00", "")
    text = text.replace("  ", " ")
    text = _join_split_words(text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r"\s+([,.;:!?%\)\]])", r"\1", text)
    text = re.sub(r"([\(\[\{‘“])\s+", r"\1", text)
    return text.strip()


class PDFScanner(DataCollector):
    def __init__(
        self,
        output_file: str = "pdf_data.json",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        use_ocr: Optional[bool] = None,
        ocr_language: str = "eng",
        ocr_dpi: int = 200,
    ):
        super().__init__(output_file)
        self.chunker = Chunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.use_ocr = use_ocr
        self.ocr_language = ocr_language
        self.ocr_dpi = ocr_dpi
        self._ocr_unavailable_logged = False

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
                text, extraction_method = self._extract_page_text(pdf_path, page_num - 1, page)
                if text and text.strip():
                    chapter = _extract_chapter_info(text)
                    document = Document.create(
                        source=f"{base_source}#page={page_num}",
                        title=f"{title} (page {page_num})",
                        content=text,
                        metadata={
                            "page": page_num,
                            "total_pages": len(reader.pages),
                            "extraction_method": extraction_method,
                            "book": title.replace(".pdf", "").replace("_", " ").replace("-", " "),
                            "chapter": chapter,
                        },
                    )
                    self.documents.append(document)
            except Exception as e:
                logger.warning(f"Failed to extract text from page {page_num}: {e}")

        logger.info(f"Extracted {len(self.documents)} pages from {pdf_path}")
        return self.documents

    def _extract_page_text(self, pdf_path: str, page_index: int, page) -> tuple[str, str]:
        raw_text = page.extract_text() or ""
        cleaned_text = _clean_extracted_text(raw_text)

        if self.use_ocr is True:
            ocr_text = self._ocr_page(pdf_path, page_index)
            if ocr_text and ocr_text.strip():
                return _clean_extracted_text(ocr_text), "ocr"
            if cleaned_text and cleaned_text.strip():
                return cleaned_text, "pypdf"
            return "", "empty"

        if cleaned_text and cleaned_text.strip() and not _looks_like_broken_pdf_text(raw_text):
            return cleaned_text, "pypdf"

        ocr_text = self._ocr_page(pdf_path, page_index)
        if ocr_text and ocr_text.strip():
            return _clean_extracted_text(ocr_text), "ocr"
        if cleaned_text and cleaned_text.strip():
            return cleaned_text, "pypdf-cleaned"
        return "", "empty"

    def _ocr_page(self, pdf_path: str, page_index: int) -> str:
        try:
            import fitz
            import pytesseract
            from PIL import Image
        except Exception as e:
            self._log_ocr_unavailable(e)
            return ""

        try:
            with fitz.open(pdf_path) as document:
                page = document.load_page(page_index)
                scale = self.ocr_dpi / 72
                pixmap = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
                image_bytes = pixmap.tobytes("png")

            with Image.open(io.BytesIO(image_bytes)) as image:
                return pytesseract.image_to_string(
                    image,
                    lang=self.ocr_language,
                    config="--oem 3 --psm 6",
                )
        except Exception as e:
            logger.warning(f"OCR failed for page {page_index + 1}: {e}")
            return ""

    def _log_ocr_unavailable(self, error: Exception) -> None:
        if self._ocr_unavailable_logged:
            return
        self._ocr_unavailable_logged = True
        logger.warning(
            "OCR fallback is unavailable. Install pytesseract, pillow, pymupdf, and the tesseract binary to scan image-only PDFs. "
            f"Original error: {error}"
        )

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
