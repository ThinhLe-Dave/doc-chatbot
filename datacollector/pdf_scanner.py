import io
import json
import logging
import os
import re
import shutil
import unicodedata
from pathlib import Path
from typing import List, Optional, Tuple

from pypdf import PdfReader

from chunker.document import Document, compute_content_hash
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
_CHUNK_PREVIEW_MAX_PAGES = 50


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
                page = int(part)
            except ValueError:
                continue
            if 1 <= page <= total_pages:
                pages.add(page)
    return pages if pages else None


def _page_heading(text: str, page_num: int) -> Optional[dict]:
    chapter = _extract_chapter_info(text)
    if not chapter:
        return None
    return {"chapter": chapter, "page": page_num}


def preflight_chapters(pdf_path: str, max_pages: Optional[int] = _CHUNK_PREVIEW_MAX_PAGES) -> List[dict]:
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    try:
        reader = PdfReader(pdf_path)
    except Exception as exc:
        raise RuntimeError(f"Failed to read PDF {pdf_path}: {exc}") from exc

    count = 0
    chapters: List[dict] = []
    last_chapter: Optional[str] = None
    for page_num, page in enumerate(reader.pages, start=1):
        if max_pages is not None and count >= max_pages:
            break
        count += 1
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        if not text.strip():
            continue
        chapter = _extract_chapter_info(text)
        if chapter and chapter != last_chapter:
            chapters.append({"chapter": chapter, "page": page_num, "pages_span": page_num})
            last_chapter = chapter
        elif last_chapter is not None and chapters:
            chapters[-1]["pages_span"] = page_num
    return chapters


def compute_chapter_pages(pdf_path: str, chapters: Optional[List[str]] = None) -> tuple[Optional[set], dict[str, set], int]:
    """Return allowed page set for given chapter labels, or None if all pages.
    
    Also returns the raw chapter->pages mapping and total page count.
    When ``chapters`` is empty/None, returns ``None`` for allowed pages, meaning
    no filtering should be applied.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    try:
        reader = PdfReader(pdf_path)
    except Exception as exc:
        raise RuntimeError(f"Failed to read PDF {pdf_path}: {exc}") from exc

    total_pages = len(reader.pages)
    if not chapters:
        return None, {}, total_pages

    chapter_pages: dict[str, set] = {}
    for page_num, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        chapter = _extract_chapter_info(text)
        if chapter:
            chapter_pages.setdefault(chapter, set()).add(page_num)

    allowed_pages: set = set()
    for label in chapters:
        allowed_pages.update(chapter_pages.get(label, set()))

    if not allowed_pages:
        return None, chapter_pages, total_pages

    return allowed_pages, chapter_pages, total_pages


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


def _page_quality(text: str) -> dict:
    clean = text.strip()
    words = clean.split()
    word_count = len(words)
    char_count = len(clean)
    tokens = _alpha_tokens(clean)
    suspicious = sum(1 for t in tokens if _is_suspicious_token(t))
    suspicious_ratio = suspicious / len(tokens) if tokens else 0.0
    probably_broken = word_count > 10 and suspicious_ratio > 0.06

    return {
        "word_count": word_count,
        "char_count": char_count,
        "suspicious_token_ratio": round(suspicious_ratio, 4),
        "probably_broken": probably_broken,
        "near_empty": word_count == 0,
    }


class PDFScanner(DataCollector):
    def __init__(
        self,
        output_file: str = "pdf_data.json",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        use_ocr: Optional[bool] = None,
        ocr_language: str = "eng",
        ocr_dpi: int = 200,
        ocr_preprocess: bool = False,
    ):
        super().__init__(output_file)
        self.chunker = Chunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.use_ocr = use_ocr
        self.ocr_language = ocr_language
        self.ocr_dpi = ocr_dpi
        self.ocr_preprocess = ocr_preprocess
        self._ocr_unavailable_logged = False

    def collect(self, source: str, **kwargs) -> List[Document]:
        return self.scan(source, **kwargs)

    def scan(self, pdf_path: str = None, **kwargs) -> List[Document]:
        """Alias for scan_pdf."""
        source = kwargs.get("source")
        chapters = kwargs.get("chapters")
        page_range = kwargs.get("page_range")
        original_filename = kwargs.get("original_filename")
        return self.scan_pdf(pdf_path, source=source, chapters=chapters, page_range=page_range, original_filename=original_filename)

    def _compute_document_hash(self, pdf_path: str) -> str:
        try:
            with open(pdf_path, "rb") as f:
                return compute_content_hash(f.read().decode("latin-1", errors="ignore"))
        except Exception:
            return compute_content_hash(pdf_path)

    def scan_pdf_chapters(self, pdf_path: str, source: str = None, original_filename: Optional[str] = None, chapters: Optional[List[str]] = None) -> List[Document]:
        """Scan only pages matching specific chapters."""
        return self.scan_pdf(pdf_path, source=source, original_filename=original_filename, chapters=chapters, page_range=None)

    def scan_pdf(
        self,
        pdf_path: str,
        source: str = None,
        chapters: Optional[List[str]] = None,
        page_range: Optional[str] = None,
        original_filename: Optional[str] = None,
    ) -> List[Document]:
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        title = Path(original_filename).stem if original_filename else Path(pdf_path).stem
        document_hash = self._compute_document_hash(pdf_path)

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
        total_pages = len(reader.pages)
        allowed_pages: Optional[set] = None
        skipped = 0

        if chapters:
            allowed_pages, _, total_pages = compute_chapter_pages(pdf_path, chapters)
            if allowed_pages is None:
                logger.info(f"No chapters matched the selection {chapters}; scanning all pages from {pdf_path}")

        if page_range:
            range_pages = _parse_page_ranges(page_range, total_pages)
            if range_pages:
                if allowed_pages is not None:
                    allowed_pages.intersection_update(range_pages)
                else:
                    allowed_pages = range_pages
            if not allowed_pages:
                allowed_pages = None
                logger.info(f"No pages matched the range {page_range}; scanning all pages from {pdf_path}")

        pre_extracted: Optional[dict[int, tuple[str, Optional[str]]]] = None
        if allowed_pages is not None:
            pre_extracted = {}
            for page_num, page in enumerate(reader.pages, start=1):
                if page_num in allowed_pages:
                    try:
                        raw_text = page.extract_text() or ""
                    except Exception:
                        raw_text = ""
                    pre_extracted[page_num] = (raw_text, None)

        self.documents = []
        for page_num, page in enumerate(reader.pages, start=1):
            try:
                if allowed_pages is not None and page_num not in allowed_pages:
                    skipped += 1
                    continue
                if pre_extracted and pre_extracted.get(page_num) is not None:
                    raw_text, _ = pre_extracted[page_num]
                    text, extraction_method = self._extract_page_text(pdf_path, page_num - 1, page, raw_text=raw_text)
                else:
                    text, extraction_method = self._extract_page_text(pdf_path, page_num - 1, page)
                if text and text.strip():
                    page_hash = compute_content_hash(text)
                    chapter = _extract_chapter_info(text)
                    metadata = {
                        "page": page_num,
                        "total_pages": total_pages,
                        "extraction_method": extraction_method,
                        "book": re.sub(r'\s+', ' ', title.replace(".pdf", "").replace("_", " ").replace("-", " ")).strip(),
                        "chapter": chapter,
                        "source_hash": document_hash,
                        "page_hash": page_hash,
                    }
                    if extraction_method == "ocr":
                        metadata["ocr_confidence"] = self._get_last_ocr_confidence()
                    metadata.update(_page_quality(text))
                    document = Document.create(
                        source=f"{base_source}#page={page_num}",
                        title=f"{title} (page {page_num})",
                        content=text,
                        metadata=metadata,
                    )
                    self.documents.append(document)
            except Exception as e:
                logger.warning(f"Failed to extract text from page {page_num}: {e}")

        if skipped:
            logger.info(f"Skipped {skipped} pages for {pdf_path}")
        logger.info(f"Extracted {len(self.documents)} pages from {pdf_path}")
        return self.documents

    def _extract_page_text(self, pdf_path: str, page_index: int, page, raw_text: Optional[str] = None) -> tuple[str, str]:
        if raw_text is None:
            raw_text = page.extract_text() or ""
        cleaned_text = _clean_extracted_text(raw_text)

        if self.use_ocr is True:
            ocr_text, ocr_confidence = self._ocr_page(pdf_path, page_index)
            if ocr_text and ocr_text.strip():
                return _clean_extracted_text(ocr_text), "ocr"
            if cleaned_text and cleaned_text.strip():
                return cleaned_text, "pypdf"
            return "", "empty"

        if cleaned_text and cleaned_text.strip() and not _looks_like_broken_pdf_text(raw_text):
            return cleaned_text, "pypdf"

        if len(cleaned_text.strip().split()) >= 8:
            return cleaned_text, "pypdf-cleaned"

        ocr_text, ocr_confidence = self._ocr_page(pdf_path, page_index)
        if ocr_text and ocr_text.strip():
            return _clean_extracted_text(ocr_text), "ocr"
        if cleaned_text and cleaned_text.strip():
            return cleaned_text, "pypdf-cleaned"
        return "", "empty"

    def _preprocess_ocr_image(self, image):
        try:
            from PIL import Image, ImageOps, ImageFilter

            image = image.convert("L")
            if self.ocr_preprocess:
                image = ImageOps.autocontrast(image)
                image = image.point(lambda p: 255 if p > 200 else 0)
                image = image.filter(ImageFilter.MedianFilter())
            return image
        except Exception:
            return image

    def _get_last_ocr_confidence(self) -> Optional[float]:
        return getattr(self, "_last_ocr_confidence", None)

    def _set_last_ocr_confidence(self, value: Optional[float]) -> None:
        self._last_ocr_confidence = value

    def _ocr_page(self, pdf_path: str, page_index: int) -> tuple[str, float]:
        try:
            import fitz
            import pytesseract
            from PIL import Image
        except Exception as e:
            self._log_ocr_unavailable(e)
            return "", 0.0

        try:
            with fitz.open(pdf_path) as document:
                page = document.load_page(page_index)
                scale = self.ocr_dpi / 72
                pixmap = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
                image_bytes = pixmap.tobytes("png")

            image = Image.open(io.BytesIO(image_bytes))
            image = self._preprocess_ocr_image(image)
            data = pytesseract.image_to_data(
                image,
                lang=self.ocr_language,
                config="--oem 3 --psm 6",
                output_type=pytesseract.Output.DICT,
            )
            conf_values = [c for c in data.get("conf", []) if str(c).strip() and str(c) != "-1"]
            if conf_values:
                confidence = float(sum(int(c) for c in conf_values)) / len(conf_values) / 100.0
                confidence = max(0.0, min(1.0, confidence))
                self._set_last_ocr_confidence(round(confidence, 4))
            else:
                confidence = 0.0
            return pytesseract.image_to_string(
                image,
                lang=self.ocr_language,
                config="--oem 3 --psm 6",
            ), confidence
        except Exception as e:
            logger.warning(f"OCR failed for page {page_index + 1}: {e}")
            return "", 0.0

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
