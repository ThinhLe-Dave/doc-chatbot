import json
import time
import uuid
from contextlib import suppress
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from processor.processor import recommend_documents
from vector_store.db_store import PostgresVectorStore
from vector_store.db_config import DatabaseConfig
from chunker.chunker import Chunker
from chunker.document import Document
from embedding.embedding import get_embedding_model
from utils.db_utils import (
    insert_document,
    store_chunk_batch,
    SQL_UPSERT_JOB,
    SQL_GET_JOB,
    SQL_RECENT_JOBS,
    SQL_CREATE_JOBS_TABLE,
)
from datacollector.crawler import Scraper
from datacollector.pdf_scanner import PDFScanner, preflight_chapters
from utils.config import get_cors_allowed_origins, get_cors_allow_credentials

from fastapi.staticfiles import StaticFiles

from web_frontend.job_manager import JobManager

app = FastAPI(title="Doc Chatbot", version="1.0.0")

CORS_ALLOWED_ORIGINS = get_cors_allowed_origins()
if CORS_ALLOWED_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=CORS_ALLOWED_ORIGINS,
        allow_credentials=get_cors_allow_credentials(),
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

BASE_DIR = Path(__file__).resolve().parent.parent
HTML_PATH = BASE_DIR / "web_frontend" / "index.html"
PDF_DIR = BASE_DIR / "pdfs"
PDF_DIR.mkdir(parents=True, exist_ok=True)

MAX_PDF_SIZE = 100 * 1024 * 1024

if PDF_DIR.exists():
    app.mount("/pdfs", StaticFiles(directory=str(PDF_DIR)), name="pdfs")


class SearchRequest(BaseModel):
    query: str
    categories: Optional[List[str]] = None


@app.on_event("startup")
def _init_jobs_table():
    db_config = DatabaseConfig.from_config_file()
    if not db_config.is_configured():
        return
    store = PostgresVectorStore(config=db_config)
    try:
        store.load()
        with store._conn.cursor() as cur:
            cur.execute(SQL_CREATE_JOBS_TABLE)
        store._conn.commit()
    except Exception:
        with suppress(Exception):
            store.close()


@app.get("/health", response_class=JSONResponse)
def health():
    return {"status": "ok"}


@app.post("/api/search", response_class=JSONResponse)
def search(req: SearchRequest):
    from vector_store.db_config import SearchConfig
    default_config = SearchConfig.from_config_file()
    
    start = time.time()
    try:
        results = recommend_documents(
            query=req.query,
            top_k=10000,
            chunk_k=default_config.chunk_k,
            min_score=default_config.min_score,
            hybrid=default_config.hybrid,
            hybrid_weight=default_config.hybrid_weight,
            categories=req.categories if req.categories else default_config.categories,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    elapsed_ms = int((time.time() - start) * 1000)
    return {
        "results": results,
        "query": req.query,
        "elapsed_ms": elapsed_ms,
        "count": len(results),
    }


@app.get("/", response_class=HTMLResponse)
def index():
    if not HTML_PATH.exists():
        raise HTTPException(status_code=404, detail="Missing UI: web_frontend/index.html")
    return HTML_PATH.read_text(encoding="utf-8")


@app.get("/api/config", response_class=JSONResponse)
def get_config():
    db_config = DatabaseConfig.from_config_file()
    return {
        "database_available": db_config.is_configured(),
    }


@app.get("/api/document/{document_id}", response_class=JSONResponse)
def get_document(document_id: str):
    db_config = DatabaseConfig.from_config_file()
    if not db_config.is_configured():
        raise HTTPException(status_code=503, detail="Database not configured")

    store = PostgresVectorStore(config=db_config)
    try:
        store.load()
        with store._conn.cursor() as cur:
            cur.execute("SELECT id, source, title, path, metadata FROM documents WHERE id = %s", (document_id,))
            row = cur.fetchone()
            if row:
                metadata = dict(row[4]) if row[4] else {}
                return {
                    "id": row[0],
                    "source": row[1],
                    "title": row[2],
                    "path": row[3] if row[3] else [],
                    "categories": metadata.get("categories", []),
                    "metadata": row[4] if row[4] else {},
                }
            raise HTTPException(status_code=404, detail="Document not found")
    finally:
        store.close()


@app.get("/api/categories", response_class=JSONResponse)
def get_categories():
    """List all unique categories across all documents."""
    db_config = DatabaseConfig.from_config_file()
    if not db_config.is_configured():
        raise HTTPException(status_code=503, detail="Database not configured")

    store = PostgresVectorStore(config=db_config)
    try:
        store.load()
        with store._conn.cursor() as cur:
            cur.execute("SELECT metadata FROM documents")
            categories_set = set()
            for row in cur.fetchall():
                metadata = dict(row[0]) if row[0] else {}
                for cat in metadata.get("categories", []):
                    categories_set.add(str(cat).lower())
        return {"categories": sorted(list(categories_set))}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        store.close()


def _check_database_configured():
    """Check if database is configured, raise HTTPException if not."""
    db_config = DatabaseConfig.from_config_file()
    if not db_config.is_configured():
        raise HTTPException(status_code=503, detail="Database not configured")
    return db_config


def _build_doc_path(doc: Document) -> List[str]:
    path = []
    headers = doc.metadata.get("headers")
    if isinstance(headers, list):
        path.extend([h for h in headers if isinstance(h, str)])
    for key in ("book", "chapter", "verse", "section"):
        value = doc.metadata.get(key)
        if value and str(value) not in path:
            path.append(str(value))
    if doc.title and doc.title not in path:
        path.append(doc.title)
    return path


def _persist_job(job_id: str, status: Optional[str], progress: Optional[int] = None, message: Optional[str] = None, error: Optional[str] = None, result: Optional[dict] = None, started_at: Optional[str] = None, finished_at: Optional[str] = None) -> None:
    db_config = DatabaseConfig.from_config_file()
    if not db_config.is_configured():
        return

    store = PostgresVectorStore(config=db_config)
    try:
        store.load()
        now = datetime.now(timezone.utc).isoformat()
        input_payload = {"job_id": job_id}
        with store._conn.cursor() as cur:
            cur.execute(
                SQL_UPSERT_JOB,
                (
                    job_id,
                    None,
                    status,
                    progress,
                    message,
                    error,
                    json.dumps(result) if result is not None else None,
                    started_at,
                    finished_at,
                    json.dumps(input_payload),
                ),
            )
        store._conn.commit()
    except Exception:
        with suppress(Exception):
            store.close()


def _load_job_from_db(job_id: str) -> Optional[dict]:
    db_config = DatabaseConfig.from_config_file()
    if not db_config.is_configured():
        return None
    store = PostgresVectorStore(config=db_config)
    try:
        store.load()
        with store._conn.cursor() as cur:
            cur.execute(SQL_GET_JOB, (job_id,))
            row = cur.fetchone()
            if not row:
                return None
            return {
                "job_id": row[0],
                "job_type": row[1],
                "status": row[2],
                "progress": row[3],
                "message": row[4],
                "error": row[5],
                "result": row[6],
                "created_at": row[7].isoformat() if row[7] else None,
                "updated_at": row[8].isoformat() if row[8] else None,
                "started_at": row[9].isoformat() if row[9] else None,
                "finished_at": row[10].isoformat() if row[10] else None,
            }
    except Exception:
        return None
    finally:
        with suppress(Exception):
            store.close()


def _store_documents(job_manager, job_id: str, documents: List[Document]) -> int:
    db_config = DatabaseConfig.from_config_file()
    if not db_config.is_configured():
        raise RuntimeError("Database not configured")

    model = get_embedding_model()
    store = PostgresVectorStore(config=db_config)
    store.load()
    conn = store._conn

    seen_docs = set()
    batch_chunks = []
    chunker = Chunker()
    total_chunks = 0

    for doc in documents:
        doc_path = _build_doc_path(doc)
        if doc.id not in seen_docs:
            seen_docs.add(doc.id)
            with conn.cursor() as cur:
                insert_document(cur, doc.id, doc.source, doc.title, doc_path, doc.metadata)

        for chunk in chunker.create_chunks_from_document(doc):
            batch_chunks.append(chunk)
            if len(batch_chunks) >= 64:
                store_chunk_batch(conn, batch_chunks, model)
                total_chunks += len(batch_chunks)
                conn.commit()
                batch_chunks = []

    if batch_chunks:
        store_chunk_batch(conn, batch_chunks, model)
        total_chunks += len(batch_chunks)

    conn.commit()
    store.close()
    return total_chunks


def _run_scrape(job_id: str, url: str, max_pages: int):
    started_at = datetime.now(timezone.utc).isoformat()
    job_manager = JobManager()
    _persist_job(job_id, status="running", progress=0, message="Initializing scraper...", started_at=started_at)
    job_manager.update_job(job_id, status="running", progress=0, message="Initializing scraper...")

    try:
        db_config = DatabaseConfig.from_config_file()
        if not db_config.is_configured():
            raise RuntimeError("Database not configured")

        scraper = Scraper(base_url=url)
        scraper.crawl(max_pages=max_pages)

        progress_message = f"Crawled {len(scraper.scraped_data)} pages, building chunks..."
        job_manager.update_job(job_id, status="running", progress=50, message=progress_message)
        _persist_job(job_id, status="running", progress=50, message=progress_message)

        documents = [Document.from_dict(item) for item in scraper.scraped_data]
        total_chunks = _store_documents(job_manager, job_id, documents)

        finished_at = datetime.now(timezone.utc).isoformat()
        result = {"total_chunks": total_chunks, "pages_scraped": len(scraper.scraped_data)}
        message = f"Scraping completed. {total_chunks} chunks saved to database."
        job_manager.update_job(job_id, status="completed", progress=100, result=result, message=message)
        _persist_job(job_id, status="completed", progress=100, message=message, result=result, finished_at=finished_at)
    except Exception as e:
        finished_at = datetime.now(timezone.utc).isoformat()
        job_manager.update_job(job_id, status="failed", error=str(e))
        _persist_job(job_id, status="failed", error=str(e), finished_at=finished_at)


def _run_pdf_scan_from_bytes(job_id: str, file_content: bytes, use_ocr: bool, ocr_language: str, ocr_dpi: int):
    temp_path = PDF_DIR / f"{uuid.uuid4().hex}.pdf"
    started_at = datetime.now(timezone.utc).isoformat()
    job_manager = JobManager()
    _persist_job(job_id, status="running", progress=0, message="Initializing PDF scanner...", started_at=started_at)
    job_manager.update_job(job_id, status="running", progress=0, message="Initializing PDF scanner...")

    try:
        db_config = DatabaseConfig.from_config_file()
        if not db_config.is_configured():
            raise RuntimeError("Database not configured")

        with open(temp_path, "wb") as f:
            f.write(file_content)

        scanner = PDFScanner(use_ocr=use_ocr, ocr_language=ocr_language, ocr_dpi=ocr_dpi)
        documents = scanner.scan_pdf(str(temp_path))

        progress_message = f"Scanned {len(documents)} pages, building chunks..."
        job_manager.update_job(job_id, status="running", progress=50, message=progress_message)
        _persist_job(job_id, status="running", progress=50, message=progress_message)

        total_chunks = _store_documents(job_manager, job_id, documents)

        finished_at = datetime.now(timezone.utc).isoformat()
        result = {"total_chunks": total_chunks, "pages_processed": len(documents)}
        message = f"PDF scan completed. {total_chunks} chunks saved to database."
        job_manager.update_job(job_id, status="completed", progress=100, result=result, message=message)
        _persist_job(job_id, status="completed", progress=100, message=message, result=result, finished_at=finished_at)
    except Exception as e:
        finished_at = datetime.now(timezone.utc).isoformat()
        job_manager.update_job(job_id, status="failed", error=str(e))
        _persist_job(job_id, status="failed", error=str(e), finished_at=finished_at)
    finally:
        with suppress(OSError):
            temp_path.unlink()


@app.post("/api/scrape", response_class=JSONResponse)
def scrape(
    background_tasks: BackgroundTasks,
    url: str = Form(...),
    max_pages: int = Form(100),
    sitemap_first: bool = Form(False),
    force: bool = Form(False),
):
    """Start a background scrape job for a website."""
    _check_database_configured()

    if max_pages > 500:
        max_pages = 500

    job_manager = JobManager()
    job_id = job_manager.create_job()
    _persist_job(job_id, status="pending", progress=0, message="Queued", input_payload={"url": url, "max_pages": max_pages, "sitemap_first": sitemap_first, "force": force})

    background_tasks.add_task(_run_scrape, job_id, url, max_pages, sitemap_first, force)

    return {"status": "started", "job_id": job_id}


@app.post("/api/pdf-scan", response_class=JSONResponse)
async def pdf_scan(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    use_ocr: bool = Form(False),
    ocr_language: str = Form("eng"),
    ocr_dpi: int = Form(200),
):
    """Start a background PDF scan job."""
    _check_database_configured()

    if file.content_type not in ("application/pdf", "application/octet-stream"):
        raise HTTPException(status_code=400, detail="Invalid file type. Only PDF files are allowed.")

    job_manager = JobManager()
    job_id = job_manager.create_job()
    _persist_job(job_id, status="pending", progress=0, message="Queued")

    file_content = await file.read()
    if len(file_content) > MAX_PDF_SIZE:
        job_manager.update_job(job_id, status="failed", error="File too large. Maximum size is 100MB.")
        _persist_job(job_id, status="failed", error="File too large. Maximum size is 100MB.")
        raise HTTPException(status_code=413, detail="File too large. Maximum size is 100MB.")

    background_tasks.add_task(_run_pdf_scan_from_bytes, job_id, file_content, use_ocr, ocr_language, ocr_dpi)

    return {"status": "started", "job_id": job_id}


@app.get("/api/jobs/{job_id}", response_class=JSONResponse)
def get_job_status(job_id: str):
    """Get the status of a background job."""
    db_record = _load_job_from_db(job_id)
    if db_record:
        return db_record

    job_manager = JobManager()
    job = job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job_manager.to_dict(job)
