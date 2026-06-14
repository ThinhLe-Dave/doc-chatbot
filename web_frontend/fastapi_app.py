import time
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
from utils.db_utils import insert_document, store_chunk_batch
from datacollector.crawler import Scraper
from datacollector.pdf_scanner import PDFScanner

from fastapi.staticfiles import StaticFiles

from web_frontend.job_manager import JobManager

app = FastAPI(title="Doc Chatbot", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent.parent
HTML_PATH = BASE_DIR / "web_frontend" / "index.html"
PDF_DIR = BASE_DIR / "pdfs"
PDF_DIR.mkdir(parents=True, exist_ok=True)

MAX_PDF_SIZE = 100 * 1024 * 1024  # 100MB

if PDF_DIR.exists():
    app.mount("/pdfs", StaticFiles(directory=str(PDF_DIR)), name="pdfs")


class SearchRequest(BaseModel):
    query: str
    categories: Optional[List[str]] = None


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


def _store_documents(job_manager: JobManager, job_id: str, documents: List[Document]) -> int:
    """Process and store documents in database. Returns total chunk count."""
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
    """Background task to scrape a website and build chunks."""
    job_manager = JobManager()

    try:
        job_manager.update_job(job_id, status="running", progress=0, message="Initializing scraper...")

        db_config = DatabaseConfig.from_config_file()
        if not db_config.is_configured():
            raise RuntimeError("Database not configured")

        scraper = Scraper(base_url=url)
        scraper.crawl(max_pages=max_pages)

        job_manager.update_job(job_id, status="running", progress=50, message=f"Crawled {len(scraper.scraped_data)} pages, building chunks...")

        documents = [Document.from_dict(item) for item in scraper.scraped_data]
        total_chunks = _store_documents(job_manager, job_id, documents)

        job_manager.update_job(
            job_id,
            status="completed",
            progress=100,
            result={"total_chunks": total_chunks, "pages_scraped": len(scraper.scraped_data)},
            message=f"Scraping completed. {total_chunks} chunks saved to database.",
        )
    except Exception as e:
        job_manager.update_job(job_id, status="failed", error=str(e))


def _run_pdf_scan_from_bytes(job_id: str, file_content: bytes, filename: str, use_ocr: bool, ocr_language: str, ocr_dpi: int):
    """Background task to scan a PDF from bytes and build chunks."""
    job_manager = JobManager()

    try:
        job_manager.update_job(job_id, status="running", progress=0, message="Initializing PDF scanner...")

        temp_path = PDF_DIR / filename
        with open(temp_path, "wb") as f:
            f.write(file_content)

        db_config = DatabaseConfig.from_config_file()
        if not db_config.is_configured():
            raise RuntimeError("Database not configured")

        scanner = PDFScanner(use_ocr=use_ocr, ocr_language=ocr_language, ocr_dpi=ocr_dpi)
        documents = scanner.scan_pdf(str(temp_path))

        job_manager.update_job(job_id, status="running", progress=50, message=f"Scanned {len(documents)} pages, building chunks...")

        total_chunks = _store_documents(job_manager, job_id, documents)

        job_manager.update_job(
            job_id,
            status="completed",
            progress=100,
            result={"total_chunks": total_chunks, "pages_processed": len(documents)},
            message=f"PDF scan completed. {total_chunks} chunks saved to database.",
        )
    except Exception as e:
        job_manager.update_job(job_id, status="failed", error=str(e))


@app.post("/api/scrape", response_class=JSONResponse)
def scrape(
    background_tasks: BackgroundTasks,
    url: str = Form(...),
    max_pages: int = Form(100),
):
    """Start a background scrape job for a website."""
    _check_database_configured()

    if max_pages > 500:
        max_pages = 500

    job_manager = JobManager()
    job_id = job_manager.create_job()

    background_tasks.add_task(_run_scrape, job_id, url, max_pages)

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

    file_content = await file.read()
    if len(file_content) > MAX_PDF_SIZE:
        job_manager.update_job(job_id, status="failed", error="File too large. Maximum size is 100MB.")
        raise HTTPException(status_code=413, detail="File too large. Maximum size is 100MB.")

    background_tasks.add_task(_run_pdf_scan_from_bytes, job_id, file_content, file.filename, use_ocr, ocr_language, ocr_dpi)

    return {"status": "started", "job_id": job_id}


@app.get("/api/jobs/{job_id}", response_class=JSONResponse)
def get_job_status(job_id: str):
    """Get the status of a background job."""
    job_manager = JobManager()
    job = job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job_manager.to_dict(job)