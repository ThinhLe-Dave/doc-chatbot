from fastapi import FastAPI, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from typing import Optional
import asyncio
import os
import uuid
import tempfile
from pathlib import Path
from datetime import datetime, timezone

from processor.processor import recommend_documents
from utils.config import get_cors_allowed_origins, get_cors_allow_credentials

app = FastAPI(title="Doc Chatbot", description="Semantic document search and chatbot interface")

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_cors_allowed_origins() or ["*"],
    allow_credentials=get_cors_allow_credentials(),
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="web_frontend/static"), name="static")
app.mount("/pdfs", StaticFiles(directory="pdfs"), name="pdfs")

UPLOAD_DIR = Path(tempfile.gettempdir()) / "doc_chatbot_uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
_jobs: dict = {}
_running_tasks: set = set()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _update_job(job_id: str, **kwargs):
    if job_id in _jobs:
        _jobs[job_id].update(kwargs)
    _jobs[job_id]["updated_at"] = _now_iso()


async def _run_scrape_job(job_id: str, url: str, limit: int, sitemap_first: bool, force: bool, no_robots: bool):
    _update_job(job_id, status="running", message="Crawling website...", progress=10)
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, scrape_blocking, url, limit, sitemap_first, force, no_robots)
        _update_job(job_id, status="completed", progress=100, message="Scraping complete", result=result)
    except Exception as e:
        _update_job(job_id, status="failed", message=str(e), error=str(e))


def scrape_blocking(url: str, limit: int, sitemap_first: bool, force: bool, no_robots: bool) -> dict:
    from app import scrape_website
    return scrape_website(url, limit=limit, sitemap_first=sitemap_first, force=force, no_robots=no_robots)


async def _run_pdf_scan_job(job_id: str, file_path: str, chapters: Optional[list]):
    _update_job(job_id, status="running", message="Scanning PDF...", progress=10)
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, scan_blocking, file_path, chapters)
        _update_job(job_id, status="completed", progress=100, message="PDF scan complete", result=result)
    except Exception as e:
        _update_job(job_id, status="failed", message=str(e), error=str(e))
    finally:
        try:
            if os.path.exists(file_path):
                os.unlink(file_path)
        except Exception:
            pass


def scan_blocking(file_path: str, chapters: Optional[list]) -> dict:
    from app import scan_pdf
    return scan_pdf(path=file_path, chapters=chapters)


@app.get("/", response_class=HTMLResponse)
async def index():
    with open("web_frontend/static/index.html", "r", encoding="utf-8") as f:
        return f.read()


@app.get("/api/health")
async def health():
    return {"status": "ok"}


@app.post("/api/search")
async def search(request: Request):
    body = await request.json()
    query = body.get("query", "").strip()
    if not query:
        return JSONResponse({"error": "Missing query parameter"}, status_code=400)

    try:
        page = int(body.get("page", 1))
        page_size = int(body.get("page_size", 10))
        chunk_k = int(body.get("chunk_k", 3))
        min_score = float(body.get("min_score", 0.01))
        hybrid = bool(body.get("hybrid", True))
        hybrid_weight = float(body.get("hybrid_weight", 0.1))
        categories = body.get("categories")
    except (ValueError, TypeError):
        return JSONResponse({"error": "Invalid parameter format"}, status_code=400)

    try:
        all_results = recommend_documents(
            query=query,
            top_k=1000,
            chunk_k=chunk_k,
            min_score=min_score,
            hybrid=hybrid,
            hybrid_weight=hybrid_weight,
            categories=categories,
        )
        total = len(all_results)
        total_pages = max(1, (total + page_size - 1) // page_size)
        page = max(1, min(page, total_pages))
        start = (page - 1) * page_size
        end = start + page_size
        paged = all_results[start:end]
        return JSONResponse({
            "query": query,
            "results": paged,
            "total": total,
            "page": page,
            "page_size": page_size,
            "total_pages": total_pages,
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/scrape")
async def scrape_endpoint(request: Request):
    body = await request.json()
    url = (body.get("url") or "").strip()
    if not url:
        return JSONResponse({"error": "Missing url parameter"}, status_code=400)

    limit = int(body.get("limit", 1000))
    sitemap_first = bool(body.get("sitemap_first", False))
    force = bool(body.get("force", False))
    no_robots = bool(body.get("no_robots", False))

    job_id = str(uuid.uuid4())
    _jobs[job_id] = {
        "id": job_id,
        "type": "scrape",
        "status": "pending",
        "progress": 0,
        "message": "Queued",
        "input": {"url": url, "limit": limit, "sitemap_first": sitemap_first, "force": force, "no_robots": no_robots},
        "result": {},
        "error": None,
        "created_at": _now_iso(),
        "updated_at": _now_iso(),
    }
    task = asyncio.create_task(_run_scrape_job(job_id, url, limit, sitemap_first, force, no_robots))
    _running_tasks.add(task)
    task.add_done_callback(lambda t: _running_tasks.discard(t))
    return JSONResponse({"job_id": job_id, "status": "pending"})


@app.post("/api/pdf-scan")
async def pdf_scan_endpoint(request: Request):
    body = await request.json()
    path = (body.get("path") or "").strip()
    if not path:
        return JSONResponse({"error": "Missing path parameter"}, status_code=400)

    chapters = body.get("chapters") or None

    job_id = str(uuid.uuid4())
    _jobs[job_id] = {
        "id": job_id,
        "type": "pdf_scan",
        "status": "pending",
        "progress": 0,
        "message": "Queued",
        "input": {"path": path, "chapters": chapters},
        "result": {},
        "error": None,
        "created_at": _now_iso(),
        "updated_at": _now_iso(),
    }
    task = asyncio.create_task(_run_pdf_scan_job(job_id, path, chapters))
    _running_tasks.add(task)
    task.add_done_callback(lambda t: _running_tasks.discard(t))
    return JSONResponse({"job_id": job_id, "status": "pending"})


@app.post("/api/pdf-scan-upload")
async def pdf_scan_upload(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        return JSONResponse({"error": "Uploaded file must be a PDF"}, status_code=400)

    original_name = Path(file.filename).name
    base_name = Path(file.filename).stem
    suffix = Path(file.filename).suffix
    tmp_path = UPLOAD_DIR / original_name
    counter = 1
    while tmp_path.exists():
        tmp_path = UPLOAD_DIR / f"{base_name}_{counter}{suffix}"
        counter += 1
    content = await file.read()
    with open(tmp_path, "wb") as f:
        f.write(content)

    job_id = str(uuid.uuid4())
    _jobs[job_id] = {
        "id": job_id,
        "type": "pdf_scan",
        "status": "pending",
        "progress": 0,
        "message": "Queued",
        "input": {"filename": original_name, "chapters": None},
        "result": {},
        "error": None,
        "created_at": _now_iso(),
        "updated_at": _now_iso(),
    }
    task = asyncio.create_task(_run_pdf_scan_job(job_id, str(tmp_path), None))
    _running_tasks.add(task)
    task.add_done_callback(lambda t: _running_tasks.discard(t))
    return JSONResponse({"job_id": job_id, "status": "pending"})


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        return JSONResponse({"error": "Job not found"}, status_code=404)
    return JSONResponse(job)


@app.get("/api/jobs")
async def list_jobs(limit: int = 20):
    jobs = sorted(_jobs.values(), key=lambda j: j["created_at"], reverse=True)[:limit]
    return JSONResponse(jobs)
