from fastapi import FastAPI, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from typing import Optional, AsyncIterator
import asyncio
import json
import os
import uuid
import tempfile
from pathlib import Path
from datetime import datetime, timezone

from processor.processor import recommend_documents, ask_question
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


async def _chat_stream_generator(stream_iterator, job_id: str) -> AsyncIterator[str]:
    """SSE generator for streaming chat responses."""
    try:
        for token in stream_iterator:
            yield f"data: {json.dumps({'content': token, 'job_id': job_id})}\n\n"
        yield "data: [DONE]\n\n"
    except Exception as e:
        _update_job(job_id, status="failed", error=str(e), message=f"Generation error: {e}")
        yield f"data: {json.dumps({'error': str(e), 'job_id': job_id})}\n\n"


async def _run_chat_job(job_id: str, query: str, top_k: int, chunk_k: int, min_score: float, hybrid: bool, hybrid_weight: float, categories: Optional[list]):
    _update_job(job_id, status="running", message="Thinking...", progress=30)
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: ask_question(
                query=query,
                top_k=top_k,
                chunk_k=chunk_k,
                min_score=min_score,
                hybrid=hybrid,
                hybrid_weight=hybrid_weight,
                categories=categories,
            ),
        )
        _update_job(job_id, status="completed", progress=100, message="Done", result=result, answer=result.get("answer", ""))
    except Exception as e:
        _update_job(job_id, status="failed", message=str(e), error=str(e))


@app.post("/api/chat")
async def chat_endpoint(request: Request):
    body = await request.json()
    query = (body.get("query") or "").strip()
    if not query:
        return JSONResponse({"error": "Missing query parameter"}, status_code=400)

    try:
        top_k = int(body.get("top_k", 10))
        chunk_k = int(body.get("chunk_k", 3))
        min_score = float(body.get("min_score", 0.01))
        hybrid = bool(body.get("hybrid", True))
        hybrid_weight = float(body.get("hybrid_weight", 0.1))
        categories = body.get("categories")
    except (ValueError, TypeError):
        return JSONResponse({"error": "Invalid parameter format"}, status_code=400)

    job_id = str(uuid.uuid4())
    _jobs[job_id] = {
        "id": job_id,
        "type": "chat",
        "status": "pending",
        "progress": 0,
        "message": "Queued",
        "input": {"query": query, "top_k": top_k},
        "result": {},
        "answer": "",
        "error": None,
        "created_at": _now_iso(),
        "updated_at": _now_iso(),
    }
    task = asyncio.create_task(_run_chat_job(job_id, query, top_k, chunk_k, min_score, hybrid, hybrid_weight, categories))
    _running_tasks.add(task)
    task.add_done_callback(lambda t: _running_tasks.discard(t))
    return JSONResponse({"job_id": job_id, "status": "pending"})


@app.get("/api/chat-stream/{job_id}")
async def chat_stream(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        return JSONResponse({"error": "Job not found"}, status_code=404)
    
    if job["status"] == "completed":
        return StreamingResponse(
            _chat_stream_generator(iter([job["answer"] or ""]), job_id),
            media_type="text/event-stream",
        )
    
    if job["status"] == "failed":
        return StreamingResponse(
            _chat_stream_generator(iter([f"Error: {job['error']}"]), job_id),
            media_type="text/event-stream",
        )
    
    async def wait_and_stream():
        while job["status"] == "pending" or job["status"] == "running":
            await asyncio.sleep(0.1)
            if job["status"] == "completed":
                async for token in _stream_completed_job(job_id):
                    yield token
                return
            if job["status"] == "failed":
                yield f"data: {json.dumps({'error': job['error'], 'job_id': job_id})}\n\n"
                yield "data: [DONE]\n\n"
                return
    
    async def _stream_completed_job(jid: str):
        result = _jobs.get(jid, {})
        answer = result.get("answer", "") or ""
        for i in range(0, len(answer), 5):
            yield f"data: {json.dumps({'content': answer[i:i+5], 'job_id': jid})}\n\n"
        yield "data: [DONE]\n\n"
    
    return StreamingResponse(wait_and_stream(), media_type="text/event-stream")


@app.post("/api/chat-direct")
async def chat_direct(request: Request):
    body = await request.json()
    query = (body.get("query") or "").strip()
    if not query:
        return JSONResponse({"error": "Missing query parameter"}, status_code=400)

    try:
        top_k = int(body.get("top_k", 10))
        chunk_k = int(body.get("chunk_k", 3))
        min_score = float(body.get("min_score", 0.01))
        hybrid = bool(body.get("hybrid", True))
        hybrid_weight = float(body.get("hybrid_weight", 0.1))
        categories = body.get("categories")
    except (ValueError, TypeError):
        return JSONResponse({"error": "Invalid parameter format"}, status_code=400)

    try:
        result = ask_question(
            query=query,
            top_k=top_k,
            chunk_k=chunk_k,
            min_score=min_score,
            hybrid=hybrid,
            hybrid_weight=hybrid_weight,
            categories=categories,
            stream=False,
        )
        return JSONResponse({
            "query": query,
            "answer": result.get("answer", ""),
            "sources": result.get("sources", []),
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
