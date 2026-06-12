import time
from pathlib import Path
from typing import List, Optional, Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from processor.processor import recommend_documents
from vector_store.db_store import PostgresVectorStore
from vector_store.db_config import DatabaseConfig

from fastapi.staticfiles import StaticFiles

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
                return {
                    "id": row[0],
                    "source": row[1],
                    "title": row[2],
                    "path": row[3] if row[3] else [],
                    "metadata": row[4] if row[4] else {},
                }
            raise HTTPException(status_code=404, detail="Document not found")
    finally:
        store.close()
