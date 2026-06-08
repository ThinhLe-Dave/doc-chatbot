import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from processor.processor import recommend_documents

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


class SearchRequest(BaseModel):
    query: str
    input: str = "database/research_data.json"
    top_k: int = 10
    chunk_k: int = 3
    min_score: float = 0.01
    hybrid: bool = True
    hybrid_weight: float = 0.4


@app.get("/health", response_class=JSONResponse)
def health():
    return {"status": "ok"}


@app.post("/api/search", response_class=JSONResponse)
def search(req: SearchRequest):
    start = time.time()
    try:
        results = recommend_documents(
            query=req.query,
            input_file=req.input,
            top_k=req.top_k,
            chunk_k=req.chunk_k,
            min_score=req.min_score,
            hybrid=req.hybrid,
            hybrid_weight=req.hybrid_weight,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
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
    return {
        "defaults": {
            "input": "database/research_data.json",
            "top_k": 10,
            "chunk_k": 3,
            "min_score": 0.01,
            "hybrid": True,
            "hybrid_weight": 0.4,
        }
    }
