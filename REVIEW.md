# REVIEW.md

## Summary

- Core RAG pipeline is functional and reasonably structured.
- Frontend/backend separation is clean: FastAPI serves API + static HTML, frontend is decoupled via configurable `apiBase`.
- Debug logging uses centralized category-based filtering via `config/config.cfg` `[logging] categories` (no env vars needed).
- Refactoring complete: `app.py` reduced from 366→102 lines. Processing logic moved to `processor/processor.py`.
- Vector store abstraction added (`vector_store/` module) with numpy-backed `VectorIndex` and `VectorStore`.
- Content-derived chunk categories implemented via lightweight TF keyword extractor (`chunker/keywords.py`).
- Category filtering exposed in CLI (`-c`/`--category`) and wired into `recommend_documents()`.

---

## Critical / High

### 1. Frontend FastAPI does not expose `categories` parameter
`web_frontend/fastapi_app.py:25-32` `SearchRequest` lacks a `categories` field. The UI cannot filter by category even though the backend supports it.

**Recommendation**: add `categories: Optional[List[str]] = None` to `SearchRequest` and pass it through to `recommend_documents()`.

---

### 2. `scrape` CLI uses `typer.Abort` (CLI-only, safe)
`app.py:233` raises `typer.Abort` in the `scrape` command. This is CLI-only and **not** reachable from FastAPI, so it's acceptable.

---

### 3. CORS is wide-open
`web_frontend/fastapi_app.py:13-19` allows `*` origins. Currently `allow_credentials=False` which mitigates some risk. Methods restricted to `GET` and `POST`.

**Recommendation**: restrict to same-origin or a whitelist in production.

---

## Medium

### 4. `HF_TOKEN` defined but never injected
`utils/config.py` reads `[hf] token` from `config/config.cfg`, and `embedding/embedding.py` defines `_get_hf_token()`, but the token is **not** passed to `SentenceTransformer(...)` or set as `os.environ["HF_TOKEN"]` before model load.

**Recommendation**: set `os.environ["HF_TOKEN"]` in `utils/config.py:_load()` or pass `use_auth_token=...` to `SentenceTransformer()`.

---

### 5. Query relevance for abstract / short queries
Semantic search on `paraphrase-multilingual-MiniLM-L12-v2` maps short queries like "AI" or "how AI should be used" to GDPR concepts like "automated processing" and "data portability". Stopword filtering and word-boundary checks help slightly, but the model's training distribution causes the mismatch.

**Mitigations available**:
- Use longer, more specific queries
- Use `--category` to narrow to relevant sections
- Adjust `--hybrid-weight` (lower = more semantic, higher = more keyword)

---

### 6. Version pinning
`requirements.txt` has no upper bounds. A minor release could break the API.

---

## Low

### 7. Frontend `filter(Boolean)` hides falsy metadata
`web_frontend/index.html:111` filters with `.filter(Boolean)`. A legitimately falsy value like `verse: 0` would be hidden. Consider `.filter(v => v != null && v !== "")`.

---

### 8. `utils/config.py` caches at import time, no reload
`_config` is populated on first import. Any later changes to `config/config.cfg` are ignored in the same process.

---

### 9. Duplicate URLs (http vs https)
Results often show the same document twice under `http://` and `https://` variants. Deduplication should canonicalize URLs.

---

## Fixed since last review

- `web_frontend/fastapi_app.py`: Removed unused `use_db` parameter from `SearchRequest`
- Removed `--use-db` from README.md documentation (PostgreSQL is used automatically when configured)
- `app.py` reduced from 366→102 lines. Processing logic moved to `processor/processor.py`.
- `embedding/embedding.py` now raises proper `MemoryError`, `ValueError`, `RuntimeError` instead of `typer.Abort`
- `chunker/document.py` raises `MemoryError` (not `typer.Abort`)
- `processor/processor.py` created with all processing functions extracted from `app.py`
- `recommend_documents()` accepts `input_file` parameter
- `hybrid_weight` clamping added
- Debug logging uses `utils/logging.py` with category filtering from `config/config.cfg`
- Cancel button added to frontend with AbortController support
- Frontend fetches `/api/config` for `hybrid_weight` default
- `scrape` command auto-builds chunks and embeddings after crawling
- Vector store abstraction (`vector_store/`) replaces ad-hoc memmap logic
- `chunker/keywords.py` — TF keyword extractor for content-derived categories
- `Chunker._build_categories()` merges structural metadata + content keywords
- `Chunker.from_dict()` backward-compatible: derives `categories` for old chunks
- Category filtering (`--category`/`-c`) in CLI and `recommend_documents()`
- Stopword filtering in keyword scoring
- Word-boundary keyword overlap filter in `_gather_candidate_chunks()`
- Frontend displays `location` (book/chapter/verse) and `path` from API response
- `.kilo/` added to `.gitignore`
- Multiprocessing semaphore leak warning suppressed via `utils/logging.py`
