# REVIEW.md

## Summary

- Core RAG pipeline is functional and reasonably structured.
- Frontend/backend separation is clean: FastAPI serves API + static HTML, frontend is decoupled via configurable `apiBase`.
- Debug logging gated behind `DOC_DEBUG=1` env flag (fixed).
- Refactoring complete: `app.py` reduced from 366→108 lines. Processing logic moved to `processor/processor.py`.

---

## Critical / High

### 1. `scrape` CLI uses `typer.Abort` (CLI-only, safe)
`app.py:233` raises `typer.Abort` in the `scrape` command. This is CLI-only and **not** reachable from FastAPI, so it's acceptable.

### 2. CORS is wide-open
`web_frontend/fastapi_app.py:13-19` allows `*` origins. Currently `allow_credentials=False` which mitigates some risk. Now restricted to `GET` and `POST` methods only.

**Recommendation**: restrict to same-origin or a whitelist in production.

---

## Medium

### 3. Version pinning
`requirements.txt` has no upper bounds. A minor release could break the API.

---

## Low

### 4. Frontend `filter(Boolean)` hides falsy metadata
`web_frontend/index.html:111` filters with `.filter(Boolean)`. A legitimately falsy value like `verse: 0` would be hidden. Consider `.filter(v => v != null && v !== "")`.

---

## Fixed since last review

- `app.py` reduced from 366→108 lines. Processing logic moved to `processor/processor.py`.
- `embedding/embedding.py` now raises proper `MemoryError`, `ValueError`, `RuntimeError` instead of `typer.Abort`
- `chunker/document.py` already raises `MemoryError` (not `typer.Abort`)
- `processor/processor.py` created with all processing functions extracted from `app.py`
- `recommend_documents()` already accepts `input_file` parameter (not `input`)
- `hybrid_weight` clamping added in `recommend_documents`
- Debug logging gated behind `DOC_DEBUG=1` env flag
- Cancel button added to frontend with AbortController support
- Frontend now fetches `/api/config` for `hybrid_weight` default
- `scrape` command now auto-builds chunks and embeddings after crawling
