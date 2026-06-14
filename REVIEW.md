# REVIEW.md

Last reviewed: 2026-06-14

## Summary

- Core RAG pipeline is functional and centered on PostgreSQL/pgvector for production search.
- CLI and FastAPI share the same `recommend_documents()` search path.
- Frontend/backend separation is clean: FastAPI serves API endpoints plus static HTML, while the frontend uses same-origin API paths.
- FastAPI supports background scrape and PDF scan jobs through `web_frontend/job_manager.py`.
- PDF uploads are written to generated temporary `.pdf` paths instead of client-supplied filenames.
- CORS is disabled by default and can be enabled with an explicit origin allowlist.
- Debug logging remains centralized and category-based via `config/config.cfg` `[logging] categories`.
- `HF_TOKEN` is now injected into `os.environ` before embedding model load.
- Category filtering is exposed in the CLI and FastAPI request model, and is pushed down into the PostgreSQL candidate search.
- URL canonicalization is applied during crawling/ingestion, and the frontend deduplicates results by canonical URL key.
- Hybrid keyword scoring uses stopword-filtered word-boundary matching.

---

## Critical / High

None currently tracked.

---

## Medium

### 3. Version pinning

`requirements.txt:1-13` has no upper bounds. A minor dependency release could introduce behavior changes or break compatibility.

**Recommendation**: pin or constrain versions after compatibility testing.

---

### 4. Config is cached at import time

`utils/config.py:9-27` reads `config/config.cfg` once on import. Later config changes are ignored until the process restarts.

**Recommendation**: add an explicit reload mechanism or avoid import-time caching for config-heavy workflows.

---

### 5. `SearchConfig.categories` is not loaded from config

`vector_store/db_config.py:20-40` defines `categories`, but `from_config_file()` always returns `None`.

**Recommendation**: either wire category defaults from `config.cfg` or remove the unused field.

---

### 6. Job state is in-memory only

`web_frontend/job_manager.py:6-24` stores jobs in process memory. Server restarts lose job status and results.

**Recommendation**: acceptable for local use; use Redis or a database-backed job store for production.

---

## Low

### 9. Frontend still uses `filter(Boolean)` for location display

`web_frontend/index.html:356` can hide falsy location values such as `verse: 0`.

**Recommendation**: use `v != null && v !== ""` instead.

---

### 10. Some chunk reads remain per-row

`processor/processor.py:232-257` and `vector_store/db_store.py:185-190` loop over chunk IDs individually.

**Impact**: acceptable for small datasets, but batch queries would scale better.

---

## Fixed since last review

- `web_frontend/fastapi_app.py`: `SearchRequest` now includes `categories` and passes them to `recommend_documents()`.
- CORS is no longer wide-open: FastAPI only registers CORS when an explicit origin allowlist is configured.
- PDF uploads no longer use client-supplied filenames; uploads are saved as generated temporary `.pdf` files and cleaned up after scanning.
- Category filtering is pushed into PostgreSQL candidate search, so category-filtered searches no longer depend on post-retrieval filtering.
- Hybrid keyword scoring now filters stopwords and matches whole words with word boundaries.
- Frontend API calls now use same-origin paths instead of a hard-coded `127.0.0.1:8000` base URL.
- Crawled and ingested URLs are canonicalized, and the frontend deduplicates search results by canonical URL key.
- `HF_TOKEN` is injected into `os.environ` from `[hf] token`.
- `scrape` CLI now raises `typer.Exit(1)` instead of `typer.Abort`.
- `app.py` remains reduced from 366→102 lines; processing logic lives in `processor/processor.py`.
- `embedding/embedding.py` and `chunker/document.py` raise proper exceptions instead of Typer aborts.
- `processor/processor.py` contains the extracted search/ranking logic.
- `hybrid_weight` clamping is implemented.
- Debug logging uses `utils/logging.py` with category filtering from `config/config.cfg`.
- Frontend cancel button uses `AbortController`.
- Frontend fetches `/api/config` for configuration availability.
- `scrape` command auto-builds chunks and embeddings after crawling.
- Vector store abstraction (`vector_store/`) replaces ad-hoc memmap logic for production storage.
- `chunker/keywords.py` provides a lightweight TF keyword extractor.
- `Chunker._build_categories()` merges structural metadata, page metadata, and content keyword fallback.
- `Chunk.from_dict()` is backward-compatible and derives categories for older chunks.
- Category filtering (`--category`/`-c`) is wired into the CLI and `recommend_documents()`.
- Frontend displays `location` and `path` from API responses.
- `.kilo/` is added to `.gitignore`.
- Multiprocessing semaphore leak warning is suppressed via `utils/logging.py`.
