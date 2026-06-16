# REVIEW.md

Last reviewed: 2026-06-15

## Architecture Snapshot

- CLI and FastAPI share the same `recommend_documents()` search path in `processor/processor.py`.
- FastAPI app (`web_frontend/fastapi_app.py`) serves the UI and API; frontend is vanilla JS at `web_frontend/static/index.html`.
- Scrape/PDF endpoints run as `asyncio` background tasks for non-blocking operation.
- PDF text extraction is **text-only** (OCR removed). `_join_split_words` handles common pypdf split-word artifacts; `_fix_split_words` adds a dictionary-based fallback.
- Served static files: `/static/` for UI, `/pdfs/` for indexed documents.
- CORS is off by default and controlled by `config/config.cfg` `[cors]` section.

## Key Files

- `app.py` — shared service functions (`scrape_website`, `scan_pdf`) and Typer CLI.
- `processor/processor.py` — search, ranking, hybrid scoring, and result building.
- `chunker/chunker.py` — chunking logic, categories, and dedup helpers.
- `chunker/document.py` — document model, hashing, and entry builders.
- `datacollector/pdf_scanner.py` — PDF extraction and text cleaning.
- `datacollector/crawler.py` — website crawling, robots, sitemap, HTTP validators.
- `embedding/embedding.py` — model loading and embedding.
- `vector_store/db_store.py` — PostgreSQL/pgvector storage.
- `utils/db_utils.py` — low-level SQL helpers.
- `utils/config.py` — config caching (loaded once at import).

## Notable Design Points

- Document upserts are idempotent via content hashes.
- Hybrid scoring combines semantic vector search with keyword matching.
- Pagination is handled in the FastAPI search endpoint (`page`, `page_size`).
- PDF uploads keep original filenames; `/pdfs/` serves them directly.
- `HF_TOKEN` is injected into `os.environ` from `[hf] token` in `config.cfg`.

## Risks / Watch Items

- `utils/config.py` is cached at import; restart required for config changes.
- `pdf_scanner._fix_split_words` relies on dictionary words found in the same text; rare proper nouns may still have split artifacts.
- In-memory job store (`_jobs` dict) is lost on server restart.
- No upper-bound version pinning in `requirements.txt`.
