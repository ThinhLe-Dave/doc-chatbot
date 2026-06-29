---
title: Doc Chatbot
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---

# Doc Chatbot

A lightweight document search and chatbot-style retrieval system. This repository supports:

- scraping website content for semantic search
- scanning PDF files for document processing
- building embedding caches with Sentence Transformers
- querying documents using cosine similarity ranking
- **PostgreSQL storage with pgvector for production deployments**
- an optional FastAPI-based web UI for browser search

## Repository Structure

- `app.py` — main CLI entry point with commands for search, scraping, and PDF scanning. Also exports shared service functions (`scrape_website`, `scan_pdf`) used by the web API.
- `web_frontend/` — FastAPI backend + browser UI
- `chunker/` — document and chunk utilities
- `embedding/` — embedding model loading, encoding, and persistence helpers
- `datacollector/` — web scraping crawler and PDF scanner
- `utils/` — utility functions including data utilities
- `vector_store/` — Vector store implementations (file-based and PostgreSQL)
- `database/` — example scraped data and cache files
- `requirements.txt` — required Python dependencies

## Architecture

The CLI (`app.py`) and web API (`web_frontend/fastapi_app.py`) share the same business logic. `app.py` exports `scrape_website()` and `scan_pdf()` functions, which the FastAPI endpoints call directly. This eliminates code duplication and removes the need for a separate job manager.

## PostgreSQL Setup

To use PostgreSQL for vector storage (recommended for production):

1. Install PostgreSQL 13+ with pgvector extension:
   ```sql
   CREATE EXTENSION IF NOT EXISTS vector;
   ```

2. Configure connection in `config/config.cfg`:
   ```ini
   [database]
   host = localhost
   port = 5432
   name = docchatbot
   user = docuser
   # password = your_password_here
   ```

3. Or set environment variable:
   ```bash
   export DATABASE_URL="postgresql://user:password@localhost:5432/docchatbot"
   ```

## Usage

### Search documents from the command line

```bash
python app.py search "your query here"
```

#### Search options

| Option | Description | Default |
|--------|-------------|---------|
| `--top-k, -k` | Number of documents to return | 10 |
| `--chunk-k` | Max matched chunks per document | 3 |
| `--min-score` | Minimum combined score threshold | 0.01 |
| `--hybrid/--no-hybrid` | Use hybrid semantic+keyword scoring | true |
| `--hybrid-weight` | Keyword weight in hybrid mode (0-1) | 0.1 |
| `--category, -c` | Filter chunks by category tag (repeatable) | - |
| `--json` | Output raw JSON instead of formatted text | false |

### Scrape a website

```bash
python app.py scrape https://example.com --limit 100
```

Crawls a website and stores content directly in PostgreSQL for chatbot processing.

### Scan a PDF file

```bash
python app.py pdf-scan /path/to/document.pdf
```

Extracts text from PDF files and builds document chunks directly in PostgreSQL.

#### PDF scan options

| Option | Description | Default |
|--------|-------------|---------|
| `--ocr/--no-ocr` | Use OCR fallback for scanned PDFs | false |
| `--ocr-language` | Tesseract language code for OCR | eng |
| `--ocr-dpi` | Rendering DPI for OCR | 200 |
| `--chapter, -c` | Filter by chapter/section (repeatable) | - |
| `--page-range` | Page range filter (e.g., "1,3-5,8") | - |

When using the web UI, click "Preview Chapters" to detect chapters in your PDF, then select which ones to scan. You can also specify a page range for more precise control.

### Clear the database

```bash
python app.py clear-db --force
```

Drops all tables and recreates them (testing only).

### Run the web UI (FastAPI)

```bash
./run.sh serve
```

Then open `http://127.0.0.1:8000`. To run on a custom address:

```bash
./run.sh serve 8000 0.0.0.0
```

## Notes

- The current embedding model is `paraphrase-multilingual-MiniLM-L12-v2` from Sentence Transformers (384 dimensions).
- The search flow uses chunked document embeddings and a cosine similarity ranking over top results.
- If you hit memory issues during embedding generation, reduce the number of chunks or run on a machine with more RAM.
- To silence the HuggingFace Hub unauthenticated request warning after the model has been cached, set `HF_HUB_OFFLINE=1` when running the app. This requires an initial internet connection to download the embedding model first.

## Recent Changes

- **Document view**: chunks are now sorted by chapter/verse/page order instead of database insertion order.
- **Content cleaning**: HTML-style thinking blocks (`<thinking>`, `<reasoning>`, `<environment_details>`) are stripped from search results and document previews.
- **Hybrid scoring fixed**: sorting crash when metadata contains mixed numeric/string types (e.g. page numbers alongside text headers) has been resolved.
- **Default token limit**: raised from 512 to 2048 to reduce truncated answers on long documents.

## Development

To run the test suite:

```bash
./run.sh test
```

To check all `.py` files compile cleanly:

```bash
./run.sh compile
```