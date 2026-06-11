# Doc Chatbot

A lightweight document search and chatbot-style retrieval system built from scraped JSON content. This repository supports:

- scraping website content into JSON using a custom crawler
- chunking documents for semantic search
- building embedding caches with Sentence Transformers
- querying documents using cosine similarity ranking
- **PostgreSQL storage with pgvector for production deployments**
- an optional FastAPI-based web UI for browser search
- an optional Jupyter notebook interface

## Repository Structure

- `app.py` — main CLI entry point with commands for search, chunk creation, and scraping
- `web_frontend/` — FastAPI backend + browser UI + Jupyter notebook
- `chunker/` — document and chunk utilities
- `embedding/` — embedding model loading, encoding, and persistence helpers
- `datacollector/` — web scraping crawler and PDF scanner
- `utils/` — utility functions including data utilities
- `vector_store/` — Vector store implementations (file-based and PostgreSQL)
- `database/` — example scraped data and cache files
- `requirements.txt` — required Python dependencies

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

PostgreSQL is used automatically when configured in `config/config.cfg`.

### Build document chunks and embeddings

Build chunks and embeddings (stored in PostgreSQL when configured):

```bash
python app.py build-chunks --input database/research_data.json
```

### Migrate existing chunks to PostgreSQL

```bash
python app.py migrate --input "database/*_chunks.json"
```

### Scrape a website

```bash
python app.py scrape https://example.com --output research_data.json --limit 100
```

The scraped output is saved under `database/`.

### Run the web UI (FastAPI)

```bash
./run.sh serve
```

Then open `http://127.0.0.1:8000`.

To run on a custom address:
```bash
./run.sh serve 8000 0.0.0.0
```

### Run the notebook interface

```bash
./run.sh notebook
```

Open the URL shown in the terminal and navigate to `web_frontend/notebook.ipynb`.

To run on a custom address:
```bash
./run.sh notebook 8888 0.0.0.0
```

## Notes

- The current embedding model is `paraphrase-multilingual-MiniLM-L12-v2` from Sentence Transformers (384 dimensions).
- The search flow uses chunked document embeddings and a cosine similarity ranking over top results.
- If you hit memory issues during embedding generation, reduce the number of chunks or run on a machine with more RAM.
- When using PostgreSQL, source documents remain in JSON files; only chunks and embeddings are migrated.

## Development

To run the existing low-memory regression test:

```bash
./run.sh test
```

To check all `.py` files compile cleanly:

```bash
./run.sh compile
```

### Scan a PDF file

```bash
python app.py pdf-scan /path/to/document.pdf --output pdf_data.json
```

This command extracts text from each PDF page, creates chunks, and builds embeddings saved under `database/`.