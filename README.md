# Doc Chatbot

A lightweight document search and chatbot-style retrieval system built from scraped JSON content. This repository supports:

- scraping website content into JSON using a custom crawler
- chunking documents for semantic search
- building embedding caches with Sentence Transformers
- querying documents using cosine similarity ranking
- an optional FastAPI-based web UI for browser search
- an optional Jupyter notebook interface

## Repository Structure

- `app.py` — main CLI entry point with commands for search, chunk creation, and scraping
- `web_frontend/` — FastAPI backend + browser UI + Jupyter notebook
- `chunker/` — document and chunk utilities
- `embedding/` — embedding model loading, encoding, and persistence helpers
- `datacollector/` — web scraping crawler and PDF scanner
- `utils/` — utility functions including data utilities
- `database/` — example scraped data and cache files
- `requirements.txt` — required Python dependencies

## Setup

```bash
cd /Users/thinhphu/Desktop/python/doc-chatbot
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Search documents from the command line

```bash
python app.py search "your query here"
```

You can specify the scraped JSON file and ranking options:

```bash
python app.py search "how does X work" --input database/research_data.json --top-k 5 --chunk-k 3 --min-score 0.1
```

### Build document chunks and embeddings

```bash
python app.py build-chunks --input database/research_data.json
```

This command:

1. loads scraped JSON documents
2. creates text chunks
3. builds an embedding cache using `sentence-transformers`
4. saves chunk IDs and embedding matrices alongside the chunk file

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

- The current embedding model is `paraphrase-multilingual-MiniLM-L12-v2` from Sentence Transformers.
- The search flow uses chunked document embeddings and a cosine similarity ranking over top results.
- If you hit memory issues during embedding generation, reduce the number of chunks or run on a machine with more RAM.

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
