# Doc Chatbot — Architecture

## 1. Overview

A lightweight document search and chatbot-style retrieval system (RAG) that ingests documents from websites or PDFs, chunks them, embeds the chunks, stores them in a vector database, and retrieves the most relevant chunks to answer user questions.

**Tech Stack**
- **Language**: Python 3.11+
- **CLI Framework**: Typer (rich)
- **Web Framework**: FastAPI + Uvicorn
- **Embeddings**: Sentence Transformers (`paraphrase-multilingual-MiniLM-L12-v2`, 384-dim, normalized)
- **Vector Storage (Production)**: PostgreSQL + pgvector (ivfflat index, cosine ops)
- **Vector Storage (Legacy/Offline)**: NumPy memmap + JSONL sidecar files
- **PDF Processing**: PyMuPDF (fitz)
- **Web Scraping**: BeautifulSoup4 + requests + robots.txt + sitemap support
- **LLM Providers**: OpenAI-compatible API, HuggingFace Inference API, local Transformers
- **Config**: `configparser` (INI files), environment variable fallbacks

## 2. Directory Structure

| Directory | Role |
|---|---|
| `app.py` | **Main entry point**: Typer CLI + exported service functions (`scrape_website`, `scan_pdf`) shared with the web API. |
| `web_frontend/` | FastAPI backend + static HTML/JS browser UI. |
| `processor/` | **Business logic**: orchestrates retrieval, ranking, hybrid scoring, context expansion, and RAG orchestration. |
| `chunker/` | Document model (`Document`, `Chunk`), linear chunker, graph-aware chunker (`ChunkGraph`), chunk file I/O, keyword extraction. |
| `embedding/` | Model loading (SentenceTransformer singleton), batched text encoding with memory fallback. |
| `datacollector/` | Data ingestion: `Scraper` (web crawler) and `PDFScanner` (PDF text extraction). |
| `vector_store/` | Dual vector storage: file-based `VectorStore` (memmap + JSONL) and `PostgresVectorStore` (pgvector). |
| `generator/` | RAG answer generation with provider abstraction. |
| `config/` | INI configuration files (`config.cfg`, `config.local.cfg` overrides). |
| `utils/` | Shared utilities: config loaders, PostgreSQL helper SQL/CRUD. |
| `docs/` | Design documents and architecture notes. |
| `test/` | Unit tests. |

## 3. Entry Points

### 3.1 CLI (`app.py` → Typer)
Run via `python app.py` or `./run.sh`.

- **`search`** / **default**: Query documents (`recommend_documents` → display results).
- **`scrape`**: Crawl a website, chunk, embed, and store directly in PostgreSQL or file backend.
- **`pdf-scan`**: Scan a PDF, chunk, embed, and store.
- **`chat`** / **`ask`**: RAG question answering (`ask_question` → retrieve + generate).
- **`clear-db`**: Drop all tables (testing utility).

### 3.2 Web Frontend (`web_frontend/fastapi_app.py` → FastAPI)
Run via `./run.sh serve` (default port 8000, or Docker port 7860).

**Endpoints**
- `GET /` → Serves `web_frontend/static/index.html`
- `POST /api/search` → Paginated semantic search
- `POST /api/scrape` / `POST /api/pdf-scan` / `POST /api/pdf-scan-upload` → Async ingestion jobs
- `GET /api/jobs` / `GET /api/jobs/{job_id}` → Job status polling
- `POST /api/chat` → Async RAG chat job
- `GET /api/chat-stream/{job_id}` → SSE streaming for chat responses
- `POST /api/chat-direct` → Synchronous RAG chat (no job queue)
- `GET /api/document/{document_id}` → Retrieve full document with sorted chunks

**Key Design**: The FastAPI app imports `app.py` functions directly, eliminating code duplication between CLI and web.

### 3.3 Docker
`Dockerfile` builds on `python:3.11-slim`, installs dependencies, exposes port **7860**, and runs `uvicorn web_frontend.fastapi_app:app`.

## 4. Data Flow

### 4.1 Ingestion (Scrape / PDF Scan)
```
Raw Source (URL / PDF)
       ↓
DataCollector subclass
  ├─ Scraper.crawl() → BeautifulSoup extraction → list[dict]
  └─ PDFScanner.scan_pdf() → PyMuPDF text extraction → list[Document]
       ↓
Document.from_dict() → canonicalizes URLs, adds stable IDs, enriches metadata
       ↓
PostgresVectorStore or VectorStore
  - insert_document (if new)
  - Chunker.create_chunks_from_document() → List[Chunk]
  - embed_texts() in batches
  - store_chunk_batch() → chunks + embeddings
```

### 4.2 Chunking
**Linear mode**: Paragraph-first split, then sentence-boundary split, with configurable `chunk_size` and `chunk_overlap`. Each chunk gets:
- `id = {document_id}_chunk_{index}`
- `path` = hierarchical location path
- `categories` derived from metadata
- `source_hash` for deduplication

**Graph mode**: Available via `Chunker.create_graph_chunks()` but is experimental and not wired into the standard retrieval pipeline. Graph-aware chunking produces `ChunkGraph` objects with structural, hierarchical, and semantic edges, but these edges are not currently persisted or used as a retrieval signal.

### 4.3 Embedding
- Model: `paraphrase-multilingual-MiniLM-L12-v2` (384-dim)
- Singleton loading with device auto-selection
- Batched encoding with adaptive memory fallback (halve batch size on MemoryError)

### 4.4 Storage
**File backend** (`VectorStore`):
- Chunks: JSONL
- Embeddings: NumPy memmap
- IDs + metadata: JSON sidecar files

**PostgreSQL backend** (`PostgresVectorStore`):
- `documents(id TEXT PK, source TEXT, title TEXT, path JSONB, metadata JSONB)`
- `chunks(id TEXT PK, document_id TEXT FK, content TEXT, path JSONB, metadata JSONB)`
- `embeddings(chunk_id TEXT PK FK, embedding VECTOR(384))`
- `jobs` table for async job tracking
- Indexes on embeddings (ivfflat, cosine ops), document_id, and metadata fields

**Note**: There is no `chunk_edges` or graph-edge persistence table. Graph relationships are computed at chunk creation time but not stored or queried during retrieval.

## 5. Retrieval / Search

### 5.1 Current Implementation
```
Query string
    ↓
encode_query() → normalized embedding vector
    ↓
VectorStore.search() or PostgresVectorStore.search()
  - Top-k vector search (semantic cosine similarity)
    ↓
_expand_candidate_chunks()
  - For each matched chunk, add ±3 neighboring chunks within same document
  - Ordered by page/chunk_index/verse
    ↓
_rank_results()
  - Optional hybrid scoring: score = (1 - w) * semantic + w * keyword
  - Category filtering
  - Group by document/chapter
  - Deduplicate chunks, keep top chunk_k per document
  - Sort by score, filter min_score, slice top_k
    ↓
List[dict] with score, best_chunk, chunks[], location, metadata
```

**Important**: Context expansion is linear (neighboring chunks within the same document), not graph-traversal. There is no graph-edge-based retrieval path at this time.

### 5.2 Hybrid Search
- **Semantic score**: cosine similarity via pgvector or exact matrix multiply.
- **Keyword score**: ratio of matched query terms in chunk content + title using word-boundary regex.
- Combined with configurable `hybrid_weight` (default 0.1).

### 5.3 Category Filtering
- Categories derived from metadata (`book`, `chapter`, `verse`, `section`, `headers`).
- PostgreSQL filters via `jsonb_array_elements_text()`. File backend filters in Python.

## 6. Chat / Generation (RAG)

1. **Retrieve**: `recommend_documents()` returns ranked chunks.
2. **Context Formatting**: `format_context()` builds a numbered context block with location headers.
3. **Prompting**: System prompt + user prompt (context + question).
4. **Generation**: Provider-specific `generate()` or `generate_stream()`.
5. **Response Cleaning**: Strips `<thinking>`, `<reasoning>`, `<environment_details>` tags.
6. **Source Citation**: Parses `[Book Chapter:Verse]` patterns from the answer.

## 7. Provider Abstraction

**Registry pattern** in `generator/providers/__init__.py`:
```python
PROVIDERS = {
    "hf_api": hf_api, "hf": hf_api, "huggingface": hf_api,
    "openai": openai_api, "openai_api": openai_api,
    "local": local,
}
```

Each provider implements `generate()` and `generate_stream()`. Configurable via `[generator]` section.

## 8. Configuration System

**Files**:
1. `config/config.cfg` — committed defaults.
2. `config/config.local.cfg` — gitignored local overrides.
3. `config/config.cfg.example` — template.

**Loading** (`utils/config.py`): `configparser.ConfigParser` with env-var fallbacks.

**Sections**: `[search]`, `[database]`, `[generator]`, `[features]`, `[logging]`, `[graph]`, `[cors]`.

## 9. Design Patterns

### 9.1 Duck-Typed Dual Store
`VectorStore` (file) and `PostgresVectorStore` (DB) share no common base class. The processor selects between them via string flags (`"file"` / `"postgres"`) and branches on `store_type`. This works but is not a formal abstraction.

### 9.2 Lazy Singleton
- Embedding model: module-level cache.
- Generator clients: per-provider module globals.
- Chunk store: `_CACHED_CHUNK_STORE` in `processor.py`.

### 9.3 In-Memory Job Queue
FastAPI endpoints create `asyncio.create_task()` background jobs tracked in `_jobs` dict + `_running_tasks` set. This is suitable for demos/single-process deployments. The `jobs` database table exists in schema but is not used by the web layer for job persistence.

### 9.4 Adaptive Memory Management
- `embed_texts()` retries with halved `batch_size` on MemoryError.
- File backend uses `np.memmap` to avoid loading all embeddings into RAM.

### 9.5 Content Hash Deduplication
`compute_content_hash()` uses SHA-256 on whitespace-normalized text for `source_hash` and `document_hash`.

## 10. Known Gaps / Future Work

1. **Graph retrieval**: Graph-aware chunking exists but graph-based retrieval (traversing semantic/structural/hierarchical edges at query time) is not implemented. Context expansion remains linear.
2. **Graph persistence**: No `chunk_edges` or graph tables in the database schema. Graph relationships are transient.
3. **Production job durability**: In-memory job state is lost on restart. The `jobs` table schema is defined but unused by the web layer.
4. **Formal dual-store interface**: The two vector stores are accessed via backend-specific branches and duck typing. A shared abstract base class would improve maintainability.

## 11. Notable Files for Reference

| Concern | Primary File(s) |
|---|---|
| CLI entry point | `app.py` |
| Web API entry point | `web_frontend/fastapi_app.py` |
| Search orchestration | `processor/processor.py` |
| Database schema & SQL | `utils/db_utils.py` |
| Chunker (linear + graph) | `chunker/chunker.py`, `chunker/graph.py` |
| PostgreSQL vector store | `vector_store/db_store.py`, `vector_store/db_config.py` |
| File-based vector store | `vector_store/store.py`, `vector_store/index.py` |
| Generator & providers | `generator/generator.py`, `generator/providers/` |
| Prompt engineering | `generator/prompts.py` |
