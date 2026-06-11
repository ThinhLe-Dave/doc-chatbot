import os
from typing import Annotated, List, Optional

import typer

from processor.processor import (
    recommend_documents,
    display_results,
)
from chunker.chunker import Chunker
from chunker.document import Document
from embedding.embedding import get_embedding_model
from vector_store.db_config import DatabaseConfig, SearchConfig
from vector_store.db_store import PostgresVectorStore
from datacollector.crawler import Scraper
from datacollector.pdf_scanner import PDFScanner
from utils.db_utils import insert_document, store_chunk_batch, SQL_DROP_TABLES


_search_config = SearchConfig.from_config_file()


def _build_doc_path(doc: Document) -> List[str]:
    path = []
    headers = doc.metadata.get("headers")
    if isinstance(headers, list):
        path.extend([h for h in headers if isinstance(h, str)])
    for key in ("book", "chapter", "verse", "section"):
        value = doc.metadata.get(key)
        if value and str(value) not in path:
            path.append(str(value))
    if doc.title and doc.title not in path:
        path.append(doc.title)
    return path


app = typer.Typer(rich_markup_mode="markdown", no_args_is_help=False)


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    query: Annotated[Optional[str], typer.Option("--query", "-q", help="A prompt to find documents to read")] = None,
    top_k: Annotated[int, typer.Option("--top-k", "-k", help="Number of documents to recommend")] = _search_config.top_k,
    chunk_k: Annotated[int, typer.Option("--chunk-k", help="Max matched chunks per document")] = _search_config.chunk_k,
    min_score: Annotated[float, typer.Option("--min-score", help="Minimum combined score to include")] = _search_config.min_score,
    hybrid: Annotated[bool, typer.Option("--hybrid/--no-hybrid", help="Use hybrid semantic+keyword scoring")] = _search_config.hybrid,
    hybrid_weight: Annotated[float, typer.Option("--hybrid-weight", help="Keyword weight in hybrid score (0-1)")] = _search_config.hybrid_weight,
    categories: Annotated[Optional[List[str]], typer.Option("--category", "-c", help="Filter chunks by category tag (repeatable)")] = _search_config.categories,
    json_output: Annotated[bool, typer.Option("--json", help="Output raw JSON instead of formatted text")] = False,
):
    if ctx.invoked_subcommand is None:
        prompt = query or typer.prompt("Enter a prompt describing the documents you want to read")
        results = recommend_documents(prompt, top_k, chunk_k, min_score, hybrid, hybrid_weight, categories)
        display_results(results, json_output)


@app.command()
def search(
    query: Annotated[Optional[str], typer.Argument(help="A prompt to find documents to read")] = None,
    top_k: Annotated[int, typer.Option("--top-k", "-k", help="Number of documents to recommend")] = _search_config.top_k,
    chunk_k: Annotated[int, typer.Option("--chunk-k", help="Max matched chunks per document")] = _search_config.chunk_k,
    min_score: Annotated[float, typer.Option("--min-score", help="Minimum combined score to include")] = _search_config.min_score,
    hybrid: Annotated[bool, typer.Option("--hybrid/--no-hybrid", help="Use hybrid semantic+keyword scoring")] = _search_config.hybrid,
    hybrid_weight: Annotated[float, typer.Option("--hybrid-weight", help="Keyword weight in hybrid score (0-1)")] = _search_config.hybrid_weight,
    categories: Annotated[Optional[List[str]], typer.Option("--category", "-c", help="Filter chunks by category tag (repeatable)")] = _search_config.categories,
    json_output: Annotated[bool, typer.Option("--json", help="Output raw JSON instead of formatted text")] = False,
):
    """Search scraped documents and return the most relevant pages."""
    prompt = query or typer.prompt("Enter a prompt describing the documents you want to read")
    results = recommend_documents(prompt, top_k, chunk_k, min_score, hybrid, hybrid_weight, categories)
    display_results(results, json_output)


@app.command()
def scrape(
    url: Optional[str] = typer.Argument(None, help="The starting URL to scrape"),
    limit: int = typer.Option(10000, "--limit", "-l", help="Limit the number of pages to scrape"),
):
    """
    **Document Chatbot Data Scraper**
    
    Crawls a website and stores content directly in PostgreSQL for chatbot processing.
    """
    url = url.strip() if url else None
    if not url:
        typer.secho("Error: No URL provided. Use --help for usage.", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)

    db_config = DatabaseConfig.from_config_file()
    if not db_config.is_configured():
        typer.secho("Database not configured. Add [database] section to config.cfg", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)

    print("🚀 Starting the web scraper...")
    print(f"Starting crawl on: {url}...")

    scraper = Scraper(base_url=url)
    scraper.crawl(max_pages=limit)

    model = get_embedding_model()
    store = PostgresVectorStore(config=db_config)
    store.load()
    conn = store._conn

    total_chunks = 0
    seen_docs = set()
    batch_chunks = []
    chunker = Chunker()

    for item in scraper.scraped_data:
        doc = Document.from_dict(item)
        doc_path = _build_doc_path(doc)
        if doc.id not in seen_docs:
            seen_docs.add(doc.id)
            with conn.cursor() as cur:
                insert_document(cur, doc.id, doc.source, doc.title, doc_path, doc.metadata)

        for chunk in chunker.create_chunks_from_document(doc):
            batch_chunks.append(chunk)
            if len(batch_chunks) >= 64:
                store_chunk_batch(conn, batch_chunks, model)
                total_chunks += len(batch_chunks)
                conn.commit()
                batch_chunks = []

    if batch_chunks:
        store_chunk_batch(conn, batch_chunks, model)
        total_chunks += len(batch_chunks)

    conn.commit()
    store.close()
    print(f"Scraping completed. {total_chunks} chunks saved to database.")


@app.command()
def pdf_scan(
    path: Annotated[Optional[str], typer.Argument(help="Path to PDF file or directory")] = None,
):
    """
    **PDF Scanner**
    
    Extracts text from PDF files and builds document chunks directly in PostgreSQL.
    """
    db_config = DatabaseConfig.from_config_file()
    if not db_config.is_configured():
        typer.secho("Database not configured. Add [database] section to config.cfg", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)

    path = path.strip() if path else None
    if not path:
        typer.secho("Error: No path provided. Use --help for usage.", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)

    if not os.path.exists(path) or not path.lower().endswith(".pdf"):
        typer.secho(f"Error: Invalid PDF path: {path}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)

    typer.echo(f"Scanning PDF: {path}...")

    scanner = PDFScanner()
    documents = scanner.scan_pdf(path)

    model = get_embedding_model()
    store = PostgresVectorStore(config=db_config)
    store.load()
    conn = store._conn

    total_chunks = 0
    seen_docs = set()
    batch_chunks = []

    for doc in documents:
        doc_path = _build_doc_path(doc)
        if doc.id not in seen_docs:
            seen_docs.add(doc.id)
            with conn.cursor() as cur:
                insert_document(cur, doc.id, doc.source, doc.title, doc_path, doc.metadata)

    chunker = Chunker()
    for doc in documents:
        for chunk in chunker.create_chunks_from_document(doc):
            batch_chunks.append(chunk)
            if len(batch_chunks) >= 64:
                store_chunk_batch(conn, batch_chunks, model)
                total_chunks += len(batch_chunks)
                conn.commit()
                batch_chunks = []

    if batch_chunks:
        store_chunk_batch(conn, batch_chunks, model)
        total_chunks += len(batch_chunks)

    conn.commit()
    store.close()
    print(f"Saved {total_chunks} chunks to database.")


@app.command()
def clear_db(
    force: Annotated[bool, typer.Option("--force", help="Skip confirmation prompt")] = False,
):
    """Drop all tables and recreate them (testing only)."""
    db_config = DatabaseConfig.from_config_file()
    if not db_config.is_configured():
        typer.secho("Database not configured. Add [database] section to config.cfg", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)

    if not force:
        typer.secho("This will permanently delete all documents, chunks, and embeddings.", fg=typer.colors.YELLOW, err=True)
        typer.secho("Run with --force to confirm.", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)

    store = PostgresVectorStore(config=db_config)
    try:
        store.load()
        with store._conn.cursor() as cur:
            cur.execute(SQL_DROP_TABLES)
        store._conn.commit()
        typer.secho("Database cleared successfully.", fg=typer.colors.GREEN)
    except Exception as e:
        typer.secho(f"Error clearing database: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)
    finally:
        store.close()


if __name__ == "__main__":
    app()