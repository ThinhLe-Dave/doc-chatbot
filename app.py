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
from vector_store.db_config import DatabaseConfig
from vector_store.db_store import PostgresVectorStore
from datacollector.crawler import Scraper
from datacollector.pdf_scanner import PDFScanner
from utils.db_utils import insert_document, store_chunk_batch, SQL_DROP_TABLES
from utils.config import (
    get_search_top_k,
    get_search_chunk_k,
    get_search_min_score,
    get_search_hybrid,
    get_search_hybrid_weight,
)
from utils.logging import debug


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
    top_k: Annotated[int, typer.Option("--top-k", "-k", help="Number of documents to recommend")] = None,
    chunk_k: Annotated[int, typer.Option("--chunk-k", help="Max matched chunks per document")] = None,
    min_score: Annotated[float, typer.Option("--min-score", help="Minimum combined score to include")] = None,
    hybrid: Annotated[bool, typer.Option("--hybrid/--no-hybrid", help="Use hybrid semantic+keyword scoring")] = None,
    hybrid_weight: Annotated[float, typer.Option("--hybrid-weight", help="Keyword weight in hybrid score (0-1)")] = None,
    categories: Annotated[Optional[List[str]], typer.Option("--category", "-c", help="Filter chunks by category tag (repeatable)")] = None,
    json_output: Annotated[bool, typer.Option("--json", help="Output raw JSON instead of formatted text")] = False,
):
    top_k = top_k if top_k is not None else get_search_top_k()
    chunk_k = chunk_k if chunk_k is not None else get_search_chunk_k()
    min_score = min_score if min_score is not None else get_search_min_score()
    hybrid = hybrid if hybrid is not None else get_search_hybrid()
    hybrid_weight = hybrid_weight if hybrid_weight is not None else get_search_hybrid_weight()

    if ctx.invoked_subcommand is None:
        prompt = query or typer.prompt("Enter a prompt describing the documents you want to read")
        results = recommend_documents(prompt, top_k, chunk_k, min_score, hybrid, hybrid_weight, categories)
        display_results(results, json_output)


@app.command()
def search(
    query: Annotated[Optional[str], typer.Argument(help="A prompt to find documents to read")] = None,
    top_k: Annotated[int, typer.Option("--top-k", "-k", help="Number of documents to recommend")] = None,
    chunk_k: Annotated[int, typer.Option("--chunk-k", help="Max matched chunks per document")] = None,
    min_score: Annotated[float, typer.Option("--min-score", help="Minimum combined score to include")] = None,
    hybrid: Annotated[bool, typer.Option("--hybrid/--no-hybrid", help="Use hybrid semantic+keyword scoring")] = None,
    hybrid_weight: Annotated[float, typer.Option("--hybrid-weight", help="Keyword weight in hybrid score (0-1)")] = None,
    categories: Annotated[Optional[List[str]], typer.Option("--category", "-c", help="Filter chunks by category tag (repeatable)")] = None,
    json_output: Annotated[bool, typer.Option("--json", help="Output raw JSON instead of formatted text")] = False,
):
    """Search scraped documents and return the most relevant pages."""
    top_k = top_k if top_k is not None else get_search_top_k()
    chunk_k = chunk_k if chunk_k is not None else get_search_chunk_k()
    min_score = min_score if min_score is not None else get_search_min_score()
    hybrid = hybrid if hybrid is not None else get_search_hybrid()
    hybrid_weight = hybrid_weight if hybrid_weight is not None else get_search_hybrid_weight()
    prompt = query or typer.prompt("Enter a prompt describing the documents you want to read")
    results = recommend_documents(prompt, top_k, chunk_k, min_score, hybrid, hybrid_weight, categories)
    display_results(results, json_output)


def build_chunk_cache(input_file: str, chunk_file: str):
    from chunker.document import load_documents_from_json
    from chunker.chunker import write_chunks_to_file
    from embedding.embedding import get_embedding_model
    from vector_store.store import VectorStore

    documents = load_documents_from_json(input_file)
    write_chunks_to_file(documents, chunk_file)
    store = VectorStore(chunk_file)
    model = get_embedding_model()
    return store.build(model)


def scrape_website(url: str, limit: int = 10000, sitemap_first: bool = False, force: bool = False, no_robots: bool = False) -> dict:
    url = url.strip()
    if not url:
        raise ValueError("No URL provided")

    db_config = DatabaseConfig.from_config_file()
    if not db_config.is_configured():
        raise RuntimeError("Database not configured")

    scraper = Scraper(base_url=url, sitemap_first=sitemap_first, obey_robots=not no_robots)
    scraper.crawl(max_pages=limit, force=force)

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
    return {
        "total_chunks": total_chunks,
        "pages_scraped": len(scraper.scraped_data),
    }


def scan_pdf(path: str, chapters: Optional[List[str]] = None) -> dict:
    debug("Starting PDF scan...", category="app")
    if not os.path.exists(path) or not path.lower().endswith(".pdf"):
        raise ValueError(f"Invalid PDF path: {path}")

    db_config = DatabaseConfig.from_config_file()
    scanner = PDFScanner()
    documents = scanner.scan_pdf(path, original_filename=path, chapters=chapters)

    model = get_embedding_model()
    store = PostgresVectorStore(config=db_config)
    store.load()
    conn = store._conn

    total_chunks = 0
    seen_docs = set()
    batch_chunks = []

    chunker = Chunker()
    for doc in documents:
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
    return {
        "total_chunks": total_chunks,
        "pages_processed": len(documents),
    }


@app.command()
def scrape(
    url: Optional[str] = typer.Argument(None, help="The starting URL to scrape"),
    limit: int = typer.Option(10000, "--limit", "-l", help="Limit the number of pages to scrape"),
    sitemap_first: bool = typer.Option(False, "--sitemap-first", help="Discover URLs from sitemap before link crawling"),
    force: bool = typer.Option(False, "--force", help="Force reprocess even if HTTP validators indicate unchanged content"),
    no_robots: bool = typer.Option(False, "--no-robots", help="Ignore robots.txt rules"),
):
    """
    **Document Chatbot Data Scraper**

    Crawls a website and stores content directly in PostgreSQL for chatbot processing.
    """
    url = url.strip() if url else None
    if not url:
        typer.secho("Error: No URL provided. Use --help for usage.", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)

    typer.echo(f"Starting crawl on: {url}...")

    result = scrape_website(url, limit, sitemap_first, force, no_robots)

    typer.echo(f"Discovery metrics - discovered: {result.get('pages_scraped', 0)}, fetched: {result.get('pages_scraped', 0)}, skipped: 0, failed: 0")
    typer.echo(f"Scraping completed. {result['total_chunks']} chunks saved to database.")


@app.command()
def pdf_scan(
    path: Annotated[Optional[str], typer.Argument(help="Path to PDF file")] = None,
    force: Annotated[bool, typer.Option("--force", help="Force reprocess even if unchanged pages are detected")] = False,
    chapters: Annotated[Optional[List[str]], typer.Option("--chapter", "-c", help="Filter by chapter/section (repeatable)")] = None,
):
    """
    **PDF Scanner**

    Extracts text from PDF files and builds document chunks directly in PostgreSQL.
    """
    debug("Starting PDF scan...", category="app")
    path = path.strip() if path else None
    if not path:
        typer.secho("Error: No path provided. Use --help for usage.", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)

    if not os.path.exists(path) or not path.lower().endswith(".pdf"):
        typer.secho(f"Error: Invalid PDF path: {path}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)

    result = scan_pdf(path, chapters)
    typer.echo(f"PDF scan completed. {result['total_chunks']} chunks saved to database.")


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


@app.command(name="chat")
def chat(
    query: Annotated[Optional[str], typer.Argument(help="Question to ask")] = None,
    top_k: Annotated[int, typer.Option("--top-k", "-k", help="Number of documents to retrieve")] = None,
    chunk_k: Annotated[int, typer.Option("--chunk-k", help="Max matched chunks per document")] = None,
    min_score: Annotated[float, typer.Option("--min-score", help="Minimum combined score to include")] = None,
    hybrid: Annotated[bool, typer.Option("--hybrid/--no-hybrid", help="Use hybrid semantic+keyword scoring")] = None,
    hybrid_weight: Annotated[float, typer.Option("--hybrid-weight", help="Keyword weight in hybrid score (0-1)")] = None,
    json_output: Annotated[bool, typer.Option("--json", help="Output raw JSON instead of formatted text")] = False,
):
    """Ask a question using RAG (retrieval-augmented generation)."""
    from processor.processor import ask_question
    import json
    
    if query is None:
        query = typer.prompt("Enter your question")
    
    try:
        result = ask_question(
            query=query,
            top_k=top_k if top_k is not None else get_search_top_k(),
            chunk_k=chunk_k if chunk_k is not None else get_search_chunk_k(),
            min_score=min_score if min_score is not None else get_search_min_score(),
            hybrid=hybrid if hybrid is not None else get_search_hybrid(),
            hybrid_weight=hybrid_weight if hybrid_weight is not None else get_search_hybrid_weight(),
)
        
        if json_output:
            typer.echo(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            typer.secho(f"Q: {query}", fg=typer.colors.CYAN, bold=True)
            typer.echo(f"A: {result.get('answer', 'No answer generated')}")
            sources = result.get("sources", [])
            if sources:
                typer.secho(f"\nSources ({len(sources)}):", fg=typer.colors.BLACK)
            for s in sources:
                book = s.get("book") or ""
                chapter = s.get("chapter") or ""
                verse = s.get("verse") or ""
                if book:
                    ref = book
                    if chapter:
                        ref += f" {chapter}"
                        if verse:
                            ref += f":{verse}"
                    typer.echo(f"  [{ref}]")
                else:
                    title = s.get("title") or "Untitled"
                    typer.echo(f"  {title}")
    except Exception as e:
        typer.secho(f"Error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)


@app.command(name="ask")
def ask(
    query: Annotated[Optional[str], typer.Argument(help="Question to ask")] = None,
    top_k: Annotated[int, typer.Option("--top-k", "-k", help="Number of documents to retrieve")] = None,
    json_output: Annotated[bool, typer.Option("--json", help="Output raw JSON instead of formatted text")] = False,
):
    """Ask a question (alias for chat)."""
    return chat(query=query, top_k=top_k, json_output=json_output)


if __name__ == "__main__":
    app()
