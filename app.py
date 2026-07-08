import os
import re
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
from utils.db_utils import (
    insert_document,
    store_chunk_batch,
    drop_tables,
    get_document_by_id,
    delete_document_chunks,
    list_documents,
    has_graph_chunks,
)
from utils.config import (
    get_search_top_k,
    get_search_chunk_k,
    get_search_min_score,
    get_search_hybrid,
    get_search_hybrid_weight,
    get_graph_enabled,
    get_graph_semantic_threshold,
    get_graph_community_resolution,
)
from utils.logging import debug


def _strip_verse_ref(text: str, ref: str) -> str:
    import re
    if not ref:
        return text
    pattern = rf"^{re.escape(ref)}\s*[-–—]?\s*"
    return re.sub(pattern, "", text, flags=re.IGNORECASE)


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


def build_chunk_cache(input_file: str, chunk_file: str, graph_mode: bool = False):
    from chunker.document import load_documents_from_json
    from chunker.chunker import write_chunks_to_file
    from embedding.embedding import get_embedding_model
    from vector_store.store import VectorStore

    effective_graph = graph_mode or get_graph_enabled()
    documents = load_documents_from_json(input_file)
    write_chunks_to_file(documents, chunk_file, graph_mode=effective_graph)
    store = VectorStore(chunk_file)
    model = get_embedding_model()
    return store.build(model)


def scrape_website(url: str, limit: int = 10000, sitemap_first: bool = False, force: bool = False, no_robots: bool = False, graph_mode: bool = False) -> dict:
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

    total_chunks = 0
    seen_docs = set()
    batch_chunks = []
    chunker = Chunker()

    for item in scraper.scraped_data:
        doc = Document.from_dict(item)
        doc_path = _build_doc_path(doc)
        if doc.id not in seen_docs:
            seen_docs.add(doc.id)
            conn = store._get_connection()
            with conn.cursor() as cur:
                insert_document(cur, doc.id, doc.source, doc.title, doc_path, doc.metadata)

        if graph_mode:
            chunks = chunker.create_graph_chunks(doc)[0]
        else:
            chunks = chunker.create_chunks_from_document(doc)
        for chunk in chunks:
            batch_chunks.append(chunk)
            if len(batch_chunks) >= 64:
                conn = store._get_connection()
                store_chunk_batch(conn, batch_chunks, model)
                total_chunks += len(batch_chunks)
                conn.commit()
                batch_chunks = []

    if batch_chunks:
        conn = store._get_connection()
        store_chunk_batch(conn, batch_chunks, model)
        total_chunks += len(batch_chunks)

    if conn:
        conn.commit()
    store.close()
    return {
        "total_chunks": total_chunks,
        "pages_scraped": len(scraper.scraped_data),
    }


def scan_pdf(path: str, chapters: Optional[List[str]] = None, graph_mode: bool = False) -> dict:
    debug("Starting PDF scan...", category="app")
    if not os.path.exists(path) or not path.lower().endswith(".pdf"):
        raise ValueError(f"Invalid PDF path: {path}")

    db_config = DatabaseConfig.from_config_file()
    scanner = PDFScanner()
    documents = scanner.scan_pdf(path, original_filename=path, chapters=chapters)

    model = get_embedding_model()
    store = PostgresVectorStore(config=db_config)
    store.load()

    total_chunks = 0
    seen_docs = set()
    batch_chunks = []
    chunker = Chunker()

    for doc in documents:
        doc_path = _build_doc_path(doc)
        if doc.id not in seen_docs:
            seen_docs.add(doc.id)
            conn = store._get_connection()
            with conn.cursor() as cur:
                insert_document(cur, doc.id, doc.source, doc.title, doc_path, doc.metadata)

        if graph_mode:
            chunks = chunker.create_graph_chunks(doc)[0]
        else:
            chunks = chunker.create_chunks_from_document(doc)
        for chunk in chunks:
            batch_chunks.append(chunk)
            if len(batch_chunks) >= 64:
                conn = store._get_connection()
                store_chunk_batch(conn, batch_chunks, model)
                total_chunks += len(batch_chunks)
                conn.commit()
                batch_chunks = []

    if batch_chunks:
        conn = store._get_connection()
        store_chunk_batch(conn, batch_chunks, model)
        total_chunks += len(batch_chunks)

    if conn:
        conn.commit()
    store.close()
    return {
        "total_chunks": total_chunks,
        "pages_processed": len(documents),
    }


def regraph_documents(
    doc_ids: Optional[List[str]] = None,
    source: Optional[str] = None,
    semantic_threshold: float = 0.75,
    resolution: float = 1.0,
    force: bool = False,
    limit: Optional[int] = None,
    verbose: bool = False,
) -> dict:
    """Re-chunk existing documents into graph chunks (in place).

    The database only stores chunk content, not the original document text,
    so each document is reconstructed by joining its chunk contents and then
    re-processed with ``create_graph_chunks``. Old chunks and embeddings are
    replaced with the new graph-based chunks.
    """
    from chunker.document import Document
    import time

    def vlog(msg: str) -> None:
        if verbose:
            typer.echo(msg)

    db_config = DatabaseConfig.from_config_file()
    if not db_config.is_configured():
        raise RuntimeError("Database not configured")

    chunker = Chunker()
    model = get_embedding_model()
    store = PostgresVectorStore(config=db_config)
    store.load()
    conn = store._get_connection()

    if doc_ids:
        docs = []
        for did in doc_ids:
            with conn.cursor() as cur:
                row = get_document_by_id(cur, did)
            if row:
                docs.append(row)
            else:
                typer.secho(f"Document not found: {did}", fg=typer.colors.YELLOW)
    else:
        with conn.cursor() as cur:
            docs = list_documents(cur, source=source, limit=limit)

    debug(
        "regraph: start docs=%d force=%s threshold=%.4f resolution=%.4f source=%s doc_ids=%s"
        % (len(docs), force, semantic_threshold, resolution, source, doc_ids),
        category="graph",
    )
    debug("regraph: selected %d documents to process" % len(docs), category="graph")
    vlog(f"regraph: selected {len(docs)} documents (force={force}, threshold={semantic_threshold}, resolution={resolution})")

    if not docs:
        store.close()
        return {"total_chunks": 0, "documents_processed": 0, "documents_skipped": 0}

    processed = 0
    skipped = 0
    total_chunks = 0
    pending = []  # list of (doc_id, chunks) awaiting insert
    flush_threshold = 2000
    start_time = time.time()
    progress_every = 50

    def flush() -> None:
        nonlocal pending
        if not pending:
            return
        all_chunks = [c for _, c in pending]
        doc_ids = [did for did, _ in pending]
        n = len(all_chunks)
        debug("regraph: embedding+inserting %d graph chunks (batched)" % n, category="graph")
        vlog(f"embedding + inserting {n} graph chunks (batched)...")
        # Insert new graph chunks first and commit, so a later failure can't
        # lose data (old chunks are only removed afterward).
        store_chunk_batch(conn, all_chunks, model)
        conn.commit()
        for did in doc_ids:
            with conn.cursor() as cur:
                delete_document_chunks(cur, did)
            conn.commit()
        pending = []

    for doc_row in docs:
        doc_id, doc_source, doc_title, doc_path, doc_metadata = doc_row
        vlog(f"processing: {doc_title or doc_id} ({doc_id})")
        with conn.cursor() as cur:
            if not force and has_graph_chunks(cur, doc_id):
                debug(f"regraph: skipping already-graph document {doc_id}", category="graph")
                skipped += 1
                vlog(f"  skip (already graph): {doc_title or doc_id}")
                continue
            cur.execute("SELECT content FROM chunks WHERE document_id=%s ORDER BY id", (doc_id,))
            contents = [row[0] for row in cur.fetchall() if row[0]]

        if not contents:
            debug(f"regraph: no chunks for document {doc_id}, skipping", category="graph")
            skipped += 1
            vlog(f"  skip (no chunks): {doc_title or doc_id}")
            continue

        content = "\n\n".join(contents)
        debug(
            "regraph: doc=%s title=%r source=%r source_chunks=%d recombined_chars=%d"
            % (doc_id, doc_title, doc_source, len(contents), len(content)),
            category="graph",
        )
        document = Document(
            id=doc_id,
            source=doc_source,
            title=doc_title,
            content=content,
            metadata=doc_metadata or {},
        )
        graph_result = chunker.create_graph_chunks(
            document,
            semantic_threshold=semantic_threshold,
            resolution=resolution,
        )
        chunks = graph_result[0]
        graph = graph_result[1]
        debug(
            "regraph: doc=%s graph_units=%d graph_edges=%d communities=%d -> %d graph chunks"
            % (
                doc_id,
                len(graph.units),
                len(graph.edges),
                len(chunks),
                len(chunks),
            ),
            category="graph",
        )

        # Queue new graph chunks; old chunks are removed only after a successful
        # batched insert+commit (see flush), so a failure never loses data.
        pending.append((doc_id, chunks))
        if sum(len(c) for _, c in pending) >= flush_threshold:
            flush()

        processed += 1
        total_chunks += len(chunks)
        debug(f"regraph: document {doc_id} -> {len(chunks)} graph chunks queued", category="graph")
        vlog(f"  [{processed}/{len(docs)}] {doc_title or doc_id}: {len(contents)} chunks -> {len(chunks)} graph chunks")

        if processed % progress_every == 0:
            elapsed = time.time() - start_time
            rate = processed / elapsed if elapsed else 0.0
            debug(
                "regraph: progress processed=%d skipped=%d queued_graph_chunks=%d elapsed=%.1fs docs/sec=%.2f"
                % (processed, skipped, total_chunks, elapsed, rate),
                category="graph",
            )
            vlog(
                f"progress: {processed}/{len(docs)} processed, {skipped} skipped, "
                f"{total_chunks} graph chunks queued, {elapsed:.1f}s elapsed, {rate:.2f} docs/s"
            )

    # Flush any remaining queued graph chunks in a final batched pass.
    flush()

    elapsed = time.time() - start_time
    debug(
        "regraph: done processed=%d skipped=%d total_graph_chunks=%d elapsed=%.1fs avg_graph_chunks_per_doc=%.2f"
        % (
            processed,
            skipped,
            total_chunks,
            elapsed,
            (total_chunks / processed) if processed else 0.0,
        ),
        category="graph",
    )
    vlog(
        f"regraph complete: {processed} processed, {skipped} skipped, "
        f"{total_chunks} graph chunks, {elapsed:.1f}s, "
        f"avg { (total_chunks / processed) if processed else 0.0:.2f} graph chunks/doc"
    )

    store.close()
    return {
        "total_chunks": total_chunks,
        "documents_processed": processed,
        "documents_skipped": skipped,
    }


@app.command()
def scrape(
    url: Optional[str] = typer.Argument(None, help="The starting URL to scrape"),
    limit: int = typer.Option(10000, "--limit", "-l", help="Limit the number of pages to scrape"),
    sitemap_first: bool = typer.Option(False, "--sitemap-first", help="Discover URLs from sitemap before link crawling"),
    force: bool = typer.Option(False, "--force", help="Force reprocess even if HTTP validators indicate unchanged content"),
    no_robots: bool = typer.Option(False, "--no-robots", help="Ignore robots.txt rules"),
    graph_mode: Optional[bool] = typer.Option(None, "--graph/--no-graph", help="Enable graph-aware chunking (falls back to [graph] enabled in config)"),
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

    effective_graph = bool(graph_mode) if graph_mode is not None else get_graph_enabled()
    result = scrape_website(url, limit, sitemap_first, force, no_robots, effective_graph)

    typer.echo(f"Discovery metrics - discovered: {result.get('pages_scraped', 0)}, fetched: {result.get('pages_scraped', 0)}, skipped: 0, failed: 0")
    typer.echo(f"Scraping completed. {result['total_chunks']} chunks saved to database.")


@app.command()
def pdf_scan(
    path: Annotated[Optional[str], typer.Argument(help="Path to PDF file")] = None,
    force: Annotated[bool, typer.Option("--force", help="Force reprocess even if unchanged pages are detected")] = False,
    chapters: Annotated[Optional[List[str]], typer.Option("--chapter", "-c", help="Filter by chapter/section (repeatable)")] = None,
    graph_mode: Optional[bool] = typer.Option(None, "--graph/--no-graph", help="Enable graph-aware chunking (falls back to [graph] enabled in config)"),
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

    effective_graph = bool(graph_mode) if graph_mode is not None else get_graph_enabled()
    result = scan_pdf(path, chapters, effective_graph)
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
        conn = store._get_connection()
        with conn.cursor() as cur:
            drop_tables(cur)
        conn.commit()
        typer.secho("Database cleared successfully.", fg=typer.colors.GREEN)
    except Exception as e:
        typer.secho(f"Error clearing database: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)
    finally:
        store.close()


@app.command(name="regraph")
def regraph(
    doc_ids: Annotated[Optional[List[str]], typer.Option("--doc-id", help="Re-chunk a specific document id (repeatable). Defaults to all documents.")] = None,
    source: Annotated[Optional[str], typer.Option("--source", help="Only re-chunk documents from this source URL.")] = None,
    threshold: Annotated[Optional[float], typer.Option("--threshold", help="Semantic edge threshold (default: [graph] semantic_threshold or 0.75)")] = None,
    resolution: Annotated[Optional[float], typer.Option("--resolution", help="Community detection resolution (default: [graph] community_resolution or 1.0)")] = None,
    force: Annotated[bool, typer.Option("--force", help="Re-process documents that already have graph chunks.")] = False,
    limit: Annotated[Optional[int], typer.Option("--limit", help="Max number of documents to process.")] = None,
    verbose: Annotated[bool, typer.Option("--verbose", help="Show per-document progress and live stats.")] = False,
):
    """Re-chunk documents already in the database into graph chunks.

    Useful for testing graph mode without re-scraping/re-scanning sources.
    Each document is reconstructed from its existing chunk contents, then
    re-processed with graph-aware chunking (new chunks + embeddings replace
    the old ones in place).
    """
    semantic_threshold = threshold if threshold is not None else get_graph_semantic_threshold()
    community_resolution = resolution if resolution is not None else get_graph_community_resolution()
    try:
        result = regraph_documents(
            doc_ids=doc_ids,
            source=source,
            semantic_threshold=semantic_threshold,
            resolution=community_resolution,
            force=force,
            limit=limit,
            verbose=verbose,
        )
    except Exception as e:
        typer.secho(f"regraph failed: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)
    typer.echo(
        f"regraph complete. documents_processed={result['documents_processed']} "
        f"documents_skipped={result['documents_skipped']} total_chunks={result['total_chunks']}"
    )


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
                    chunks = s.get("chunks") or []
                    if chunks and isinstance(chunks[0], dict):
                        text = chunks[0].get("text", "")
                    else:
                        text = s.get("best_chunk", "")
                    if text:
                        ref = f"{book} {chapter}"
                        if verse:
                            ref += f":{verse}"
                        text = _strip_verse_ref(text, ref)
                        preview = text[:100].replace('\n', ' ')
                        if len(text) > 100:
                            preview += "..."
                        typer.echo(f"    {preview}")
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
