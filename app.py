import os
import glob
from typing import Annotated, List, Optional

import typer

from processor.processor import (
    recommend_documents,
    display_results,
    build_chunk_cache,
)
import utils.data_utils as data_utils
from vector_store.db_config import DatabaseConfig
from vector_store.db_store import PostgresVectorStore
from chunker.chunker import iter_chunks_from_json
from embedding.embedding import embed_texts, get_embedding_model

app = typer.Typer(rich_markup_mode="markdown")


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    query: Annotated[Optional[str], typer.Option("--query", "-q", help="A prompt to find documents to read")] = None,
    input: Annotated[str, typer.Option("--input", "-i", help="Path to scraped JSON file")] = "database/research_data.json",
    top_k: Annotated[int, typer.Option("--top-k", "-k", help="Number of documents to recommend")] = 10,
    chunk_k: Annotated[int, typer.Option("--chunk-k", help="Max matched chunks per document")] = 3,
    min_score: Annotated[float, typer.Option("--min-score", help="Minimum combined score to include")] = 0.01,
    hybrid: Annotated[bool, typer.Option("--hybrid/--no-hybrid", help="Use hybrid semantic+keyword scoring")] = True,
    hybrid_weight: Annotated[float, typer.Option("--hybrid-weight", help="Keyword weight in hybrid score (0-1)")] = 0.4,
    categories: Annotated[Optional[List[str]], typer.Option("--category", "-c", help="Filter chunks by category tag (repeatable)")] = None,
    json_output: Annotated[bool, typer.Option("--json", help="Output raw JSON instead of formatted text")] = False,
):
    if ctx.invoked_subcommand is None:
        prompt = query or typer.prompt("Enter a prompt describing the documents you want to read")
        results = recommend_documents(prompt, input, top_k, chunk_k, min_score, hybrid, hybrid_weight, categories)
        display_results(results, json_output)


@app.command()
def search(
    query: Annotated[Optional[str], typer.Argument(help="A prompt to find documents to read")] = None,
    input: Annotated[str, typer.Option("--input", "-i", help="Path to scraped JSON file")] = "database/research_data.json",
    top_k: Annotated[int, typer.Option("--top-k", "-k", help="Number of documents to recommend")] = 10,
    chunk_k: Annotated[int, typer.Option("--chunk-k", help="Max matched chunks per document")] = 3,
    min_score: Annotated[float, typer.Option("--min-score", help="Minimum combined score to include")] = 0.01,
    hybrid: Annotated[bool, typer.Option("--hybrid/--no-hybrid", help="Use hybrid semantic+keyword scoring")] = True,
    hybrid_weight: Annotated[float, typer.Option("--hybrid-weight", help="Keyword weight in hybrid score (0-1)")] = 0.4,
    categories: Annotated[Optional[List[str]], typer.Option("--category", "-c", help="Filter chunks by category tag (repeatable)")] = None,
    json_output: Annotated[bool, typer.Option("--json", help="Output raw JSON instead of formatted text")] = False,
):
    """Search scraped documents and return the most relevant pages."""
    prompt = query or typer.prompt("Enter a prompt describing the documents you want to read")
    results = recommend_documents(prompt, input, top_k, chunk_k, min_score, hybrid, hybrid_weight, categories)
    display_results(results, json_output)


@app.command()
def build_chunks(
    input: Annotated[str, typer.Option("--input", "-i", help="Path to scraped JSON file")] = "database/research_data.json",
    output: Annotated[Optional[str], typer.Option("--output", "-o", help="Output chunk JSON file")] = None,
):
    """Build and save document chunks from a scraped JSON file."""
    chunk_count, saved_path = build_chunk_cache(input, output)
    typer.secho(f"Built {chunk_count} chunk records in PostgreSQL database", fg=typer.colors.GREEN)


@app.command()
def scrape(
    url: Annotated[Optional[str], typer.Argument(help="The starting URL to scrape")] = None,
    output: Annotated[str, typer.Option("--output", "-o", help="Output JSON filename")] = "research_data.json",
    limit: Annotated[int, typer.Option("--limit", "-l", help="Limit the number of pages to scrape")] = 10000,
):
    """
    **Document Chatbot Data Scraper**
    
    Crawls a website and exports the content to a JSON file for chatbot processing.
    """
    if not url:
        url = typer.prompt("Enter the URL to scrape").strip()

    if not url:
        typer.secho("Error: No URL provided.", fg=typer.colors.RED, err=True)
        raise typer.Abort()

    output_file = output
    if not output_file.endswith(".json"):
        output_file += ".json"

    typer.echo(f"Starting crawl on: {url}...")
    chunk_count, saved_path = data_utils.scrape(url=url, output=output_file, limit=limit)
    typer.secho(f"Scraping completed. Chunks saved to {saved_path}", fg=typer.colors.GREEN, bold=True)


@app.command()
def pdf_scan(
    path: Annotated[Optional[str], typer.Argument(help="Path to PDF file or directory")] = None,
    output: Annotated[str, typer.Option("--output", "-o", help="Output JSON filename")] = "pdf_data.json",
):
    """
    **PDF Scanner**
    
    Extracts text from PDF files and builds document chunks.
    """
    if not path:
        path = typer.prompt("Enter the PDF file or directory path").strip()

    if not path:
        typer.secho("Error: No path provided.", fg=typer.colors.RED, err=True)
        raise typer.Abort()

    if not os.path.exists(path) or not path.lower().endswith(".pdf"):
        typer.secho(f"Error: Invalid PDF path: {path}", fg=typer.colors.RED, err=True)
        raise typer.Abort()

    output_file = output
    if not output_file.endswith(".json"):
        output_file = f"{output_file}.json"

    typer.echo(f"Scanning PDF: {path}...")
    chunk_count, saved_path = data_utils.pdf_scan(path=path, output=output_file)
    typer.secho(f"Saved {chunk_count} chunks to {saved_path}", fg=typer.colors.GREEN, bold=True)


@app.command()
def migrate(
    input: Annotated[str, typer.Option("--input", "-i", help="Path to chunk JSON file or glob pattern")] = "database/*_chunks.json",
):
    """Migrate existing chunk files to PostgreSQL database."""
    db_config = DatabaseConfig.from_config_file()
    if not db_config.is_configured():
        typer.secho("Database not configured. Add [database] section to config.cfg", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)

    chunk_files = glob.glob(input)
    if not chunk_files:
        typer.secho(f"No chunk files found matching: {input}", fg=typer.colors.YELLOW)
        return

    for chunk_file in chunk_files:
        typer.echo(f"Migrating: {chunk_file}")
        try:
            store = PostgresVectorStore(config=db_config, chunk_file=chunk_file)
            store.load()
            migrate_chunks_to_db(chunk_file, store)
            store.close()
        except Exception as e:
            typer.secho(f"Error migrating {chunk_file}: {e}", fg=typer.colors.RED, err=True)


def migrate_chunks_to_db(chunk_file: str, store: PostgresVectorStore):
    """Migrate chunks and embeddings from JSONL to PostgreSQL."""
    import json
    conn = store._conn
    model = get_embedding_model()
    seen_docs = set()

    chunks = list(iter_chunks_from_json(chunk_file))
    total = len(chunks)

    for i, chunk in enumerate(chunks, start=1):
        batch_texts = [chunk.content]
        batch_embeddings = embed_texts(model, batch_texts)

        embedding = batch_embeddings[0]
        with conn.cursor() as cur:
            if chunk.document_id not in seen_docs:
                cur.execute(
                    """
                    INSERT INTO documents (id, source, title, path, metadata)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (id) DO NOTHING
                    """,
                    (
                        chunk.document_id,
                        chunk.metadata.get("source", ""),
                        chunk.metadata.get("title", ""),
                        json.dumps(chunk.path),
                        json.dumps(chunk.metadata),
                    )
                )
                seen_docs.add(chunk.document_id)
            cur.execute(
                """
                INSERT INTO chunks (id, document_id, content, path, metadata)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    content = EXCLUDED.content,
                    path = EXCLUDED.path,
                    metadata = EXCLUDED.metadata
                """,
                (
                    chunk.id,
                    chunk.document_id,
                    chunk.content,
                    json.dumps(chunk.path),
                    json.dumps(chunk.metadata),
                )
            )
            cur.execute(
                """
                INSERT INTO embeddings (chunk_id, embedding)
                VALUES (%s, %s)
                ON CONFLICT (chunk_id) DO UPDATE SET
                    embedding = EXCLUDED.embedding
                """,
                (chunk.id, embedding.tolist())
            )

        if i % 100 == 0:
            conn.commit()
            if i % 1000 == 0 or i == total:
                typer.echo(f"  Processed {i}/{total} chunks...")

    conn.commit()
    typer.secho(f"Migrated {total} chunks to database", fg=typer.colors.GREEN)


if __name__ == "__main__":
    app()