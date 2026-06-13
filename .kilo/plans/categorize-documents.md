# Category Enhancement Plan (Completed)

## Goal
Enable users to see and filter documents by subject/category when researching.

## Completed Changes
- [x] Display categories in CLI search results (processor/processor.py)
- [x] Include categories in document entry (chunker/document.py)
- [x] Add `/api/categories` endpoint for web UI (web_frontend/fastapi_app.py)
- [x] Add test coverage (test/test_processor.py)
- [x] Improve category extraction for PDFs - set `book` and `chapter` metadata
- [x] Keywords only used as categories when no structural metadata exists

## Current Behavior
Categories are now built from:
1. `book` metadata (e.g., "GDPR" for PDF documents)
2. `chapter` metadata (e.g., "page 57" for PDF pages)
3. `section`, `verse` metadata fields
4. Headers list (h1, h2, h3 from web pages)
5. Title and source as fallback
6. Extracted keywords (only when no structural metadata)

This ensures PDFs show "GDPR, page 57" instead of "the, collection, personal".