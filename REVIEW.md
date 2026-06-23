# Code Review

## Review Log

| Date | Reviewer | File | Summary | Status |
|------|----------|------|---------|--------|
| 2026-06-16 | — | — | Initial creation | Pending |
| 2026-06-16 | Kilo | embedding/embedding.py, datacollector/pdf_scanner.py | Replaced print/logging with utils.logging module | Approved |
| 2026-06-23 | Kilo | processor/processor.py, generator/generator.py, utils/config.py, web_frontend/fastapi_app.py, web_frontend/static/index.html | Fix sort crash, strip noisy tags, sort chunks by doc order, clean document modal display, fix Vietnamese garbling | Approved |

## Review Findings

### embedding/embedding.py
- **Status**: Approved
- **Notes**:
  - All `print()` calls correctly replaced with `debug()`/`error()` from `utils.logging`
  - Unused imports (`json`, `os`) removed
  - Consistent `category="embedding"` applied to all logging calls

### datacollector/pdf_scanner.py
- **Status**: Approved
- **Notes**:
  - Standard `logging` module replaced with `utils.logging`
  - All `logger.info()` and `logger.warning()` calls migrated
  - **Minor**: Import order slightly inconsistent (utils.logging before stdlib `os`, `re`, `shutil`) — should follow PEP 8 (stdlib → third-party → local)
  - **Minor**: `info("building metadata")` at line 186 will fire for every page; may be overly verbose in production

### processor/processor.py + generator/generator.py + utils/config.py
- **Status**: Approved
- **Notes**:
  - `_get_ordering_key` fallback now returns `(float("inf"), chunk_id)` instead of `(chunk_id,)`, fixing sort crash when metadata types are mixed (int vs str)
  - `_clean_response` and `_clean_stream` now strip unclosed `<thinking>` and `<reasoning>` tags (matching existing `<environment_details>` behavior)
  - Default `max_new_tokens` raised from 512 to 2048 in both code and `config.cfg`
  - New `clean_content()` helper strips noisy tags and optional leading location reference from chunk text
  - `best_chunk` in search results is now cleaned via `clean_content()`

### web_frontend/fastapi_app.py
- **Status**: Approved
- **Notes**:
  - `/api/document/{document_id}` now sorts chunks by `_get_ordering_key` (verse/page/chapter order) instead of random DB id
  - Added `_make_location` and `_extract_short_ref` helpers; chunks now carry `location` and `short_ref` fields

### web_frontend/static/index.html
- **Status**: Approved
- **Notes**:
  - `showDocument` now renders chunks individually with short refs (e.g., `46:3`, `Page 2`) shown only when the ref changes between consecutive chunks
  - Removed redundant `readablePath` breadcrumb when `location` metadata is present in search results
  - Fixed Vietnamese/Unicode garbling by expanding excerpt character whitelist from `\xa0-\xff` (Latin-1) to `\u00a0-\uffff` (full BMP)

## Notes

- Use this file to track review findings and follow-up actions.
- Add rows to the table above for each reviewed change.
