# Code Review

## Review Log

| Date | Reviewer | File | Summary | Status |
|------|----------|------|---------|--------|
| 2026-06-16 | — | — | Initial creation | Pending |
| 2026-06-16 | Kilo | embedding/embedding.py, datacollector/pdf_scanner.py | Replaced print/logging with utils.logging module | Approved |

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

## Notes

- Use this file to track review findings and follow-up actions.
- Add rows to the table above for each reviewed change.
