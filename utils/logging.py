from __future__ import annotations

import os
import re
import warnings
from typing import List


_ENABLED_CATEGORIES: List[str] = []
_PARSED = False

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module=r"multiprocessing\.resource_tracker",
)


def _parse_categories() -> None:
    global _ENABLED_CATEGORIES, _PARSED
    if _PARSED:
        return
    _PARSED = True
    raw = ""
    try:
        from utils.config import get_logging_categories

        raw = get_logging_categories()
    except Exception:
        pass
    if raw:
        _ENABLED_CATEGORIES = [part.strip() for part in raw.split(",") if part.strip()]


def is_enabled(category: str = "") -> bool:
    if not _PARSED:
        _parse_categories()
    if not _ENABLED_CATEGORIES:
        return False
    if not category:
        return True
    for entry in _ENABLED_CATEGORIES:
        if entry == "*":
            return True
        if category == entry:
            return True
        if entry.endswith(".*") and category.startswith(entry[:-1]):
            return True
        if category.startswith(entry + "."):
            return True
    return False


def debug(msg: str, category: str = "") -> None:
    if not is_enabled(category):
        return
    try:
        from utils.config import get_debug_enabled

        if not get_debug_enabled():
            return
    except Exception:
        return
    try:
        import typer

        prefix = f"[DEBUG][{category}]" if category else "[DEBUG]"
        typer.secho(f"{prefix} {msg}", fg=typer.colors.YELLOW, dim=True)
    except Exception:
        pass


def info(msg: str, category: str = "") -> None:
    if not is_enabled(category):
        return
    try:
        import typer

        prefix = f"[INFO][{category}]" if category else "[INFO]"
        typer.secho(f"{prefix} {msg}", fg=typer.colors.CYAN)
    except Exception:
        pass


def warning(msg: str, category: str = "") -> None:
    if not is_enabled(category):
        return
    try:
        import typer

        prefix = f"[WARN][{category}]" if category else "[WARN]"
        typer.secho(f"{prefix} {msg}", fg=typer.colors.MAGENTA)
    except Exception:
        pass


def error(msg: str, category: str = "") -> None:
    try:
        import typer

        prefix = f"[ERROR][{category}]" if category else "[ERROR]"
        typer.secho(f"{prefix} {msg}", fg=typer.colors.RED, bold=True, err=True)
    except Exception:
        pass


def text_sample(text: str, max_chars: int = 240) -> str:
    compact = " ".join((text or "").split())
    if len(compact) > max_chars:
        compact = compact[:max_chars] + "..."
    return compact


def split_stats(text: str) -> str:
    words = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ]+", text or "")
    short_words = [word for word in words if len(word) <= 3]
    short_ratio = len(short_words) / len(words) if words else 0.0
    sample_short = ",".join(short_words[:12])
    return f"words={len(words)} short_words={len(short_words)} short_ratio={short_ratio:.2f} sample_short={sample_short!r}"
