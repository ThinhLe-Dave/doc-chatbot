import os
import configparser
from pathlib import Path
from typing import Optional, List


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "config.cfg"

_config: configparser.ConfigParser = configparser.ConfigParser()


def get(section: str, key: str, default: str = "") -> str:
    try:
        return _config.get(section, key, fallback=default)
    except Exception:
        return default


def _load(path: str | Path | None = None) -> None:
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    _config.read(config_path)
    _hf_token = get("hf", "token", "")
    if _hf_token and not os.environ.get("HF_TOKEN"):
        os.environ["HF_TOKEN"] = _hf_token


_load()


def get_logging_categories() -> str:
    return get("logging", "categories", "")


def get_hf_token() -> str:
    return get("hf", "token", "") or os.environ.get("HF_TOKEN", "")


# Search configuration defaults
SEARCH_DEFAULTS = {
    "top_k": 10,
    "chunk_k": 3,
    "min_score": 0.01,
    "hybrid": True,
    "hybrid_weight": 0.1,
}


def get_search_top_k() -> int:
    try:
        return int(get("search", "top_k", str(SEARCH_DEFAULTS["top_k"])))
    except ValueError:
        return SEARCH_DEFAULTS["top_k"]


def get_search_chunk_k() -> int:
    try:
        return int(get("search", "chunk_k", str(SEARCH_DEFAULTS["chunk_k"])))
    except ValueError:
        return SEARCH_DEFAULTS["chunk_k"]


def get_search_min_score() -> float:
    try:
        return float(get("search", "min_score", str(SEARCH_DEFAULTS["min_score"])))
    except ValueError:
        return SEARCH_DEFAULTS["min_score"]


def get_search_hybrid() -> bool:
    val = get("search", "hybrid", "")
    if not val:
        return SEARCH_DEFAULTS["hybrid"]
    return val.lower() in ("true", "1", "yes")


def get_search_hybrid_weight() -> float:
    try:
        return float(get("search", "hybrid_weight", str(SEARCH_DEFAULTS["hybrid_weight"])))
    except ValueError:
        return SEARCH_DEFAULTS["hybrid_weight"]


# Database configuration defaults
DATABASE_DEFAULTS = {
    "host": "localhost",
    "port": 5432,
    "name": "docchatbot",
    "user": "docuser",
    "password": "",
}


def get_db_host() -> str:
    return get("database", "host") or os.environ.get("DB_HOST") or DATABASE_DEFAULTS["host"]


def get_db_port() -> int:
    try:
        return int(get("database", "port") or os.environ.get("DB_PORT") or str(DATABASE_DEFAULTS["port"]))
    except ValueError:
        return DATABASE_DEFAULTS["port"]


def get_db_name() -> str:
    return get("database", "name") or os.environ.get("DB_NAME") or DATABASE_DEFAULTS["name"]


def get_db_user() -> str:
    return get("database", "user") or os.environ.get("DB_USER") or DATABASE_DEFAULTS["user"]


def get_db_password() -> str:
    return get("database", "password") or os.environ.get("DB_PASSWORD") or DATABASE_DEFAULTS["password"]


def get_db_url() -> Optional[str]:
    return get("database", "url") or os.environ.get("DATABASE_URL")
