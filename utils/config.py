import os
import configparser
from pathlib import Path
from typing import Optional, List


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "config.cfg"
# Local config for user-specific settings (gitignored, takes precedence)
LOCAL_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "config.local.cfg"

_config: configparser.ConfigParser = configparser.ConfigParser()


def get(section: str, key: str, default: str = "") -> str:
    try:
        return _config.get(section, key, fallback=default)
    except Exception:
        return default


def _load(path: str | Path | None = None) -> None:
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    _config.read(config_path)
    # Load local config as override (gitignored)
    local_path = LOCAL_CONFIG_PATH
    if local_path.exists():
        _config.read(local_path)


_load()


def get_logging_categories() -> str:
    return get("logging", "categories", "")


def get_debug_enabled() -> bool:
    return _parse_bool(get("logging", "debug", "") or os.environ.get("DOC_CHATBOT_DEBUG", ""), False)


def _parse_bool(value: str, default: bool = False) -> bool:
    if not value:
        return default
    return value.strip().lower() in ("true", "1", "yes", "on")


def get_cors_allowed_origins() -> List[str]:
    raw = get("cors", "allowed_origins", "") or os.environ.get("CORS_ALLOW_ORIGINS", "")
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


def get_cors_allow_credentials() -> bool:
    return _parse_bool(
        get("cors", "allow_credentials", "") or os.environ.get("CORS_ALLOW_CREDENTIALS", ""),
        False,
    )


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


# Generator configuration defaults
GENERATOR_DEFAULTS = {
    "provider": "hf_api",
    # NOTE: Change this default model to switch models
    # The config.cfg file takes precedence, so edit config/config.cfg for your model choice
    "model_name": "Qwen/Qwen2.5-7B-Instruct",
    "api_key": "",
    "base_url": "",
    "max_new_tokens": 2048,
    "temperature": 0.7,
    "top_p": 0.95,
}


def get_generator_provider() -> str:
    return get("generator", "provider", GENERATOR_DEFAULTS["provider"]) or GENERATOR_DEFAULTS["provider"]


def get_generator_model_name() -> str:
    return get("generator", "model_name") or GENERATOR_DEFAULTS["model_name"]


def get_generator_api_key() -> str:
    raw = get("generator", "api_key", GENERATOR_DEFAULTS["api_key"])
    if raw.strip('"').strip("'"):
        return raw.strip('"').strip("'")
    return os.environ.get("HF_TOKEN", os.environ.get("OPENAI_API_KEY", ""))


def get_generator_base_url() -> str:
    raw = get("generator", "base_url", GENERATOR_DEFAULTS["base_url"])
    return raw.strip('"').strip("'") or os.environ.get("OPENAI_BASE_URL", "")


def get_generator_max_new_tokens() -> int:
    try:
        return int(get("generator", "max_new_tokens") or str(GENERATOR_DEFAULTS["max_new_tokens"]))
    except ValueError:
        return GENERATOR_DEFAULTS["max_new_tokens"]


def get_generator_temperature() -> float:
    try:
        return float(get("generator", "temperature") or str(GENERATOR_DEFAULTS["temperature"]))
    except ValueError:
        return GENERATOR_DEFAULTS["temperature"]


def get_generator_top_p() -> float:
    try:
        return float(get("generator", "top_p") or str(GENERATOR_DEFAULTS["top_p"]))
    except ValueError:
        return GENERATOR_DEFAULTS["top_p"]
