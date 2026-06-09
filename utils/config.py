import os
import configparser
from pathlib import Path


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "config.cfg"

_config = configparser.ConfigParser()


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
