from . import hf_api, local, openai_api

PROVIDERS = {
    "hf_api": hf_api,
    "hf": hf_api,
    "huggingface": hf_api,
    "huggingface_api": hf_api,
    "openai": openai_api,
    "openai_api": openai_api,
    "local": local,
}


def get_provider(provider: str):
    try:
        return PROVIDERS[provider.lower().replace("-", "_")]
    except KeyError as exc:
        raise ValueError(f"Unsupported generator provider: {provider}") from exc


__all__ = ["hf_api", "local", "openai_api", "get_provider"]
