from __future__ import annotations

from typing import Dict, Iterator, List, Optional

from generator.prompts import build_messages
from utils.config import (
    get_generator_api_key,
    get_generator_base_url,
    get_generator_max_new_tokens,
    get_generator_model_name,
    get_generator_temperature,
    get_generator_top_p,
)
from utils.logging import debug, error


_client = None


def get_client():
    global _client
    if _client is not None:
        return _client

    from huggingface_hub import InferenceClient

    model = get_generator_model_name()

    client_kwargs: dict = {"model": model}
    api_key = get_generator_api_key()
    if api_key:
        client_kwargs["token"] = api_key
    base_url = get_generator_base_url()
    if base_url:
        client_kwargs["base_url"] = base_url
        client_kwargs.pop("model", None)

    debug(f"initializing InferenceClient model={model} base_url={base_url or '(default)'}", "generator.hf_api")
    _client = InferenceClient(**client_kwargs)
    return _client


def generate(
    query: str,
    context: str,
    max_new_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    history: Optional[List[Dict[str, str]]] = None,
) -> str:
    client = get_client()
    max_new_tokens = max_new_tokens if max_new_tokens is not None else get_generator_max_new_tokens()
    temperature = temperature if temperature is not None else get_generator_temperature()
    top_p = top_p if top_p is not None else get_generator_top_p()

    try:
        response = client.chat.completions.create(
            messages=build_messages(query, context, history),
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=False,
        )
        return response.choices[0].message.content or ""
    except Exception as exc:
        error(f"HF API generation failed: {exc}", "generator.hf_api")
        raise


def generate_stream(
    query: str,
    context: str,
    max_new_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    history: Optional[List[Dict[str, str]]] = None,
) -> Iterator[str]:
    client = get_client()
    max_new_tokens = max_new_tokens if max_new_tokens is not None else get_generator_max_new_tokens()
    temperature = temperature if temperature is not None else get_generator_temperature()
    top_p = top_p if top_p is not None else get_generator_top_p()

    try:
        response = client.chat.completions.create(
            messages=build_messages(query, context, history),
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=True,
        )
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    except Exception as exc:
        error(f"HF API streaming failed: {exc}", "generator.hf_api")
        raise
