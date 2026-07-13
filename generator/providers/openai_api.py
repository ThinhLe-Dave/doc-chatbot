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

    from openai import OpenAI

    api_key = get_generator_api_key()
    base_url = get_generator_base_url()
    if not api_key:
        raise ValueError("OpenAI API key is not set. Set [generator] api_key in config or OPENAI_API_KEY env var.")

    debug(f"initializing OpenAI client base_url={base_url or '(default)'}", "generator.openai")
    client_kwargs = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url
    _client = OpenAI(**client_kwargs)
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
    model = get_generator_model_name()

    try:
        response = client.chat.completions.create(
            model=model,
            messages=build_messages(query, context, history),
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=False,
        )
        return response.choices[0].message.content or ""
    except Exception as exc:
        error(f"OpenAI generation failed: {exc}", "generator.openai")
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
    model = get_generator_model_name()

    try:
        response = client.chat.completions.create(
            model=model,
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
        error(f"OpenAI streaming failed: {exc}", "generator.openai")
        raise
