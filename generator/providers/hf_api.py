from __future__ import annotations

from typing import Iterator, Optional

from generator.prompts import build_messages
from utils.config import (
    get_generator_max_new_tokens,
    get_generator_model_name,
    get_generator_temperature,
    get_generator_top_p,
    get_hf_token,
)
from utils.logging import debug, error


_client = None


def get_client():
    global _client
    if _client is not None:
        return _client

    from huggingface_hub import InferenceClient

    token = get_hf_token()
    model = get_generator_model_name()
    if not token:
        raise ValueError("HF token is not configured. Set [hf] token in config.cfg or HF_TOKEN.")

    debug(f"initializing InferenceClient model={model}", "generator.hf_api")
    _client = InferenceClient(token=token, model=model)
    return _client


def generate(
    query: str,
    context: str,
    max_new_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
) -> str:
    client = get_client()
    max_new_tokens = max_new_tokens if max_new_tokens is not None else get_generator_max_new_tokens()
    temperature = temperature if temperature is not None else get_generator_temperature()
    top_p = top_p if top_p is not None else get_generator_top_p()

    try:
        response = client.chat.completions.create(
            messages=build_messages(query, context),
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
) -> Iterator[str]:
    client = get_client()
    max_new_tokens = max_new_tokens if max_new_tokens is not None else get_generator_max_new_tokens()
    temperature = temperature if temperature is not None else get_generator_temperature()
    top_p = top_p if top_p is not None else get_generator_top_p()

    try:
        response = client.chat.completions.create(
            messages=build_messages(query, context),
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
