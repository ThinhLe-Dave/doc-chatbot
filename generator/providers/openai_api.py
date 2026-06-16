from __future__ import annotations

import os
from typing import Iterator, Optional

from utils.config import (
    get_generator_model_name,
    get_generator_max_new_tokens,
    get_generator_temperature,
    get_generator_top_p,
)
from utils.logging import debug, error


_client = None


def get_client():
    """Get or create OpenAI client singleton."""
    global _client
    if _client is not None:
        return _client
    
    from openai import OpenAI
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set in environment")
    
    debug("initializing OpenAI client", "generator.openai")
    _client = OpenAI(api_key=api_key)
    return _client


def generate(
    query: str,
    context: str,
    max_new_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
) -> str:
    """Generate answer using OpenAI API (non-streaming)."""
    client = get_client()
    
    max_new_tokens = max_new_tokens if max_new_tokens is not None else get_generator_max_new_tokens()
    temperature = temperature if temperature is not None else get_generator_temperature()
    top_p = top_p if top_p is not None else get_generator_top_p()
    
    model = get_generator_model_name()
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Answer the question based on the provided context. Be concise and cite sources when possible."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"}
    ]
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=False,
        )
        return response.choices[0].message.content or ""
    except Exception as e:
        error(f"OpenAI generation failed: {e}", "generator.openai")
        raise


def generate_stream(
    query: str,
    context: str,
    max_new_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
) -> Iterator[str]:
    """Generate answer using OpenAI API (streaming)."""
    client = get_client()
    
    max_new_tokens = max_new_tokens if max_new_tokens is not None else get_generator_max_new_tokens()
    temperature = temperature if temperature is not None else get_generator_temperature()
    top_p = top_p if top_p is not None else get_generator_top_p()
    
    model = get_generator_model_name()
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Answer the question based on the provided context. Be concise and cite sources when possible."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"}
    ]
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=True,
        )
        
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    except Exception as e:
        error(f"OpenAI streaming failed: {e}", "generator.openai")
        raise