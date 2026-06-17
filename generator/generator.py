# Generator module for RAG (Retrieval-Augmented Generation)
#
# To change the model, edit config/config.cfg or utils/config.py GENERATOR_DEFAULTS:
# - For free HuggingFace API: mistralai/Mistral-7B-Instruct-v0.3
# - For paid HF API: mistralai/Mistral-7B-Instruct-v0.2, microsoft/Phi-3-mini-4k-instruct
# - For OpenAI: Set provider=openai and model_name=gpt-4o-mini (requires OPENAI_API_KEY)
# - For local: Set provider=local and install transformers+torch (requires GPU for best performance)
from __future__ import annotations

import json
from typing import Any, Dict, Iterator, List, Optional

from generator.prompts import SYSTEM_PROMPT, PROMPT_TEMPLATE
from utils.config import (
    get_generator_provider,
    get_generator_model_name,
    get_generator_max_new_tokens,
    get_generator_temperature,
    get_generator_top_p,
    get_hf_token,
)
from utils.logging import debug, error


def _format_context_chunks(results: List[Dict[str, Any]]) -> str:
    """Format search results into context for LLM prompt."""
    if not results:
        return "No relevant context found."
    
    parts = []
    for result in results:
        chunks = result.get("chunks") or []
        best_chunk = result.get("best_chunk") or ""
        
        if isinstance(chunks, list) and chunks:
            texts = [str(c.get("text", "")) for c in chunks if isinstance(c, dict)]
            context_text = "\n\n".join(t for t in texts if t)
        elif best_chunk:
            context_text = best_chunk
        else:
            continue
            
        source = result.get("source", "Unknown source")
        title = result.get("title", "Untitled")
        chapter = result.get("chapter") or result.get("location", {}).get("chapter")
        header = f"Source: {title}" + (f" (Chapter: {chapter})" if chapter else "")
        parts.append(f"{header}\n{context_text}")
    
    return "\n\n---\n\n".join(parts) if parts else "No relevant context found."


def format_context(results: List[Dict[str, Any]]) -> str:
    """Public interface to format context from search results."""
    return _format_context_chunks(results)


def _get_hf_client():
    """Get HuggingFace InferenceClient instance."""
    from huggingface_hub import InferenceClient
    token = get_hf_token()
    model = get_generator_model_name()
    debug(f"creating InferenceClient model={model}", "generator")
    if not token:
        raise ValueError("HF_API_KEY not configured. Set it in config.cfg [hf] section or HF_API_KEY environment variable.")
    return InferenceClient(token=token, model=model)


def generate_answer(
    query: str,
    context: Optional[str] = None,
    search_results: Optional[List[Dict[str, Any]]] = None,
    max_new_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    stream: bool = False,
) -> str | Iterator[str]:
    """
    Generate an answer using HuggingFace Inference API.
    
    Args:
        query: The user's question
        context: Pre-formatted context string (optional)
        search_results: Search results to format as context (optional)
        max_new_tokens: Maximum tokens in response
        temperature: Sampling temperature
        top_p: Top-p nucleus sampling
        stream: Return iterator for streaming responses
        
    Returns:
        Generated answer string, or iterator of tokens if stream=True
    """
    if context is None and search_results:
        context = _format_context_chunks(search_results)
    
    if context is None:
        context = "No context provided."
    
    max_new_tokens = max_new_tokens if max_new_tokens is not None else get_generator_max_new_tokens()
    temperature = temperature if temperature is not None else get_generator_temperature()
    top_p = top_p if top_p is not None else get_generator_top_p()
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": PROMPT_TEMPLATE.format(context=context, query=query)}
    ]
    
    client = _get_hf_client()
    
    try:
        debug(f"generating answer stream={stream} max_tokens={max_new_tokens}", "generator")
        
        response = client.chat.completions.create(
            messages=messages,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=stream,
        )
        
        if stream:
            return _stream_response(response)
        
        answer = response.choices[0].message.content or ""
        debug(f"generated answer length={len(answer)}", "generator")
        return answer
        
    except Exception as e:
        error(f"generation failed: {e}", "generator")
        raise


def _stream_response(response) -> Iterator[str]:
    """Yield tokens from streaming response."""
    for chunk in response:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


def get_generator():
    """Get the generator instance (alias for compatibility)."""
    return {"provider": get_generator_provider(), "model": get_generator_model_name}
