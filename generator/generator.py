from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Mapping, Optional

from generator.prompts import NO_ANSWER, NO_CONTEXT, build_messages
from utils.config import (
    get_generator_max_new_tokens,
    get_generator_model_name,
    get_generator_provider,
    get_generator_temperature,
    get_generator_top_p,
)
from utils.logging import debug, error


@dataclass(frozen=True)
class GenerationConfig:
    provider: str
    model_name: str
    max_new_tokens: int
    temperature: float
    top_p: float


def _resolve_provider(provider: str):
    from generator.providers import get_provider
    return get_provider(provider)


def get_generation_config() -> GenerationConfig:
    return GenerationConfig(
        provider=get_generator_provider(),
        model_name=get_generator_model_name(),
        max_new_tokens=get_generator_max_new_tokens(),
        temperature=get_generator_temperature(),
        top_p=get_generator_top_p(),
    )


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())


def _chunk_text(chunk: Any) -> str:
    if isinstance(chunk, dict):
        return _clean_text(chunk.get("text") or chunk.get("content"))
    return _clean_text(chunk)


def _format_location(item: Mapping[str, Any]) -> str:
    metadata = item.get("metadata") if isinstance(item, Mapping) else {}
    if not isinstance(metadata, Mapping):
        metadata = {}

    book = item.get("book") or metadata.get("book")
    chapter = item.get("chapter") or metadata.get("chapter")
    verse = item.get("verse") or metadata.get("verse")
    section = item.get("section") or metadata.get("section")
    
    if book:
        ref = book
        if chapter:
            ref += f" {chapter}"
            if verse:
                ref += f":{verse}"
        elif section:
            ref += f" {section}"
        return ref
    return ""


def _format_result(result: Mapping[str, Any], index: int) -> Optional[str]:
    chunks = result.get("chunks") or []
    context_text = ""

    if isinstance(chunks, list) and chunks:
        texts = [_chunk_text(chunk) for chunk in chunks]
        context_text = "\n\n".join(text for text in texts if text)
    else:
        context_text = _chunk_text(result.get("best_chunk") or result.get("content"))

    if not context_text:
        return None

    title = _clean_text(result.get("title") or result.get("source") or f"Source {index}")
    location = _format_location(result)
    score = result.get("score")
    score_text = f" [score={score:.4f}]" if isinstance(score, (int, float)) else ""

    if location:
        header = f"[{location}]"
    else:
        header = f"[{index}]"
    cleaned_text = _strip_leading_ref(context_text, location)
    return f"{header}\n{cleaned_text}"


def _strip_leading_ref(text: str, location: str) -> str:
    import re
    if not location:
        return text
    pattern = rf"^{re.escape(location)}\s*[-–—]?\s*"
    return re.sub(pattern, "", text, flags=re.IGNORECASE)


def format_context(results: Optional[List[Mapping[str, Any]]]) -> str:
    if not results:
        return NO_CONTEXT

    parts = [_format_result(result, index) for index, result in enumerate(results, start=1)]
    parts = [part for part in parts if part]
    return "\n\n---\n\n".join(parts) if parts else NO_CONTEXT


def generate_answer(
    query: str,
    context: Optional[str] = None,
    search_results: Optional[List[Mapping[str, Any]]] = None,
    max_new_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    stream: bool = False,
    history: Optional[List[Dict[str, str]]] = None,
) -> str | Iterator[str]:
    if context is None and search_results:
        context = format_context(search_results)

    if not context or context == NO_CONTEXT:
        return iter([NO_ANSWER]) if stream else NO_ANSWER

    config = get_generation_config()
    provider = _resolve_provider(config.provider)
    max_new_tokens = max_new_tokens if max_new_tokens is not None else config.max_new_tokens
    temperature = temperature if temperature is not None else config.temperature
    top_p = top_p if top_p is not None else config.top_p

    debug(
        f"generate provider={config.provider} model={config.model_name} stream={stream} max_tokens={max_new_tokens}",
        "generator",
    )

    try:
        if stream:
            return _clean_stream(
                provider.generate_stream(
                    query=query,
                    context=context,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    history=history,
                )
            )

        answer = provider.generate(
            query=query,
            context=context,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            history=history,
        )
        debug(f"generated answer length={len(answer)}", "generator")
        return _clean_response(answer)
    except Exception as exc:
        error(f"generation failed: {exc}", "generator")
        raise


def _clean_error_msg(msg: str) -> str:
    import re
    cleaned = re.sub(r"<environment_details>.*?</environment_details>", "", msg, flags=re.DOTALL | re.IGNORECASE)
    return cleaned.strip()


def _clean_response(text: str) -> str:
    import re
    cleaned = re.sub(r"<environment_details>.*?</environment_details>", "", text, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"<environment_details>.*", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"<thinking>.*?</thinking>", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"<thinking>.*", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"<reasoning>.*?</reasoning>", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"<reasoning>.*", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    return cleaned.strip()


def _clean_stream(stream: Iterator[str]) -> Iterator[str]:
    import re
    buffer = ""
    tag_re = re.compile(r"<(?:environment_details|thinking|reasoning)>.*?</(?:environment_details|thinking|reasoning)>", flags=re.DOTALL | re.IGNORECASE)
    unclosed_re = re.compile(r"<(?:environment_details|thinking|reasoning)>.*", flags=re.DOTALL | re.IGNORECASE)
    for chunk in stream:
        buffer += chunk
        cleaned = tag_re.sub("", buffer)
        if cleaned != buffer:
            buffer = cleaned
        if buffer:
            yield buffer
            buffer = ""
    if buffer:
        cleaned = unclosed_re.sub("", buffer)
        if cleaned:
            yield cleaned


def get_generator() -> Dict[str, Any]:
    config = get_generation_config()
    return {
        "provider": config.provider,
        "model": config.model_name,
        "max_new_tokens": config.max_new_tokens,
        "temperature": config.temperature,
        "top_p": config.top_p,
    }


__all__ = ["GenerationConfig", "build_messages", "format_context", "generate_answer", "get_generator"]
