from __future__ import annotations

from typing import Iterator, Optional

from generator.prompts import build_messages
from utils.config import (
    get_generator_max_new_tokens,
    get_generator_model_name,
    get_generator_temperature,
    get_generator_top_p,
)
from utils.logging import debug, error


_client = None
_tokenizer = None


def get_client_and_tokenizer():
    global _client, _tokenizer
    if _client is not None and _tokenizer is not None:
        return _client, _tokenizer

    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = get_generator_model_name()
    debug(f"loading local model: {model_name}", "generator.local")

    _tokenizer = AutoTokenizer.from_pretrained(model_name)
    _client = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
    )
    return _client, _tokenizer


def _build_prompt(query: str, context: str) -> str:
    model, tokenizer = get_client_and_tokenizer()
    messages = build_messages(query, context)
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    lines = []
    for message in messages:
        lines.append(f"{message['role']}: {message['content']}")
    lines.append("assistant:")
    return "\n".join(lines)


def generate(
    query: str,
    context: str,
    max_new_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
) -> str:
    model, tokenizer = get_client_and_tokenizer()
    max_new_tokens = max_new_tokens if max_new_tokens is not None else get_generator_max_new_tokens()
    temperature = temperature if temperature is not None else get_generator_temperature()
    top_p = top_p if top_p is not None else get_generator_top_p()
    prompt = _build_prompt(query, context)

    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
        )
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer[len(prompt):].strip()
    except Exception as exc:
        error(f"Local generation failed: {exc}", "generator.local")
        raise


def generate_stream(
    query: str,
    context: str,
    max_new_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
) -> Iterator[str]:
    from threading import Thread

    from transformers import TextIteratorStreamer

    model, tokenizer = get_client_and_tokenizer()
    max_new_tokens = max_new_tokens if max_new_tokens is not None else get_generator_max_new_tokens()
    temperature = temperature if temperature is not None else get_generator_temperature()
    top_p = top_p if top_p is not None else get_generator_top_p()
    prompt = _build_prompt(query, context)

    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        streamer = TextIteratorStreamer(
            tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )
        generation_kwargs = dict(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
            streamer=streamer,
        )
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()
        for token in streamer:
            yield token
    except Exception as exc:
        error(f"Local streaming failed: {exc}", "generator.local")
        raise
