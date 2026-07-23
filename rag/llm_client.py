"""
LLM Client — Wrapper around Groq's API for text generation.

Provides both streaming and non-streaming generation,
with response timing and token tracking.
"""

import logging
import time
from typing import Generator

from groq import Groq

import config

logger = logging.getLogger(__name__)


def _get_client() -> Groq:
    """Create a Groq client."""
    return Groq(api_key=config.GROQ_API_KEY)


def generate(
    messages: list[dict],
    model: str | None = None,
    temperature: float = 0.1,
    max_tokens: int = 1024,
) -> dict:
    """
    Generate a response from the LLM (non-streaming).

    Args:
        messages: Chat messages list [{"role": ..., "content": ...}].
        model: Model name (defaults to config.LLM_MODEL).
        temperature: Sampling temperature.
        max_tokens: Maximum tokens in response.

    Returns:
        Dict with: answer, model, tokens_used, response_time_ms
    """
    model = model or config.LLM_MODEL
    client = _get_client()

    start_time = time.time()

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    elapsed_ms = int((time.time() - start_time) * 1000)

    answer = response.choices[0].message.content.strip()
    usage = response.usage

    result = {
        "answer": answer,
        "model": model,
        "tokens_used": {
            "prompt": usage.prompt_tokens if usage else 0,
            "completion": usage.completion_tokens if usage else 0,
            "total": usage.total_tokens if usage else 0,
        },
        "response_time_ms": elapsed_ms,
    }

    logger.info(
        f"Generated response: {result['tokens_used']['total']} tokens, "
        f"{elapsed_ms}ms, model={model}"
    )

    return result


def generate_stream(
    messages: list[dict],
    model: str | None = None,
    temperature: float = 0.1,
    max_tokens: int = 1024,
) -> Generator[str, None, dict]:
    """
    Generate a streaming response from the LLM.

    Yields text chunks as they arrive. After the generator is exhausted,
    its return value contains metadata (use .value after StopIteration).

    Usage:
        gen = generate_stream(messages)
        full_text = ""
        try:
            while True:
                chunk = next(gen)
                full_text += chunk
                print(chunk, end="")
        except StopIteration as e:
            metadata = e.value
    """
    model = model or config.LLM_MODEL
    client = _get_client()

    start_time = time.time()

    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True,
    )

    full_response = ""
    for chunk in stream:
        if chunk.choices[0].delta.content:
            text = chunk.choices[0].delta.content
            full_response += text
            yield text

    elapsed_ms = int((time.time() - start_time) * 1000)

    return {
        "answer": full_response,
        "model": model,
        "response_time_ms": elapsed_ms,
    }


def ask(
    question: str,
    context_results: list[dict],
    prompt_style: str = "detailed",
    model: str | None = None,
    temperature: float = 0.1,
) -> dict:
    """
    High-level convenience: build prompt + generate answer.

    Args:
        question: User's question.
        context_results: Retrieved documents from the retriever.
        prompt_style: One of 'concise', 'detailed', 'structured'.
        model: LLM model name.
        temperature: Sampling temperature.

    Returns:
        Dict with: answer, model, tokens_used, response_time_ms, context_results
    """
    from rag.prompt_builder import build_prompt

    messages = build_prompt(question, context_results, prompt_style)
    result = generate(messages, model=model, temperature=temperature)
    result["context_results"] = context_results
    result["prompt_style"] = prompt_style
    return result
