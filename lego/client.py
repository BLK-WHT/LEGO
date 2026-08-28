from __future__ import annotations

import os
from functools import lru_cache

import numpy as np
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

EMBED_BATCH_SIZE = 32
_RETRY = dict(stop=stop_after_attempt(5), wait=wait_exponential(min=2, max=30), reraise=True)


def _require(name: str) -> str:
    value = os.environ.get(name, "")
    if not value:
        raise RuntimeError(f"{name} is not set; see README.md")
    return value


@lru_cache(maxsize=None)
def _client(base_url: str, api_key: str) -> OpenAI:
    return OpenAI(base_url=base_url, api_key=api_key, timeout=180.0)


def _thinking_off() -> dict:
    if os.environ.get("LLM_THINKING_STYLE", "nested").lower() == "flat":
        return {"enable_thinking": False}
    return {"chat_template_kwargs": {"enable_thinking": False}}


@retry(**_RETRY)
def chat(system: str, user: str, max_tokens: int = 1800, temperature: float = 0.0) -> str:
    client = _client(_require("LLM_BASE_URL"), _require("LLM_API_KEY"))
    response = client.chat.completions.create(
        model=os.environ.get("LLM_MODEL", "qwen3-8b"),
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=temperature,
        max_tokens=max_tokens,
        extra_body=_thinking_off(),
    )
    return response.choices[0].message.content or ""


@retry(**_RETRY)
def _embed_batch(texts: list[str]) -> list[list[float]]:
    client = _client(
        _require("EMBED_BASE_URL"), os.environ.get("EMBED_API_KEY", "EMPTY")
    )
    response = client.embeddings.create(
        model=os.environ.get("EMBED_MODEL", "Qwen3-Embedding-8B"), input=texts
    )
    return [item.embedding for item in response.data]


def embed(texts: list[str]) -> np.ndarray:
    vectors: list[list[float]] = []
    for start in range(0, len(texts), EMBED_BATCH_SIZE):
        vectors.extend(_embed_batch(texts[start : start + EMBED_BATCH_SIZE]))
    return np.asarray(vectors, dtype=np.float32)
