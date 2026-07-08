"""LLM provider base interface.

Subclass `LLMProvider` to add any backend (OpenAI, Groq, NVIDIA, Anthropic,
Ollama, vLLM, a custom in-house model, a rule engine, etc.). The framework
only depends on `complete()` and `complete_json()`.

`complete_json()` has a default implementation (calls `complete` + best-effort
JSON extraction) so a minimal subclass only needs to implement `complete`.
Providers with a native JSON mode (OpenAI `response_format`, Groq, etc.)
should override `complete_json` for reliability.
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod

CONTEXT_WINDOW_FALLBACK: int = 128_000


class LLMProvider(ABC):
    @abstractmethod
    def complete(self, messages: list[dict], system_prompt: str | None = None) -> str:
        ...

    def complete_json(self, messages: list[dict], system_prompt: str | None = None) -> dict:
        return extract_json(self.complete(messages, system_prompt))

    @property
    def context_window(self) -> int:
        """Total token capacity (input + output) for this model.

        Subclasses override. Default assumes a modern model (128K).
        """
        return CONTEXT_WINDOW_FALLBACK


def extract_json(raw: str) -> dict:
    """Best-effort JSON extraction from a model response (public helper)."""
    raw = (raw or "").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    start, end = raw.find("{"), raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(raw[start : end + 1])
        except json.JSONDecodeError:
            pass
    raise ValueError(f"Could not parse JSON from response: {raw[:200]}")
