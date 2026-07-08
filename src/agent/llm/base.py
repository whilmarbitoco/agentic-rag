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
from typing import Any


class LLMProvider(ABC):
    @abstractmethod
    def complete(self, messages: list[dict], system_prompt: str | None = None) -> str:
        """Return the model's text completion for the given chat messages."""
        ...

    def complete_json(self, messages: list[dict], system_prompt: str | None = None) -> dict:
        """Return parsed JSON. Default: text completion + extraction.

        Override in subclasses that have a native JSON mode.
        """
        return extract_json(self.complete(messages, system_prompt))


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
