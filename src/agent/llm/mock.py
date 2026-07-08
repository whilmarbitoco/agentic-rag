"""Deterministic mock provider for tests / offline development.

Echoes configurable answers so the pipeline can be exercised with no network
call. Useful for CI, local experimentation, and as a concrete reference for
how to implement `LLMProvider`.
"""
from __future__ import annotations

import json
from typing import Any, Callable

from .base import LLMProvider, extract_json


class MockProvider(LLMProvider):
    def __init__(self, responder: Callable[[list[dict], str | None, bool], Any] | None = None, context_window: int = 32_000):
        self.responder = responder or self._default
        self._context_window = context_window

    @property
    def context_window(self) -> int:
        return self._context_window

    def _default(self, messages, system_prompt, json_mode):
        if json_mode:
            return {
                "execution_mode": "full",
                "intent": "general",
                "is_followup": False,
                "memory_hint": "",
                "in_domain": True,
                "rewritten_query": "mock query",
                "tools": [],
                "reasoning": "mock planner",
            }
        last = messages[-1]["content"] if messages else ""
        return f"[mock reply] {last}"

    def complete(self, messages: list[dict], system_prompt: str | None = None) -> str:
        out = self.responder(messages, system_prompt, False)
        return out if isinstance(out, str) else json.dumps(out)

    def complete_json(self, messages: list[dict], system_prompt: str | None = None) -> dict:
        out = self.responder(messages, system_prompt, True)
        return out if isinstance(out, dict) else extract_json(out)
