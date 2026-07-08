"""Provider factory: maps a pipeline *role* to a primary provider plus an
optional fallback (used on rate-limit / failure).

This mirrors the klima-backend design: each stage owns its model, and a
single fallback (e.g. OpenRouter free tier) catches 429s transparently.
"""
from __future__ import annotations

from typing import Callable

from .base import LLMProvider
from .mock import MockProvider
from .openai_compat import OpenAICompatProvider


class ProviderFactory:
    def __init__(self):
        self._primaries: dict[str, LLMProvider] = {}
        self._fallbacks: dict[str, LLMProvider] = {}

    def register(self, role: str, provider: LLMProvider, fallback: LLMProvider | None = None):
        self._primaries[role] = provider
        if fallback:
            self._fallbacks[role] = fallback

    def add_openai_compat(
        self,
        role: str,
        base_url: str,
        api_key: str,
        model: str,
        fallback: LLMProvider | None = None,
        json_mode: bool = False,
    ):
        self.register(
            role,
            OpenAICompatProvider(base_url=base_url, api_key=api_key, model=model, json_mode=json_mode),
            fallback,
        )

    def get(self, role: str) -> LLMProvider:
        return self._primaries.get(role, MockProvider())

    def get_fallback(self, role: str) -> LLMProvider | None:
        return self._fallbacks.get(role)


# Convenience constructors -------------------------------------------------


def make_mock_factory() -> "ProviderFactory":
    """A fully wired factory with mock providers for every role."""
    f = ProviderFactory()
    for role in ["interpreter", "planner", "reranker", "synthesizer", "validator"]:
        f.register(role, MockProvider())
    return f


def make_openai_factory(
    api_key: str,
    *,
    base_url: str = "https://api.openai.com/v1",
    model: str = "gpt-4o-mini",
    fallback: LLMProvider | None = None,
    json_roles: tuple[str, ...] = ("interpreter", "planner", "reranker"),
) -> "ProviderFactory":
    """Wire every stage to the same OpenAI-compatible endpoint.

    Override per-role via `factory.add_openai_compat('synthesizer', ...)`.
    """
    f = ProviderFactory()
    for role in ["interpreter", "planner", "reranker", "synthesizer", "validator"]:
        f.add_openai_compat(
            role,
            base_url=base_url,
            api_key=api_key,
            model=model,
            fallback=fallback,
            json_mode=role in json_roles,
        )
    return f
