"""Token counting strategies.

Override `TokenCounter` to plug an exact tokenizer (tiktoken, HF tokenizer,
a remote counting endpoint, etc.). The default `WordTokenCounter` is a cheap
~4-chars-per-token heuristic that needs no dependency.
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class TokenCounter(ABC):
    @abstractmethod
    def count(self, text: str) -> int:
        """Return the token count for `text`."""


class WordTokenCounter(TokenCounter):
    """Heuristic: ~1 token per 4 characters. Good enough for budgeting."""

    def count(self, text: str) -> int:
        if not text:
            return 0
        return max(1, len(text) // 4)


class TiktokenTokenCounter(TokenCounter):
    """Exact count via `tiktoken` (optional dependency).

    Falls back to cl100k_base if the model name is unknown. Raises a clear
    error if tiktoken is not installed.
    """

    def __init__(self, model: str = "gpt-3.5-turbo"):
        try:
            import tiktoken
        except ImportError as exc:  # pragma: no cover - optional dep
            raise RuntimeError(
                "TiktokenTokenCounter requires `pip install tiktoken`"
            ) from exc
        try:
            self.enc = tiktoken.encoding_for_model(model)
        except Exception:  # unknown model -> safe default
            self.enc = tiktoken.get_encoding("cl100k_base")

    def count(self, text: str) -> int:
        if not text:
            return 0
        return len(self.enc.encode(text))
