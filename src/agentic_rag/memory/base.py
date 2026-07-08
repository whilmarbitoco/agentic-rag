"""Memory module base interface + a reference implementation.

Extendable-first: `MemoryModule` is the ABC you subclass to plug any store
(pgvector, FTS, Redis, a file, an API). `InMemoryMemory` is a ready-to-use
reference that needs no external service — handy for demos, tests, and
single-process apps.

The framework depends only on:
  - fetch_context(query) -> str          (feeds the interpreter)
  - store(user_message, reply)           (fire-and-forget, called after a turn)
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from datetime import datetime, timezone


class MemoryModule(ABC):
    @abstractmethod
    def fetch_context(self, query: str) -> str:
        """Return formatted memory context for the interpreter."""
        ...

    @abstractmethod
    def store(self, user_message: str, reply: str) -> None:
        """Persist a conversation turn (called fire-and-forget)."""
        ...


class NoOpMemory(MemoryModule):
    """Drops everything. Use when memory is not needed."""

    def fetch_context(self, query: str) -> str:
        return ""

    def store(self, user_message: str, reply: str) -> None:
        return None


class InMemoryMemory(MemoryModule):
    """Reference implementation: keeps the last N turns in a bounded deque.

    Swap this for a pgvector/FTS-backed subclass in production — same interface.
    """

    def __init__(self, max_turns: int = 20):
        self.max_turns = max_turns
        self._turns: deque[tuple[str, str, str]] = deque(maxlen=max_turns)

    def store(self, user_message: str, reply: str) -> None:
        ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self._turns.append((ts, user_message, reply))

    def fetch_context(self, query: str) -> str:
        if not self._turns:
            return ""
        lines = ["Recent conversation:"]
        for ts, u, r in self._turns:
            lines.append(f"[{ts}] User: {u}")
            lines.append(f"[{ts}] Assistant: {r}")
        return "\n".join(lines)
