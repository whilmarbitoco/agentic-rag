"""Context manager: allocates a token budget across pipeline sections.

The default `BudgetContextManager`:
  - budgets the memory context fed to the interpreter,
  - compacts the tool results fed to the synthesizer,
  - records per-section token usage for observability (trace["context"]).

Subclass to implement adaptive budgeting (e.g. steal from `history` when `tools`
overflows, or call an LLM judge to choose what to keep).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from ..types import RankedResult
from .tokens import TokenCounter, WordTokenCounter
from .compactor import Compactor, HeuristicCompactor, _as_text, _truncate_to_tokens

DEFAULT_BUDGETS: dict[str, int] = {
    "memory": 1500,
    "tools": 3000,
    "history": 1500,
    "total": 6000,
}


class ContextManager(ABC):
    @abstractmethod
    def budget_text(self, section: str, text: str) -> str:
        """Return a budget-fitting version of `text` for `section`."""

    @abstractmethod
    def prepare_sources(
        self, ranked: list[RankedResult], counter=None, compactor=None
    ) -> list[RankedResult]:
        """Compact tool results to the `tools` budget; return kept results."""

    @abstractmethod
    def report(self) -> dict:
        """Return per-section token usage for tracing/observability."""


class BudgetContextManager(ContextManager):
    def __init__(
        self,
        budgets: Optional[dict[str, int]] = None,
        counter: Optional[TokenCounter] = None,
        compactor: Optional[Compactor] = None,
    ):
        self.budgets = dict(DEFAULT_BUDGETS)
        if budgets:
            self.budgets.update(budgets)
        self.counter = counter or WordTokenCounter()
        self.compactor = compactor or HeuristicCompactor()
        self._usage: dict[str, int] = {k: 0 for k in self.budgets}

    def budget_text(self, section: str, text: str) -> str:
        limit = self.budgets.get(section, 4000)
        n = self.counter.count(text)
        if n <= limit:
            self._usage[section] = n
            return text
        trimmed = _truncate_to_tokens(text, limit, self.counter)
        self._usage[section] = self.counter.count(trimmed)
        return trimmed + "\n…[context truncated to budget]"

    def prepare_sources(self, ranked, counter=None, compactor=None) -> list[RankedResult]:
        counter = counter or self.counter
        compactor = compactor or self.compactor
        compacted = compactor.compact(
            ranked, self.budgets.get("tools", 3000), counter
        )
        self._usage["tools"] = sum(
            counter.count(_as_text(r.data)) for r in compacted
        )
        return compacted

    def report(self) -> dict:
        return dict(self._usage)
