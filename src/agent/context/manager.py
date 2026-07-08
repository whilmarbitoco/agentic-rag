"""Context manager: allocates a token budget across pipeline sections.

The default `BudgetContextManager`:
  - computes section budgets dynamically from the provider's context window,
  - budgets the memory context fed to the interpreter,
  - compacts the tool results fed to the synthesizer,
  - enforces a total budget across all sections,
  - records per-section token usage for observability (trace["context"]).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from ..types import RankedResult
from .tokens import TokenCounter, WordTokenCounter
from .compactor import Compactor, HeuristicCompactor, _as_text, _truncate_to_tokens

OVERHEAD: int = 500  # prompt overhead (system, formatting, role tags)

_DEFAULT_MEMORY_RATIO: float = 0.25
_DEFAULT_TOOLS_RATIO: float = 0.50
_DEFAULT_HISTORY_RATIO: float = 0.25


def _compute_budgets_from_window(context_window: int) -> dict[str, int]:
    available = max(1000, context_window - OVERHEAD)
    return {
        "memory": min(2000, int(available * _DEFAULT_MEMORY_RATIO)),
        "tools": max(500, int(available * _DEFAULT_TOOLS_RATIO)),
        "history": max(500, int(available * _DEFAULT_HISTORY_RATIO)),
        "total": available,
    }


DEFAULT_BUDGETS: dict[str, int] = _compute_budgets_from_window(128_000)


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
        context_window: int | None = None,
    ):
        if context_window is not None:
            base = _compute_budgets_from_window(context_window)
        else:
            base = dict(DEFAULT_BUDGETS)
        self.budgets = base
        if budgets:
            self.budgets.update(budgets)
        self.counter = counter or WordTokenCounter()
        self.compactor = compactor or HeuristicCompactor()
        self.reset()

    def reset(self):
        self._usage: dict[str, int] = {k: 0 for k in self.budgets}

    def _section_limit(self, section: str) -> int:
        return self.budgets.get(section, 4000)

    def _cumulative(self) -> int:
        return sum(self._usage.values())

    def budget_text(self, section: str, text: str) -> str:
        limit = self._section_limit(section)
        cum = self._cumulative()
        total_limit = self._section_limit("total")
        if cum >= total_limit:
            return "\n…[total context budget exhausted]"
        n = self.counter.count(text)
        remaining_total = total_limit - cum
        effective = min(limit, remaining_total)
        if n <= effective:
            self._usage[section] = n
            return text
        trimmed = _truncate_to_tokens(text, effective, self.counter)
        self._usage[section] = self.counter.count(trimmed)
        return trimmed + "\n…[context truncated to budget]"

    def prepare_sources(self, ranked, counter=None, compactor=None) -> list[RankedResult]:
        counter = counter or self.counter
        compactor = compactor or self.compactor
        limit = self._section_limit("tools")
        cum = self._cumulative()
        remaining_total = self._section_limit("total") - cum
        effective = min(limit, remaining_total)
        if effective <= 0:
            return []
        compacted = compactor.compact(ranked, effective, counter)
        self._usage["tools"] = sum(
            counter.count(_as_text(r.data)) for r in compacted
        )
        return compacted

    def report(self) -> dict:
        return dict(self._usage)