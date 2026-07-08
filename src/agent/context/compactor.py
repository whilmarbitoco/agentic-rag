"""Source compaction strategies.

A `Compactor` shrinks a list of `RankedResult` so the serialized payload fits a
token budget before it reaches the synthesizer. The default `HeuristicCompactor`
keeps the highest-scored results, truncates oversized items, and drops the rest.
`LLMCompactor` (optional) asks an `LLMProvider` to summarize everything into one
compact, source-tagged block.
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod

from ..types import RankedResult
from ..llm.base import LLMProvider


def _as_text(data, cap: int = 2000) -> str:
    if isinstance(data, (dict, list)):
        return json.dumps(data, ensure_ascii=False, default=str)[:cap]
    return str(data)[:cap]


def _truncate_to_tokens(text: str, budget_tokens: int, counter) -> str:
    """Binary-search shrink `text` to <= budget_tokens (token-aware)."""
    if counter.count(text) <= budget_tokens:
        return text
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if counter.count(text[:mid]) <= budget_tokens:
            lo = mid
        else:
            hi = mid - 1
    cut = max(0, lo - 1)
    return text[:cut] + " …[truncated]"


class Compactor(ABC):
    @abstractmethod
    def compact(
        self, items: list[RankedResult], budget_tokens: int, counter
    ) -> list[RankedResult]:
        """Return a budget-fitting subset of `items` (may rewrite their data)."""


class HeuristicCompactor(Compactor):
    """Keep highest-scored first; truncate the overflowing item; drop the rest."""

    def compact(self, items, budget_tokens, counter) -> list[RankedResult]:
        ordered = sorted(items, key=lambda r: getattr(r, "score", 0), reverse=True)
        kept: list[RankedResult] = []
        used = 0
        for r in ordered:
            text = _as_text(r.data)
            t = counter.count(text)
            if used + t <= budget_tokens:
                kept.append(r)
                used += t
                continue
            remaining = budget_tokens - used
            if remaining > 40:
                # reserve room for the " …[truncated]" tail (~15 tokens)
                truncated = _truncate_to_tokens(text, max(0, remaining - 15), counter)
                kept.append(
                    RankedResult(
                        tool_name=r.tool_name,
                        data={"summary": truncated},
                        score=r.score,
                        reason=r.reason,
                    )
                )
            break  # budget exhausted
        return kept


class LLMCompactor(Compactor):
    """Summarize all tool results into one compact, source-tagged block.

    Uses an `LLMProvider` (any OpenAI-compatible model). The summary is itself
    truncated to the budget as a safety net.
    """

    def __init__(self, provider: LLMProvider):
        self.provider = provider

    def compact(self, items, budget_tokens, counter) -> list[RankedResult]:
        if not items:
            return []
        blob = "\n".join(f"[{r.tool_name}] {_as_text(r.data)}" for r in items)
        prompt = (
            f"Summarize the following tool results into a single compact block under "
            f"{budget_tokens} tokens. Preserve every concrete value and tag each fact with "
            f"its source tool name in brackets, e.g. [tool_name]. Do not invent data.\n\n{blob}"
        )
        summary = self.provider.complete([{"role": "user", "content": prompt}])
        if counter.count(summary) > budget_tokens:
            summary = _truncate_to_tokens(summary, budget_tokens, counter)
        return [
            RankedResult(
                tool_name="compressed",
                data={"summary": summary},
                score=max((r.score for r in items), default=1),
                reason="llm-compressed",
            )
        ]
