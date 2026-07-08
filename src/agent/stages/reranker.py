"""Stage 3 - Reranker.

Scores and ranks tool results by relevance to the rewritten query. The default
implementation is a simple heuristic (non-null data always passes, nulls drop).
Override with a cross-encoder / LLM judge by subclassing `Stage`.
"""
from __future__ import annotations

from .base import Stage, StageContext
from ..types import ToolResult, RankedResult
from ..llm.base import LLMProvider


class RerankerStage(Stage):
    name = "reranker"

    def __init__(self, llm: LLMProvider | None = None):
        self.llm = llm

    def run(self, ctx: StageContext, results: list[ToolResult]) -> list[RankedResult]:
        ranked: list[RankedResult] = []
        for r in results:
            if r.error or r.data is None:
                continue
            ranked.append(
                RankedResult(
                    tool_name=r.tool_name,
                    data=r.data,
                    score=1,
                    reason="retrieved",
                )
            )
        return ranked
