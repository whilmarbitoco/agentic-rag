"""Stage 3 - Reranker.

Scores and ranks tool results by relevance to the rewritten query. The default
implementation is a simple heuristic (non-null data always passes, nulls drop).
Override with a cross-encoder / LLM judge by subclassing `Stage`.
"""
from __future__ import annotations

from ..llm.base import LLMProvider
from ..types import RankedResult, ToolResult
from .base import Stage, StageContext


class RerankerStage(Stage):
    name = "reranker"

    def __init__(self, llm: LLMProvider | None = None):
        self.llm = llm

    def run(self, ctx: StageContext, results: list[ToolResult]) -> list[RankedResult]:
        ranked: list[RankedResult] = []
        for r in results:
            if r.error or r.data is None:
                continue
            scored = self._score(ctx, r)
            ranked.append(scored)
        # highest score first so the compactor keeps the most relevant results
        ranked.sort(key=lambda x: x.score, reverse=True)
        return ranked

    def _score(self, ctx: StageContext, r: ToolResult) -> RankedResult:
        provider = self.llm or ctx.llm.get("reranker")
        if provider is None:
            return RankedResult(tool_name=r.tool_name, data=r.data, score=1, reason="retrieved")
        query = ctx.state.get("resolved_query", "")
        try:
            out = provider.complete_json(
                [
                    {
                        "role": "user",
                        "content": f"Query: {query}\nSource: {r.tool_name}",
                    }
                ],
                system_prompt=(
                    "Score how relevant this source is to the query on a 0-3 scale. "
                    "Respond ONLY with JSON: {\"score\": int, \"reason\": str}"
                ),
            )
            score = int(out.get("score", 1))
            return RankedResult(
                tool_name=r.tool_name,
                data=r.data,
                score=max(0, min(3, score)),
                reason=str(out.get("reason", "reranked"))[:200],
            )
        except Exception:
            return RankedResult(tool_name=r.tool_name, data=r.data, score=1, reason="retrieved")
