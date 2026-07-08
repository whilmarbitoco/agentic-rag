"""Stage 4 - Synthesizer.

Generates the final answer from the ranked tool data + rewritten query.
XAI-first: the system prompt instructs the model to ground every finding in the
provided sources and cite them. Override by subclassing for a different format.
"""
from __future__ import annotations

from ..llm.base import LLMProvider
from ..prompts import SYNTHESIZER_SYSTEM, SYNTHESIZER_USER
from ..types import InterpretedQuery, RankedResult, RetrievalPlan
from ..utils import as_text
from .base import Stage, StageContext


class SynthesizerStage(Stage):
    name = "synthesizer"

    def __init__(self, llm: LLMProvider | None = None):
        self.llm = llm

    def run(
        self,
        ctx: StageContext,
        interpreted: InterpretedQuery,
        plan: RetrievalPlan,
        ranked: list[RankedResult],
    ) -> str:
        provider = self.llm or ctx.llm.get("synthesizer")
        compacted = ctx.context.prepare_sources(ranked)
        sources = "\n".join(
            f"[{r.tool_name}] {as_text(r.data, cap=4000)}" for r in compacted
        ) or "(no tool data retrieved)"
        critique = ctx.state.get("validator_critique", "")
        user = SYNTHESIZER_USER.format(query=interpreted.resolved_query, sources=sources)
        if critique:
            user += f"\n\nThe previous answer was rejected: {critique}. Revise accordingly."
        return provider.complete(
            [{"role": "user", "content": user}], system_prompt=SYNTHESIZER_SYSTEM
        )
