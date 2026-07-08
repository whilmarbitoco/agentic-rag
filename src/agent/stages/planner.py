"""Stage 1 - Planner.

Selects which tools to call and rewrites the query for retrieval. Returns an
empty tool list + in_domain=False for out-of-scope questions (the pipeline
then answers from knowledge only). This is where you'd add an in-domain guard.
"""
from __future__ import annotations

from .base import Stage, StageContext
from ..types import InterpretedQuery, RetrievalPlan, ToolCallPlan
from ..llm.base import LLMProvider
from ..tools.base import get_tool_definitions
from ..prompts import PLANNER_SYSTEM, PLANNER_USER


class PlannerStage(Stage):
    name = "planner"

    def __init__(self, llm: LLMProvider | None = None):
        self.llm = llm

    def run(self, ctx: StageContext, interpreted: InterpretedQuery) -> RetrievalPlan:
        provider = self.llm or ctx.llm.get("planner")
        tool_defs = get_tool_definitions()
        messages = [
            {
                "role": "user",
                "content": PLANNER_USER.format(
                    query=interpreted.resolved_query,
                    tools=tool_defs,
                ),
            }
        ]
        out = provider.complete_json(messages, system_prompt=PLANNER_SYSTEM)
        tools = [
            ToolCallPlan(name=t["name"], args=t.get("args", {}))
            for t in out.get("tools", [])
            if t.get("name")
        ]
        return RetrievalPlan(
            in_domain=out.get("in_domain", True),
            rewritten_query=out.get("rewritten_query", interpreted.resolved_query),
            tools=tools,
            reasoning=out.get("reasoning", ""),
        )
