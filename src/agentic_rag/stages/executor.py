"""Stage 2 - Executor.

Deterministic, parallel tool execution. No LLM. This is the core "extendable"
hook for your data layer: register tools with `@tool(...)` and they are called
here with identity/tenant injected via `ToolContext`. Unknown tools raise
ToolNotFoundError (map to 400 upstream).
"""
from __future__ import annotations

import asyncio

from .base import Stage, StageContext
from ..types import RetrievalPlan, ToolResult
from ..tools.base import ToolExecutor, ToolNotFoundError


class ExecutorStage(Stage):
    name = "executor"

    def run(self, ctx: StageContext, plan: RetrievalPlan | None = None) -> list[ToolResult]:
        executor = ToolExecutor(ctx.tool_ctx)
        results: list[ToolResult] = []

        def call_one(plan_item):
            try:
                data = executor.execute(plan_item.name, plan_item.args)
                return ToolResult(tool_name=plan_item.name, data=data)
            except ToolNotFoundError as e:
                return ToolResult(tool_name=plan_item.name, error=str(e))
            except Exception as e:  # tool bug -> isolated failure, not pipeline crash
                return ToolResult(tool_name=plan_item.name, error=f"tool error: {e}")

        loop = asyncio.new_event_loop()
        try:
            tasks = [loop.run_in_executor(None, call_one, item) for item in (plan.tools if plan else [])]
            results = list(loop.run_until_complete(asyncio.gather(*tasks)))
        finally:
            loop.close()
        return results
