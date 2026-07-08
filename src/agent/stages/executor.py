"""Stage 2 - Executor.

Deterministic, parallel tool execution. No LLM. This is the core "extendable"
hook for your data layer: register tools with `@tool(...)` and they are called
here with identity/tenant injected via `ToolContext`. Unknown tools raise
ToolNotFoundError (map to 400 upstream).
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from ..tools.base import ToolExecutor, ToolNotFoundError
from ..types import RetrievalPlan, ToolResult
from .base import Stage, StageContext


class ExecutorStage(Stage):
    name = "executor"

    def __init__(self, max_workers: int = 4):
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

    def run(self, ctx: StageContext, plan: RetrievalPlan | None = None) -> list[ToolResult]:
        executor = ToolExecutor(ctx.tool_ctx)

        def call_one(plan_item):
            try:
                data = executor.execute(plan_item.name, plan_item.args)
                return ToolResult(tool_name=plan_item.name, data=data)
            except ToolNotFoundError as e:
                return ToolResult(tool_name=plan_item.name, error=str(e))
            except Exception as e:
                return ToolResult(tool_name=plan_item.name, error=f"tool error: {e}")

        items = plan.tools if plan else []
        return list(self._executor.map(call_one, items))
