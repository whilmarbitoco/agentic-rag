"""Stage 0 - Interpreter.

Resolves references, detects follow-ups, pulls memory, and selects the
execution mode. The default implementation is a thin JSON-call wrapper; the
prompts live in `prompts.py` so you can override them wholesale. Swap this
stage by subclassing `Stage` and injecting your own logic.
"""
from __future__ import annotations

from .base import Stage, StageContext
from ..types import InterpretedQuery
from ..llm.base import LLMProvider
from ..memory.base import MemoryModule
from ..prompts import INTERPRETER_SYSTEM, INTERPRETER_USER


class InterpreterStage(Stage):
    name = "interpreter"

    def __init__(self, llm: LLMProvider | None = None):
        self.llm = llm  # resolved lazily from ctx.llm if None

    def run(self, ctx: StageContext, query: str) -> InterpretedQuery:
        provider = self.llm or ctx.llm.get("interpreter")
        mem_ctx = ctx.memory.fetch_context(query)
        messages = [
            {"role": "user", "content": INTERPRETER_USER.format(query=query, memory=mem_ctx)}
        ]
        out = provider.complete_json(messages, system_prompt=INTERPRETER_SYSTEM)
        return InterpretedQuery(
            original_query=query,
            resolved_query=out.get("resolved_query", query),
            execution_mode=out.get("execution_mode", "full"),
            intent=out.get("intent", "general"),
            is_followup=out.get("is_followup", False),
            memory_hint=out.get("memory_hint", ""),
            fetch_tools=out.get("fetch_tools", []),
        )
