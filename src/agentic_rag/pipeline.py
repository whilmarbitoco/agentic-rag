"""AgenticPipeline: the 6-stage orchestrator.

Stages (all individually swappable via constructor injection):
  0. Interpreter  - resolve refs, pick execution mode, pull memory
  1. Planner      - select tools + rewrite query
  2. Executor     - deterministic parallel tool calls (no LLM)
  3. Reranker     - score/rank retrieved data
  4. Synthesizer  - generate the final answer
  5. Validator    - self-check; if it fails, regenerate once (feedback loop)
  +  Memory       - background fetch/store for continuity

Extendable-first contract:
  - each stage is a `Stage`; subclass and inject to override behavior.
  - providers are injected (mock by default) so the pipeline runs offline.
  - the orchestrator only wires handoffs; it never hardcodes prompts.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any

from .llm.base import LLMProvider
from .llm.mock import MockProvider
from .memory.base import MemoryModule, NoOpMemory
from .stages.base import Stage, StageContext
from .types import (
    InterpretedQuery,
    RetrievalPlan,
    ToolResult,
    RankedResult,
    ToolCallRecord,
    PipelineResult,
)
from .tools.base import ToolExecutor, ToolNotFoundError, get_tool_definitions
from .errors import PipelineError

from .stages.interpreter import InterpreterStage
from .stages.planner import PlannerStage
from .stages.executor import ExecutorStage
from .stages.reranker import RerankerStage
from .stages.synthesizer import SynthesizerStage
from .stages.validator import ValidatorStage


@dataclass
class AgenticPipeline:
    interpreter: Stage
    planner: Stage
    executor: Stage
    reranker: Stage
    synthesizer: Stage
    validator: Stage
    memory: MemoryModule = field(default_factory=NoOpMemory)
    max_validator_retries: int = 1

    @classmethod
    def build_default(
        cls,
        providers: dict[str, LLMProvider] | None = None,
        memory: MemoryModule | None = None,
    ) -> "AgenticPipeline":
        """Convenience constructor. Uses MockProvider for any missing role."""
        p = providers or {}
        get = lambda role: p.get(role, MockProvider())
        mem = memory or NoOpMemory()
        return cls(
            interpreter=InterpreterStage(get("interpreter")),
            planner=PlannerStage(get("planner")),
            executor=ExecutorStage(),
            reranker=RerankerStage(get("reranker")),
            synthesizer=SynthesizerStage(get("synthesizer")),
            validator=ValidatorStage(get("validator")),
            memory=mem,
        )

    def run(
        self,
        query: str,
        tool_ctx=None,
        state: dict | None = None,
    ) -> PipelineResult:
        from .tools.base import ToolContext

        ctx = StageContext(
            llm={
                "interpreter": _p(self.interpreter, "interpreter"),
                "planner": _p(self.planner, "planner"),
                "reranker": _p(self.reranker, "reranker"),
                "synthesizer": _p(self.synthesizer, "synthesizer"),
                "validator": _p(self.validator, "validator"),
            },
            memory=self.memory,
            tool_ctx=tool_ctx or ToolContext(),
            state=state or {},
        )

        # Stage 0: interpret
        interp: InterpretedQuery = self.interpreter.run(ctx, query=query)

        # Stage 1: plan
        plan: RetrievalPlan = self.planner.run(ctx, interpreted=interp)

        if not plan.in_domain:
            synthetic = self.synthesizer.run(ctx, interpreted=interp, plan=plan, ranked=[])
            return PipelineResult(
                reply=synthetic,
                pipeline_mode="out_of_domain",
                trace={"mode": "out_of_domain"},
            )

        # Stage 2: execute (deterministic, parallel)
        results: list[ToolResult] = self.executor.run(ctx, plan=plan)

        # Stage 3: rerank
        ranked: list[RankedResult] = self.reranker.run(ctx, results=results)

        # Stage 4 + 5: synthesize + validate feedback loop
        reply = ""
        last_candidate = ""
        for _ in range(self.max_validator_retries + 1):
            candidate = self.synthesizer.run(ctx, interpreted=interp, plan=plan, ranked=ranked)
            verdict = self.validator.run(ctx, interpreted=interp, ranked=ranked, candidate=candidate)
            last_candidate = candidate
            if verdict.get("valid", True):
                reply = candidate
                break
            ctx.state["validator_critique"] = verdict.get("critique", "")

        if not reply:
            reply = last_candidate
            ctx.state["validation_warning"] = True

        # Memory (fire-and-forget)
        try:
            self.memory.store(query, reply)
        except Exception:
            pass

        sources: list[ToolCallRecord] = [
            ToolCallRecord(tool_name=r.tool_name, summary=_summarize(r.data)) for r in ranked if r.data is not None
        ]

        return PipelineResult(
            reply=reply,
            pipeline_mode="agentic",
            retrieval_sources=sources,
            references=[r.data for r in ranked if r.data is not None],
            trace={
                "mode": interp.execution_mode,
                "plan": dataclasses.asdict(plan),
                "ranked": [dataclasses.asdict(r) for r in ranked],
                "validation_warning": ctx.state.get("validation_warning", False),
            },
        )


def _p(stage: Stage, role: str) -> LLMProvider:
    """Resolve the provider for a stage: prefer its own, else mock."""
    return getattr(stage, "llm", None) or MockProvider()


def _summarize(data: Any) -> str:
    if isinstance(data, dict):
        return (data.get("summary") or data.get("name") or str(data))[:120]
    return str(data)[:120]
