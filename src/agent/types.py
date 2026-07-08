"""Shared dataclasses for the agentic pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class InterpretedQuery:
    original_query: str
    resolved_query: str
    execution_mode: str  # "fetch_only" | "full" | "direct"
    intent: str
    is_followup: bool
    memory_hint: str
    fetch_tools: list[str] = field(default_factory=list)


@dataclass
class ToolCallPlan:
    name: str
    args: dict = field(default_factory=dict)


@dataclass
class RetrievalPlan:
    in_domain: bool
    rewritten_query: str
    tools: list[ToolCallPlan]
    reasoning: str


@dataclass
class ToolResult:
    tool_name: str
    data: Any = None
    error: str | None = None


@dataclass
class RankedResult:
    tool_name: str
    data: Any
    score: int
    reason: str


@dataclass
class ToolCallRecord:
    tool_name: str
    summary: str


@dataclass
class ValidationVerdict:
    valid: bool = True
    critique: str = ""


@dataclass
class PipelineResult:
    reply: str
    pipeline_mode: str  # "agentic" | "direct" | "fetch_only" | "out_of_domain"
    retrieval_sources: list[ToolCallRecord] = field(default_factory=list)
    references: list[Any] = field(default_factory=list)
    trace: dict = field(default_factory=dict)
