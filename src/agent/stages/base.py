"""Stage base interface.

Every pipeline stage is a `Stage`. The orchestrator calls `run(ctx, *inputs)`
on each stage in order. This lets you:

  - swap any single stage (subclass + inject),
  - insert a new stage (e.g. a safety pre-filter) between existing ones,
  - run stages remotely / behind a queue, by changing only `run`.

A stage receives a `StageContext` (pipeline state + the registered providers)
and returns whatever the next stage expects. The orchestrator (`pipeline.py`)
wires the handoffs; stages stay decoupled.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from ..llm.base import LLMProvider
from ..memory.base import MemoryModule
from ..tools.base import ToolContext, ToolExecutor


@dataclass
class StageContext:
    """Everything a stage needs at runtime."""

    llm: dict[str, LLMProvider] = field(default_factory=dict)  # role -> provider
    memory: MemoryModule = field(default_factory=MemoryModule)
    tool_ctx: ToolContext = field(default_factory=ToolContext)
    state: dict[str, Any] = field(default_factory=dict)  # scratch space across stages


class Stage(ABC):
    name: str = "stage"

    @abstractmethod
    def run(self, ctx: StageContext) -> Any:
        ...
