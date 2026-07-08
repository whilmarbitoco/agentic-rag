"""joy-agent: the 6-stage agentic RAG pipeline (JOY) from the KLIMA thesis.

Stages:
  0. Interpreter  - resolve references, pick execution mode
  1. Planner      - select tools + rewrite query
  2. Executor     - deterministic parallel tool calls (no LLM)
  3. Reranker     - score/rank retrieved data
  4. Synthesizer  - generate the final answer
  5. Validator    - self-check + regeneration (feedback loop)
  +  Memory       - background fetch/store for continuity

Extension points:
  - Register tools with `@tool(...)`.
  - Subclass / swap any stage in `AgenticPipeline`.
  - Plug any OpenAI-compatible LLM via `OpenAICompatProvider`,
    or implement `LLMProvider` for custom backends.
"""

from .context import (
    BudgetContextManager,
    Compactor,
    ContextManager,
    HeuristicCompactor,
    LLMCompactor,
    TiktokenTokenCounter,
    TokenCounter,
    WordTokenCounter,
)
from .llm.base import LLMProvider, extract_json
from .llm.factory import ProviderFactory, make_mock_factory, make_openai_factory
from .llm.mock import MockProvider
from .llm.openai_compat import OpenAICompatProvider
from .memory.base import InMemoryMemory, MemoryModule, NoOpMemory
from .pipeline import AgenticPipeline
from .stages import (
    ExecutorStage,
    InterpreterStage,
    PlannerStage,
    RerankerStage,
    SynthesizerStage,
    ValidatorStage,
)
from .stages.base import Stage, StageContext
from .tools.base import ToolContext, ToolExecutor, ToolNotFoundError, get_default_registry, get_tool_definitions, tool
from .types import (
    InterpretedQuery,
    PipelineResult,
    RankedResult,
    RetrievalPlan,
    ToolCallPlan,
    ToolCallRecord,
    ToolResult,
    ValidationVerdict,
)

__all__ = [
    "InterpretedQuery",
    "RetrievalPlan",
    "ToolCallPlan",
    "ToolResult",
    "RankedResult",
    "ToolCallRecord",
    "ValidationVerdict",
    "PipelineResult",
    "tool",
    "ToolContext",
    "ToolExecutor",
    "ToolNotFoundError",
    "get_tool_definitions",
    "get_default_registry",
    "LLMProvider",
    "extract_json",
    "OpenAICompatProvider",
    "MockProvider",
    "ProviderFactory",
    "make_mock_factory",
    "make_openai_factory",
    "MemoryModule",
    "NoOpMemory",
    "InMemoryMemory",
    "ContextManager",
    "BudgetContextManager",
    "TokenCounter",
    "WordTokenCounter",
    "TiktokenTokenCounter",
    "Compactor",
    "HeuristicCompactor",
    "LLMCompactor",
    "Stage",
    "StageContext",
    "InterpreterStage",
    "PlannerStage",
    "ExecutorStage",
    "RerankerStage",
    "SynthesizerStage",
    "ValidatorStage",
    "AgenticPipeline",
]
