"""agentic-rag: a domain-agnostic 6-stage agentic RAG pipeline.

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

from .types import (
    InterpretedQuery,
    RetrievalPlan,
    ToolCallPlan,
    ToolResult,
    RankedResult,
    ToolCallRecord,
    ValidationVerdict,
    PipelineResult,
)
from .tools.base import tool, ToolContext, ToolExecutor, ToolNotFoundError, get_tool_definitions, get_default_registry
from .llm.base import LLMProvider, extract_json
from .llm.openai_compat import OpenAICompatProvider
from .llm.mock import MockProvider
from .llm.factory import ProviderFactory, make_mock_factory, make_openai_factory
from .memory.base import MemoryModule, NoOpMemory, InMemoryMemory
from .context import (
    ContextManager,
    BudgetContextManager,
    TokenCounter,
    WordTokenCounter,
    TiktokenTokenCounter,
    Compactor,
    HeuristicCompactor,
    LLMCompactor,
)
from .stages.base import Stage, StageContext
from .stages import (
    InterpreterStage,
    PlannerStage,
    ExecutorStage,
    RerankerStage,
    SynthesizerStage,
    ValidatorStage,
)
from .pipeline import AgenticPipeline

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
