"""Context-window management & compaction (extendable-first).

Three extension points, mirroring the framework's swap-don't-fork philosophy:

  - TokenCounter    : how tokens are counted (WordTokenCounter default;
                      TiktokenTokenCounter optional for exact model counts).
  - Compactor       : how oversized tool results are shrunk to a budget
                      (HeuristicCompactor default; LLMCompactor optional).
  - ContextManager  : allocates a token budget across sections (memory /
                      tools / history) and compacts the synthesizer sources.

The orchestrator creates one ContextManager per run and threads it through
StageContext, so token usage is observable in `PipelineResult.trace["context"]`.
"""
from .compactor import Compactor, HeuristicCompactor, LLMCompactor
from .manager import BudgetContextManager, ContextManager
from .tokens import TiktokenTokenCounter, TokenCounter, WordTokenCounter

__all__ = [
    "TokenCounter",
    "WordTokenCounter",
    "TiktokenTokenCounter",
    "Compactor",
    "HeuristicCompactor",
    "LLMCompactor",
    "ContextManager",
    "BudgetContextManager",
]
