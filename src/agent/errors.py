"""Pipeline error types."""


class PipelineError(Exception):
    """Base class for pipeline errors."""


class ToolNotFoundError(PipelineError):
    """Raised when a planner requests a tool that is not registered."""


class LLMProviderError(Exception):
    """Raised when an LLM provider call fails."""


class ProviderRateLimitError(LLMProviderError):
    """Raised on 429 — the factory uses this to trigger fallback."""
