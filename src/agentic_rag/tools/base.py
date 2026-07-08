"""Tool registry and executor.

Tools are registered with `@tool(name=..., description=..., parameters=...)` and
are invoked deterministically in parallel during Stage 2. No LLM is involved in
execution itself — this keeps tool calls cheap, fast, and hallucination-free.

Security contract:
  - `farm_id`/tenant/identity is injected via ToolContext, never from the LLM args.
  - All tools are expected to be read-only (enforced by convention; document it).
  - Unknown tool names raise ToolNotFoundError (mapped to 400 upstream).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass
class ToolContext:
    """Ambient context injected into every tool call by the pipeline.

    The planner/LLM never sees or controls these values.
    """

    user_id: Optional[int] = None
    tenant_id: Optional[int] = None
    metadata: dict = field(default_factory=dict)


def tool(
    name: str,
    description: str = "",
    parameters: dict | None = None,
):
    """Decorator to register a tool handler.

    The handler receives `**args` from the planner and may accept a
    `ctx: ToolContext` keyword for identity/tenant injection.

    Example
    -------
        @tool("get_weather", description="Current weather", parameters={...})
        def get_weather(ctx: ToolContext, city: str) -> dict:
            return {...}
    """

    def decorator(fn: Callable[..., Any]):
        ToolRegistry.register(
            name=name,
            description=description,
            parameters=parameters or {"type": "object", "properties": {}},
            handler=fn,
        )
        return fn

    return decorator


class ToolRegistry:
    _tools: dict[str, dict] = {}

    @classmethod
    def register(cls, name, description, parameters, handler):
        cls._tools[name] = {
            "name": name,
            "description": description,
            "parameters": parameters,
            "handler": handler,
        }

    @classmethod
    def get_definitions(cls) -> list[dict]:
        return [
            {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["parameters"],
            }
            for t in cls._tools.values()
        ]

    @classmethod
    def clear(cls):
        cls._tools.clear()

    @classmethod
    def names(cls) -> list[str]:
        return list(cls._tools.keys())


def get_tool_definitions() -> list[dict]:
    return ToolRegistry.get_definitions()


class ToolExecutor:
    """Executes planned tool calls deterministically."""

    def __init__(self, ctx: ToolContext):
        self.ctx = ctx

    def execute(self, name: str, args: dict) -> Any:
        spec = ToolRegistry._tools.get(name)
        if not spec:
            from ..errors import ToolNotFoundError

            raise ToolNotFoundError(f"Unknown tool: '{name}'")

        handler = spec["handler"]
        # Inject ctx only if the handler accepts it by keyword.
        import inspect

        sig = inspect.signature(handler)
        if "ctx" in sig.parameters:
            return handler(ctx=self.ctx, **(args or {}))
        return handler(**(args or {}))


class ToolNotFoundError(Exception):  # surfaced for import convenience
    pass
