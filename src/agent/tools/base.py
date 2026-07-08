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

import threading
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


class ToolRegistry:
    """Per-instance tool registry with a default global instance for backward compat."""

    _default: "ToolRegistry | None" = None
    _lock = threading.Lock()

    @classmethod
    def _get_default(cls) -> "ToolRegistry":
        if cls._default is None:
            cls._default = ToolRegistry()
        return cls._default

    def __init__(self):
        self._tools: dict[str, dict] = {}
        self._lock = threading.Lock()

    def register(self, name, description, parameters, handler):
        with self._lock:
            self._tools[name] = {
                "name": name,
                "description": description,
                "parameters": parameters,
                "handler": handler,
            }

    def get_definitions(self) -> list[dict]:
        with self._lock:
            return [
                {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["parameters"],
                }
                for t in self._tools.values()
            ]

    def clear(self):
        with self._lock:
            self._tools.clear()

    def names(self) -> list[str]:
        with self._lock:
            return list(self._tools.keys())

    def get_spec(self, name: str) -> dict | None:
        with self._lock:
            return self._tools.get(name)


_default_registry = ToolRegistry._get_default()


def get_default_registry() -> ToolRegistry:
    return _default_registry


def tool(
    name: str,
    description: str = "",
    parameters: dict | None = None,
    registry: ToolRegistry | None = None,
):
    """Decorator to register a tool handler.

    The handler receives `**args` from the planner and may accept a
    `ctx: ToolContext` keyword for identity/tenant injection.

    Example
    -----
        @tool("get_weather", description="Current weather", parameters={...})
        def get_weather(ctx: ToolContext, city: str) -> dict:
            return {...}
    """
    reg = registry or _default_registry

    def decorator(fn: Callable[..., Any]):
        reg.register(
            name=name,
            description=description,
            parameters=parameters or {"type": "object", "properties": {}},
            handler=fn,
        )
        return fn

    return decorator


def get_tool_definitions(registry: ToolRegistry | None = None) -> list[dict]:
    return (registry or _default_registry).get_definitions()


class ToolExecutor:
    """Executes planned tool calls deterministically."""

    def __init__(self, ctx: ToolContext, registry: ToolRegistry | None = None):
        self.ctx = ctx
        self._registry = registry or _default_registry

    def execute(self, name: str, args: dict) -> Any:
        spec = self._registry.get_spec(name)
        if not spec:
            from ..errors import ToolNotFoundError

            raise ToolNotFoundError(f"Unknown tool: '{name}'")

        handler = spec["handler"]
        import inspect

        sig = inspect.signature(handler)
        if "ctx" in sig.parameters:
            return handler(ctx=self.ctx, **(args or {}))
        return handler(**(args or {}))


class ToolNotFoundError(Exception):
    pass