"""Shared fixtures and test helpers."""
from __future__ import annotations

import pytest

from agent import MockProvider, NoOpMemory, ToolContext, get_default_registry


@pytest.fixture(scope="module", autouse=True)
def module_clear_registry():
    get_default_registry().clear()
    yield





@pytest.fixture
def tool_ctx():
    return ToolContext(tenant_id=1)


class ScriptedProvider(MockProvider):
    def __init__(self, *, fail_validator_once=False, tools=None):
        super().__init__(self._respond)
        self.fail_validator_once = fail_validator_once
        self._vcalls = 0
        self._tools = tools or [{"name": "get_fact", "args": {}}]

    def _respond(self, messages, system_prompt, json_mode):
        if json_mode:
            text = (messages[-1]["content"] if messages else "").lower()
            if system_prompt and "Interpreter" in system_prompt:
                return {
                    "resolved_query": messages[-1]["content"],
                    "execution_mode": "full",
                    "intent": "general",
                    "is_followup": False,
                    "memory_hint": "",
                    "fetch_tools": [],
                }
            if system_prompt and "Planner" in system_prompt:
                return {
                    "in_domain": True,
                    "rewritten_query": messages[-1]["content"],
                    "tools": self._tools,
                    "reasoning": "mock lookup",
                }
            if system_prompt and "Validator" in system_prompt:
                if self.fail_validator_once and self._vcalls == 0:
                    self._vcalls += 1
                    return {"valid": False, "critique": "too vague"}
                return {"valid": True, "critique": ""}
            return {}
        return "Mock answer for: " + (messages[-1]["content"] if messages else "?")[:60]


@pytest.fixture
def scripted_providers():
    return {role: ScriptedProvider() for role in ["interpreter", "planner", "reranker", "synthesizer", "validator"]}