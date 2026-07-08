from agent import InterpretedQuery, NoOpMemory
from agent.llm.mock import MockProvider
from agent.stages.base import StageContext
from agent.stages.interpreter import InterpreterStage


class ValidInterpreter(MockProvider):
    def __init__(self):
        super().__init__(self._respond)

    def _respond(self, messages, system_prompt, json_mode):
        return {
            "resolved_query": "resolved: " + (messages[-1]["content"] if messages else ""),
            "execution_mode": "full",
            "intent": "test",
            "is_followup": False,
            "memory_hint": "",
            "fetch_tools": [],
        }


class InvalidJsonInterpreter(MockProvider):
    def complete_json(self, messages, system_prompt=None):
        raise ValueError("Could not parse JSON from response: bad")


def test_interpreter_valid_response():
    stage = InterpreterStage(ValidInterpreter())
    ctx = StageContext(memory=NoOpMemory())
    result = stage.run(ctx, query="hello")
    assert isinstance(result, InterpretedQuery)
    assert result.execution_mode == "full"
    assert result.intent == "test"
    assert result.is_followup is False
    assert "hello" in result.resolved_query


def test_interpreter_malformed_json():
    stage = InterpreterStage(InvalidJsonInterpreter())
    ctx = StageContext(memory=NoOpMemory())
    import pytest
    with pytest.raises(ValueError):
        stage.run(ctx, query="hello")


def test_interpreter_empty_query():
    stage = InterpreterStage(ValidInterpreter())
    ctx = StageContext(memory=NoOpMemory())
    result = stage.run(ctx, query="")
    assert isinstance(result, InterpretedQuery)
