from agent import ValidationVerdict, InterpretedQuery, RankedResult, NoOpMemory
from agent.stages.validator import ValidatorStage
from agent.stages.base import StageContext
from agent.llm.mock import MockProvider


class AlwaysValid(MockProvider):
    def complete_json(self, messages, system_prompt=None):
        return {"valid": True, "critique": ""}


class AlwaysInvalid(MockProvider):
    def complete_json(self, messages, system_prompt=None):
        return {"valid": False, "critique": "too vague"}


def _ctx(provider=None):
    return StageContext(
        llm={"validator": provider or AlwaysValid()},
        memory=NoOpMemory(),
    )


def _interpreted():
    return InterpretedQuery("q", "q", "full", "test", False, "")


def test_validator_passes():
    stage = ValidatorStage(AlwaysValid())
    result = stage.run(_ctx(), _interpreted(), [], "valid answer")
    assert isinstance(result, ValidationVerdict)
    assert result.valid is True


def test_validator_fails_llm():
    stage = ValidatorStage(AlwaysInvalid())
    result = stage.run(_ctx(), _interpreted(), [], "some answer")
    assert result.valid is False
    assert result.critique == "too vague"


def test_validator_empty_answer():
    stage = ValidatorStage(AlwaysValid())
    result = stage.run(_ctx(), _interpreted(), [], "")
    assert result.valid is False
    assert "empty" in result.critique.lower()


def test_validator_short_answer():
    stage = ValidatorStage(AlwaysValid())
    result = stage.run(_ctx(), _interpreted(), [], "hi")
    assert result.valid is False
    assert "short" in result.critique.lower()


def test_validator_no_access_trigger_with_data():
    stage = ValidatorStage(AlwaysValid())
    ranked = [RankedResult("t1", {"val": 1}, 1, "test")]
    result = stage.run(_ctx(), _interpreted(), ranked, "I don't have access to your data")
    assert result.valid is False
    assert "no data access" in result.critique.lower()


def test_validator_no_access_trigger_no_data():
    stage = ValidatorStage(AlwaysValid())
    result = stage.run(_ctx(), _interpreted(), [], "I don't have access to your data")
    assert result.valid is True  # no data returned, so no contradiction


def test_validator_provider_fallback():
    stage = ValidatorStage()
    ctx = StageContext(
        llm={"validator": None},
        memory=NoOpMemory(),
    )
    result = stage.run(ctx, _interpreted(), [], "some answer")
    assert isinstance(result, ValidationVerdict)
    assert result.valid is True