from agent import (
    InterpretedQuery, RetrievalPlan, RankedResult, NoOpMemory,
    BudgetContextManager,
)
from agent.stages.synthesizer import SynthesizerStage
from agent.stages.base import StageContext
from agent.llm.mock import MockProvider


class SynthProvider(MockProvider):
    def complete(self, messages, system_prompt=None):
        return "Synthesized answer based on data."


def make_ctx(provider=None):
    return StageContext(
        llm={"synthesizer": provider or SynthProvider()},
        memory=NoOpMemory(),
        state={},
        context=BudgetContextManager(),
    )


def _plan():
    return RetrievalPlan(in_domain=True, rewritten_query="test query", tools=[], reasoning="test")


def _interp():
    return InterpretedQuery("q", "test query", "full", "test", False, [])


def test_synthesizer_with_data():
    stage = SynthesizerStage(SynthProvider())
    ranked = [RankedResult("t1", {"summary": "some data", "value": 42}, 1, "test")]
    result = stage.run(make_ctx(SynthProvider()), interpreted=_interp(), plan=_plan(), ranked=ranked)
    assert isinstance(result, str)
    assert "Synthesized" in result


def test_synthesizer_no_data():
    stage = SynthesizerStage(SynthProvider())
    result = stage.run(make_ctx(SynthProvider()), interpreted=_interp(), plan=_plan(), ranked=[])
    assert isinstance(result, str)


def test_synthesizer_with_critique():
    stage = SynthesizerStage(SynthProvider())
    ctx = StageContext(
        llm={"synthesizer": SynthProvider()},
        memory=NoOpMemory(),
        state={"validator_critique": "too vague, include specifics"},
        context=BudgetContextManager(),
    )
    ranked = [RankedResult("t1", {"text": "data"}, 1, "test")]
    result = stage.run(ctx, interpreted=_interp(), plan=_plan(), ranked=ranked)
    assert isinstance(result, str)


def test_synthesizer_multiple_sources():
    stage = SynthesizerStage(SynthProvider())
    ranked = [
        RankedResult("t1", {"summary": "first"}, 2, "relevant"),
        RankedResult("t2", {"summary": "second"}, 1, "somewhat"),
    ]
    result = stage.run(make_ctx(SynthProvider()), interpreted=_interp(), plan=_plan(), ranked=ranked)
    assert isinstance(result, str)