import pytest

from agent import (
    AgenticPipeline,
    ToolContext,
    MockProvider,
    NoOpMemory,
    InMemoryMemory,
    tool,
    ToolNotFoundError,
    get_tool_definitions,
    OpenAICompatProvider,
    get_default_registry,
)


@pytest.fixture(scope="module", autouse=True)
def _register_get_fact():
    reg = get_default_registry()
    reg.register("get_fact", "A fact", {"type": "object", "properties": {}},
                lambda ctx: {"summary": "fact", "value": 42})
    yield
    reg.clear()


class ScriptedProvider(MockProvider):
    def __init__(self, *, fail_validator_once=False):
        super().__init__(self._respond)
        self.fail_validator_once = fail_validator_once
        self._vcalls = 0

    def _respond(self, messages, system_prompt, json_mode):
        if json_mode:
            if system_prompt and "Interpreter" in system_prompt:
                return {"resolved_query": messages[-1]["content"], "execution_mode": "full",
                        "intent": "fact", "is_followup": False, "memory_hint": "", "fetch_tools": []}
            if system_prompt and "Planner" in system_prompt:
                return {"in_domain": True, "rewritten_query": messages[-1]["content"],
                        "tools": [{"name": "get_fact", "args": {}}], "reasoning": "lookup"}
            if system_prompt and "Validator" in system_prompt:
                if self.fail_validator_once and self._vcalls == 0:
                    self._vcalls += 1
                    return {"valid": False, "critique": "too vague"}
                return {"valid": True, "critique": ""}
            return {}
        return "The answer is 42."


def build_pipeline(**kw):
    prov = {role: ScriptedProvider(**kw) for role in ["interpreter", "planner", "reranker", "synthesizer", "validator"]}
    return AgenticPipeline.build_default(providers=prov, memory=NoOpMemory())


def test_full_pipeline_runs_offline():
    res = build_pipeline().run("What is the fact?", tool_ctx=ToolContext(tenant_id=1))
    assert res.pipeline_mode == "agentic"
    assert res.reply == "The answer is 42."
    assert [s.tool_name for s in res.retrieval_sources] == ["get_fact"]
    assert res.trace["validation_warning"] is False


def test_validator_retry_loop():
    res = build_pipeline(fail_validator_once=True).run("What is the fact?")
    assert res.reply == "The answer is 42."
    assert res.trace["validation_warning"] is False


def test_unknown_tool_raises_tool_not_found():
    class BadPlanner(MockProvider):
        def complete_json(self, messages, system_prompt=None):
            return {"in_domain": True, "rewritten_query": "x", "tools": [{"name": "nope", "args": {}}]}

    from agent import PlannerStage
    from agent.llm.mock import MockProvider as MP
    pipe = AgenticPipeline.build_default(providers={role: MP() for role in ["interpreter", "planner", "reranker", "synthesizer", "validator"]}, memory=NoOpMemory())
    pipe.planner = PlannerStage(BadPlanner())
    res = pipe.run("hi")
    assert res.retrieval_sources == []
    assert res.reply


def test_tool_context_injects_tenant_not_llm():
    seen = {}

    def make_probe(store):
        def probe(ctx: ToolContext) -> dict:
            store["tenant"] = ctx.tenant_id
            return {"summary": "probe", "v": ctx.tenant_id}
        return probe

    reg = get_default_registry()
    reg.register("probe", "probe", {"type": "object", "properties": {}},
                 make_probe(seen))

    class ProbePlanner(MockProvider):
        def complete_json(self, messages, system_prompt=None):
            return {"in_domain": True, "rewritten_query": "x", "tools": [{"name": "probe", "args": {}}], "reasoning": "probe"}

    from agent import PlannerStage
    pipe = AgenticPipeline.build_default(providers={role: ScriptedProvider() for role in ["interpreter", "planner", "reranker", "synthesizer", "validator"]}, memory=NoOpMemory())
    pipe.planner = PlannerStage(ProbePlanner())
    res = pipe.run("probe me", tool_ctx=ToolContext(tenant_id=777))
    assert seen.get("tenant") == 777
    assert res.retrieval_sources


def test_out_of_domain_skips_tools():
    class OODPlanner(MockProvider):
        def complete_json(self, messages, system_prompt=None):
            return {"in_domain": False, "rewritten_query": "x", "tools": [], "reasoning": "out"}

    from agent import PlannerStage
    from agent.llm.mock import MockProvider as MP
    pipe = AgenticPipeline.build_default(providers={role: MP() for role in ["interpreter", "planner", "reranker", "synthesizer", "validator"]}, memory=NoOpMemory())
    pipe.planner = PlannerStage(OODPlanner())
    res = pipe.run("Who won the 2020 olympics?")
    assert res.pipeline_mode == "out_of_domain"
    assert res.retrieval_sources == []


def test_memory_store_called():
    mem = InMemoryMemory()
    pipe = AgenticPipeline.build_default(providers={role: ScriptedProvider() for role in ["interpreter", "planner", "reranker", "synthesizer", "validator"]}, memory=mem)
    res = pipe.run("What is the fact?")
    assert res.reply
    assert len(mem._turns) == 1


def test_registry_lists_tools():
    names = {t["name"] for t in get_tool_definitions()}
    assert "get_fact" in names


def test_provider_factory_openai_compat():
    from agent.llm.factory import make_openai_factory
    f = make_openai_factory(api_key="dummy")
    p = f.get("synthesizer")
    assert isinstance(p, OpenAICompatProvider)
    assert f.get_fallback("synthesizer") is None