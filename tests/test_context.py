from agent import (
    AgenticPipeline,
    BudgetContextManager,
    MockProvider,
    NoOpMemory,
    ToolContext,
    get_default_registry,
)


def setup_module(module):
    reg = get_default_registry()
    reg.clear()
    reg.register("big_doc", "A large document", {"type": "object", "properties": {}},
               lambda ctx: {"summary": "doc", "body": "x" * 4000})


class ScriptedProvider(MockProvider):
    def __init__(self):
        super().__init__(self._respond)

    def _respond(self, messages, system_prompt, json_mode):
        if json_mode:
            if system_prompt and "Interpreter" in system_prompt:
                return {"resolved_query": messages[-1]["content"], "execution_mode": "full",
                        "intent": "doc", "is_followup": False, "memory_hint": "", "fetch_tools": []}
            if system_prompt and "Planner" in system_prompt:
                return {"in_domain": True, "rewritten_query": messages[-1]["content"],
                        "tools": [{"name": "big_doc", "args": {}}], "reasoning": "doc lookup"}
            if system_prompt and "Validator" in system_prompt:
                return {"valid": True, "critique": ""}
            return {}
        return "Answer based on the doc."


def build(tools_budget):
    ctx_mgr = BudgetContextManager(budgets={"tools": tools_budget, "memory": 1500, "history": 1500, "total": 6000})
    prov = {role: ScriptedProvider() for role in ["interpreter", "planner", "reranker", "synthesizer", "validator"]}
    return AgenticPipeline.build_default(providers=prov, memory=NoOpMemory()), ctx_mgr


def test_compaction_keeps_within_budget():
    pipe, ctx_mgr = build(tools_budget=200)
    pipe.context = ctx_mgr
    res = pipe.run("summarize the doc", tool_ctx=ToolContext(tenant_id=1))
    usage = res.trace["context"]
    # Heuristic counter is approximate; assert it stays near the budget and far
    # below the uncompacted payload (~1000 tokens for a 4KB doc).
    assert usage["tools"] <= 230, f"tools usage {usage['tools']} exceeded budget"
    assert usage["tools"] < 800  # proof compaction actually fired
    assert res.reply


def test_no_compaction_when_budget_large():
    pipe, ctx_mgr = build(tools_budget=5000)
    pipe.context = ctx_mgr
    res = pipe.run("summarize the doc", tool_ctx=ToolContext(tenant_id=1))
    usage = res.trace["context"]
    # The full (truncated-to-2000 by _as_text) payload is ~500 tokens; budget ample.
    assert usage["tools"] <= 5000
    assert usage["tools"] > 200  # proves without compaction it would be larger


def test_trace_reports_per_section_usage():
    pipe, ctx_mgr = build(tools_budget=300)
    pipe.context = ctx_mgr
    res = pipe.run("summarize the doc", tool_ctx=ToolContext(tenant_id=1))
    report = res.trace["context"]
    assert set(report) >= {"memory", "tools", "total"}
    assert isinstance(report["tools"], int)
