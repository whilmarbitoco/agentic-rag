"""End-to-end example: a tiny "document Q&A" assistant.

Runs OFFLINE with a scripted provider so you can see the full 6-stage flow
without an API key. To use a real model, swap `ScriptedProvider` for
`OpenAICompatProvider(base_url=..., api_key=..., model=...)` — nothing else
changes. That is the whole point of the extendable design.
"""
from __future__ import annotations

from agent import (
    AgenticPipeline,
    MockProvider,
    NoOpMemory,
    ToolContext,
    tool,
)


# 1) Define your tools. Identity/tenant come from ToolContext, never the LLM.
@tool("get_company_faq", description="Company FAQ entries", parameters={"type": "object", "properties": {}})
def get_company_faq(ctx: ToolContext) -> dict:
    return {
        "summary": "FAQ",
        "items": [
            {"q": "What is your refund policy?", "a": "30-day full refund, no questions asked."},
            {"q": "Do you ship internationally?", "a": "Yes, to 60+ countries via express courier."},
        ],
    }


@tool("get_order_status", description="Order status by id", parameters={"type": "object", "properties": {"order_id": {"type": "string"}}})
def get_order_status(ctx: ToolContext, order_id: str) -> dict:
    # In production this would hit your DB with ctx.tenant_id enforced.
    return {"summary": f"order {order_id}", "status": "shipped", "eta": "2026-07-12"}


# 2) A scripted provider so this demo runs with zero network calls.
#    Replace with OpenAICompatProvider(...) for a real model.
class ScriptedProvider(MockProvider):
    def __init__(self):
        super().__init__(self._respond)

    def _respond(self, messages, system_prompt, json_mode):
        if json_mode:
            if system_prompt and "Interpreter" in system_prompt:
                return {"resolved_query": messages[-1]["content"], "execution_mode": "full",
                        "intent": "faq", "is_followup": False, "memory_hint": "", "fetch_tools": []}
            if system_prompt and "Planner" in system_prompt:
                return {"in_domain": True, "rewritten_query": messages[-1]["content"],
                        "tools": [{"name": "get_company_faq", "args": {}}], "reasoning": "faq lookup"}
            if system_prompt and "Validator" in system_prompt:
                return {"valid": True, "critique": ""}
            return {}
        return ("Based on [get_company_faq], our refund policy is a 30-day full refund "
                "with no questions asked, and we ship internationally to 60+ countries.")


def main():
    providers = {
        "interpreter": ScriptedProvider(),
        "planner": ScriptedProvider(),
        "reranker": ScriptedProvider(),
        "synthesizer": ScriptedProvider(),
        "validator": ScriptedProvider(),
    }
    pipeline = AgenticPipeline.build_default(providers=providers, memory=NoOpMemory())

    result = pipeline.run(
        "What is your refund policy and do you ship abroad?",
        tool_ctx=ToolContext(tenant_id=1),
    )

    print("MODE:", result.pipeline_mode)
    print("REPLY:", result.reply)
    print("SOURCES:", [s.tool_name for s in result.retrieval_sources])
    print("TRACE KEYS:", list(result.trace.keys()))


if __name__ == "__main__":
    main()
