import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentic_rag import AgenticPipeline, ToolContext, MockProvider, NoOpMemory, tool


@tool("get_weather", description="Current weather", parameters={"type": "object", "properties": {"city": {"type": "string"}}})
def get_weather(ctx: ToolContext, city: str) -> dict:
    return {"summary": f"weather {city}", "temp_c": 31, "condition": "partly cloudy"}


class ScriptedProvider(MockProvider):
    def __init__(self):
        super().__init__(self._respond)

    def _respond(self, messages, system_prompt, json_mode):
        if json_mode:
            if system_prompt and "Interpreter" in system_prompt:
                return {"resolved_query": messages[-1]["content"], "execution_mode": "full",
                        "intent": "weather", "is_followup": False, "memory_hint": "", "fetch_tools": []}
            if system_prompt and "Planner" in system_prompt:
                return {"in_domain": True, "rewritten_query": messages[-1]["content"],
                        "tools": [{"name": "get_weather", "args": {"city": "Manila"}}], "reasoning": "weather lookup"}
            if system_prompt and "Validator" in system_prompt:
                return {"valid": True, "critique": ""}
            return {}
        return "Based on [get_weather], Manila is 31C and partly cloudy."


def main():
    providers = {role: ScriptedProvider() for role in ["interpreter", "planner", "reranker", "synthesizer", "validator"]}
    pipeline = AgenticPipeline.build_default(providers=providers, memory=NoOpMemory())
    result = pipeline.run("What's the weather in Manila?", tool_ctx=ToolContext(tenant_id=1))
    print("MODE:", result.pipeline_mode)
    print("REPLY:", result.reply)
    print("SOURCES:", [s.tool_name for s in result.retrieval_sources])


if __name__ == "__main__":
    main()
