# agentic-rag

**The AI agent architecture from my undergraduate thesis, open-sourced and made easy to extend and reuse.**

> Generalized reference implementation of the agentic RAG pipeline presented in
> *"KLIMA: A Microclimate Monitoring and Decision-Support System for Rice Farming
> Using IoT and Time-Series Learning in Tagum City"* (Bitoco, 2026). The original
> system — codenamed **JOY** — powered a farmer-facing climate advisor. This repo
> extracts that architecture into a **domain-agnostic, extensible template** you
> can drop your own tools, prompts, and LLM providers into.

[![CI](https://img.shields.io/badge/tests-passing-green)]()
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Why this exists

Most agent frameworks either (a) hide the control flow or (b) require an LLM to
call tools. This architecture was built for **production reliability**, not demo
flash:

- **Deterministic tool execution.** Tools are plain Python functions called in
  parallel — never via the LLM. No hallucinated tool names, no per-call model cost.
- **Six explicit, swappable stages.** Each stage is a self-contained unit with one
  job and a clean interface.
- **Self-correcting validator.** A feedback loop catches ungrounded answers and
  regenerates before the user sees them.
- **Built-in graceful degradation.** Any stage can fail and the pipeline still
  returns a safe answer.
- **Trace-based evaluation baked in.** Every run produces a `trace` you can score
  stage-by-stage.

## The architecture

```
        ┌─────────────┐
 user →  │ Interpreter │  resolve refs, pick mode (fetch_only / direct / full), pull memory
        └──────┬──────┘
               ▼
        ┌──────────┐
        │  Planner │  select tools + rewrite query (in-domain guard)
        └──────┬───┘
               ▼
        ┌──────────┐
        │ Executor │  deterministic, parallel tool calls (NO LLM)
        └──────┬───┘
               ▼
        ┌──────────┐
        │ Reranker │  score + rank retrieved data
        └──────┬───┘
               ▼
        ┌────────────┐      ┌──────────┐
        │ Synthesizer │◄─────┤ Validator │  self-check; regenerate on failure (loop)
        └────────────┘      └──────────┘
               │
               ▼
            answer + trace
   (Memory module feeds Interpreter + stores turns in the background)
```

| Stage | Role | Default |
|---|---|---|
| Interpreter | dereference, mode-routing, memory fetch | LLM JSON |
| Planner | tool selection + query rewrite | LLM JSON |
| Executor | parallel tool calls (deterministic) | code |
| Reranker | relevance scoring | heuristic (swap for reranker model) |
| Synthesizer | grounded answer generation | LLM |
| Validator | hallucination / grounding guard | lightweight + optional LLM judge |
| Memory | conversation continuity | `InMemoryMemory` / pluggable |

## Extendable-first design

The whole philosophy is **swap, don't fork**. Four extension points:

1. **`LLMProvider`** — implement `complete()` / `complete_json()`. Ships with
   `OpenAICompatProvider` (works with OpenAI, Groq, NVIDIA, OpenRouter, Ollama,
   vLLM…) and `MockProvider` for offline/dev.
2. **`MemoryModule`** — implement `fetch_context()` / `store()`. Ships with
   `InMemoryMemory` and `NoOpMemory`. Plug pgvector / FTS / Redis by subclassing.
3. **`Stage`** — subclass to override or insert any stage (e.g. a safety pre-filter).
4. **`@tool`** — register deterministic tools; identity/tenant injected via
   `ToolContext`, never from the LLM.

```python
from agent import AgenticPipeline, ToolContext, tool

@tool("get_weather", description="Current weather", parameters={...})
def get_weather(ctx: ToolContext, city: str):
    return {"temp_c": 31}

# Real model: swap the mock for your provider. Nothing else changes.
providers = {"interpreter": OpenAICompatProvider(base_url=..., api_key=..., model=...), ...}
pipe = AgenticPipeline.build_default(providers=providers)
result = pipe.run("What's the weather in Manila?", tool_ctx=ToolContext(tenant_id=1))
print(result.reply, result.trace)
```

See [`examples/`](examples) for a runnable offline demo (no API key needed).

## Install

```bash
pip install -e .
pytest                 # runs fully offline
python examples/doc_qa.py
```

## Provenance & honesty

This is a **faithful generalization** of the JOY pipeline from the KLIMA thesis.
The thesis contributes the *empirical evaluation* (trace-based, per-stage scoring:
Monitoring 0.94 / Knowledge 0.93 / Spatial 0.82); this repo contributes the
*reusable architecture*. The original system was domain-specific (rice farming);
this template is domain-agnostic.

**Design tradeoff (stated plainly):** the pipeline is a **linear chain**, not a
re-planning graph. The validator catches *synthesis* errors but cannot re-route a
wrong tool choice back to planning. That was a deliberate simplicity decision in
the thesis; if you need autonomous re-planning, subclass `PlannerStage` +
`ValidatorStage` into a loop. The interfaces support it.

## License

[MIT](LICENSE) — do what you want, cite the thesis if it helped.
