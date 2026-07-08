# agentic-rag — The AI Agent Architecture from My Undergraduate Thesis, Open-Sourced

> A domain-agnostic, extensible 6-stage agentic RAG pipeline generalized from the
> architecture in *"KLIMA: A Microclimate Monitoring and Decision-Support System
> for Rice Farming Using IoT and Time-Series Learning in Tagum City"* (Bitoco, 2026).
> The original system — codenamed **JOY** — powered a farmer-facing climate advisor.
> This repo extracts the production-proven architecture into a reusable template.
> Drop in your own tools, prompts, and LLM providers. No fork required.

[![CI](https://img.shields.io/badge/tests-40_passing-green)]()
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)]()
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

---

## Quick Start

```python
from agent import AgenticPipeline, ToolContext, tool

@tool("get_weather", description="Current weather",
      parameters={"type": "object", "properties": {"city": {"type": "string"}}})
def get_weather(ctx: ToolContext, city: str) -> dict:
    return {"temp_c": 31, "condition": "partly cloudy"}

providers = {"interpreter": MockProvider(), "planner": MockProvider(),
             "reranker": MockProvider(), "synthesizer": MockProvider(),
             "validator": MockProvider()}
pipe = AgenticPipeline.build_default(providers=providers)
result = pipe.run("What's the weather in Manila?", tool_ctx=ToolContext(tenant_id=1))
print(result.reply)   # "Based on [get_weather], Manila is 31C and partly cloudy."
print(result.trace)    # per-stage observability
```

Swap `MockProvider()` for `OpenAICompatProvider(base_url=..., api_key=..., model=...)`
and you're in production. Nothing else changes. See [`examples/`](examples) for complete
offline demos.

---

## Why this exists

Most agent frameworks either (a) hide the control flow or (b) require an LLM to
call tools. This architecture was built for **production reliability**, not demo flash:

- **Deterministic tool execution.** Tools are plain Python functions called in
  parallel — never via the LLM. No hallucinated tool names, no per-call model cost.
- **Six explicit, swappable stages.** Each stage is a self-contained unit with one
  job and a clean interface.
- **Self-correcting validator.** A feedback loop catches ungrounded answers and
  regenerates before the user sees them.
- **Built-in graceful degradation.** Any stage can fail and the pipeline still
  returns a safe answer.
- **Dynamic context-window budgeting.** Budgets adapt to each model's capacity
  automatically. See [Context window management](#context-window-management).
- **Trace-based evaluation baked in.** Every run produces a `trace` you can score
  stage-by-stage.

---

## Architecture

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
        │Synthesizer │◄─────┤Validator │  self-check; regenerate on failure (loop)
        └────────────┘      └──────────┘
               │
               ▼
            answer + trace
   (Memory module feeds Interpreter + stores turns in the background)
```

| Stage | Role | Default |
|---|---|---|
| Interpreter | dereference references, pick execution mode, fetch memory | LLM JSON |
| Planner | tool selection + query rewrite (in-domain guard) | LLM JSON |
| Executor | parallel tool calls (deterministic — no LLM) | `ThreadPoolExecutor` |
| Reranker | relevance scoring | heuristic score (swap for LLM judge) |
| Synthesizer | grounded answer generation | LLM |
| Validator | hallucination / grounding guard | heuristic triggers + optional LLM judge |
| Memory | conversation continuity across turns | `InMemoryMemory` / pluggable |

---

## Extension points

The whole philosophy is **swap, don't fork**:

1. **`LLMProvider`** — implement `complete()` and `complete_json()`. Ships with
   `OpenAICompatProvider` (OpenAI, Groq, NVIDIA, OpenRouter, Ollama, vLLM…),
   `MockProvider` (offline/dev/CI), and `ProviderFactory` with primary + fallback
   for rate-limit graceful degradation.

2. **`MemoryModule`** — implement `fetch_context()` and `store()`. Ships with
   `InMemoryMemory` (bounded deque, thread-safe) and `NoOpMemory`. Replace with
   pgvector, FTS, Redis — same interface.

3. **`Stage`** — subclass any pipeline stage to override behavior (e.g. safety
   pre-filter before the planner, cross-encoder reranker, custom validator).

4. **`@tool`** — register deterministic functions. Identity/tenant injected via
   `ToolContext` (never the LLM, preventing prompt injection). Uses `ToolRegistry`
   — instance-based with a default global. Isolate registries with
   `get_default_registry()` for multi-tenant or test isolation.

5. **`ContextManager` / `Compactor` / `TokenCounter`** — context-window
   budgeting: memory context and tool results are token-counted and compacted
   to a budget before reaching the LLM. See [section below](#context-window-management).

6. **`ValidationVerdict`** — the validator returns a typed dataclass
   (`valid: bool`, `critique: str`) instead of a raw dict. Access it in
   custom validators for type-safe composition.

7. **Provider fallback** — `OpenAICompatProvider(..., fallback=lambda: "...")`
   transparently catches 429 rate limits and delegates to an alternative source.
   No pipeline-level config needed per stage.

```python
from agent import AgenticPipeline, OpenAICompatProvider, MockProvider

primary = OpenAICompatProvider(base_url="https://api.openai.com/v1", api_key="sk-...", model="gpt-4o-mini")
fallback = OpenAICompatProvider(base_url="https://openrouter.ai/api/v1", api_key="sk-...", model="gpt-4o-mini")

pipe = AgenticPipeline.build_default(providers={
    "synthesizer": primary,  # fallback not wired here, but you can
})
```

---

## Context window management

Different models have vastly different context windows. The pipeline handles this
**dynamically** — no hardcoded model lookup table.

### How it works

1. Each `LLMProvider` exposes a `context_window` property (total token capacity).
2. At the start of every `pipeline.run()`, the smallest window across all stage
   providers becomes the **bottleneck budget**.
3. `BudgetContextManager` computes section budgets proportionally (25% memory, 50% tools,
   25% history) and enforces a **total budget circuit breaker** — no section can exceed
   the remaining capacity.

```
Example:
  Interpreter: GPT-4o-mini   (128K)
  Planner:        GPT-4o-mini   (128K)
  Validator:  Llama-3-8B    (8K)    ← bottleneck
                                   
  Budget:  tools ~3.8K | memory ~2K | history ~0.5K | total ~7.5K
```

### Override for custom models

```python
from agent import OpenAICompatProvider, BudgetContextManager

# Explicit context_window for Ollama / vLLM / custom endpoints
provider = OpenAICompatProvider(base_url="http://localhost:11434/v1",
                                api_key="ollama", model="llama3",
                                context_window=8192)

# Or override budgets directly
ctx_mgr = BudgetContextManager(budgets={"tools": 2000, "memory": 1000})
```

### Token counting

| Counter | When to use |
|---|---|
| `WordTokenCounter` | Default. No dependencies. ~1 token per 4 chars. |
| `TiktokenTokenCounter` | Exact counts for any OpenAI model. Requires `pip install tiktoken`. |

### Compaction strategies

| Compactor | Behavior |
|---|---|
| `HeuristicCompactor` | Keeps highest-scored results within budget, truncates the overflow item. |
| `LLMCompactor` | Summarizes all results into one block via an LLM (passes the query for relevance). |

---

## Install

```bash
pip install -e .                 # runtime (httpx only)
pip install -e ".[dev]"         # + pytest (for running tests)
pip install tiktoken            # optional — exact token counting
```

### Run tests

```bash
pytest                           # 40 tests, fully offline
```

All tests use `MockProvider` — no network, no API keys.

### Run examples

```bash
python examples/doc_qa.py            # Document Q&A demo
python examples/weather_assistant.py  # Weather assistant demo
```

Both run fully offline with scripted providers. Swap `ScriptedProvider` for
`OpenAICompatProvider(base_url=..., api_key=..., model=...)` to use a real LLM.

---

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

---

## License

[MIT](LICENSE) — do what you want, cite the thesis if it helped.