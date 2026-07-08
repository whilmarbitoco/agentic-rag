# Contributing to `agentic-rag`

`agentic-rag` is the **open-source foundation** of the KLIMA project — a
domain-agnostic, extensible agentic-RAG framework generalized from the
architecture in the KLIMA undergraduate thesis (Bitoco, 2026).

This repository is the **generic plumbing**. It is intentionally free of any
single domain. The commercial KLIMA product is built *on top of* `agentic-rag`
in a separate, proprietary repository.

---

## What belongs in this repo (in-scope)

- The six-stage pipeline and its interfaces (`LLMProvider`, `MemoryModule`, `Stage`, `@tool`).
- Generic provider adapters (OpenAI-compatible, mock) and reference memory modules.
- Deterministic executor, reranker, validator, synthesizer behavior.
- Framework-level examples, tests, and documentation.
- Cross-domain improvements that benefit any agent built on `agentic-rag`.

## What does NOT belong here (out-of-scope)

- Domain-specific tools (weather APIs, government data feeds, crop-disease DBs).
- Localized prompt sets (e.g. Tagalog / Cebuano tuning) and domain knowledge.
- Trained models and weights (time-series / climate / recommendation).
- IoT sensor ingestion and device management.
- The farmer-facing mobile / web client.
- Managed deployment, hosting, billing, and customer data.

If your change targets a domain, keep it in the product repo and depend on
`agentic-rag` via `pip install agentic-rag`. If it generalizes cleanly, open a PR here.

---

## How to contribute

1. **Fork and branch** — use `feat/...`, `fix/...`, `refactor/...`, `test/...`, `docs/...`.
2. **Write tests** — the suite must pass offline with `pytest`. All tests use
   `MockProvider` so no network access is needed.
3. **Register tools in `setup_module()`** — use a module-scoped pytest fixture
   or `setup_module(module)` to register tools on `get_default_registry()`.
   This keeps tests isolated across modules.
4. **Run linting before committing:**
   ```bash
   pip install ruff
   ruff check .
   ```
5. **Keep the public API stable.** Mark breaking changes clearly in the PR. If a
   breaking change is unavoidable, discuss it first via an issue.
6. **Commit messages** follow the existing style:
   ```
   feat: short description of the feature
   fix: short description of the bug
   refactor: what changed and why
   test: what is now covered
   docs: what was updated
   chore: tooling, dependencies, config
   ```
7. **Open a pull request** with a concise description of the intent and scope.

---

## License

MIT. By contributing, you agree your contributions are released under the same
license. This repo contains no proprietary KLIMA IP.