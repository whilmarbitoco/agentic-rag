# Contributing to `agent`

`agent` is the **open-source foundation** of the KLIMA project — a
domain-agnostic, extensible agentic-RAG framework generalized from the
architecture in the KLIMA undergraduate thesis (Bitoco, 2026).

This repository is the **generic plumbing**. It is intentionally free of any
single domain. The commercial KLIMA product is built *on top of* `agent` in a
separate, proprietary repository.

## What belongs in this repo (in-scope)

- The six-stage pipeline and its interfaces (`LLMProvider`, `MemoryModule`, `Stage`, `@tool`).
- Generic provider adapters (OpenAI-compatible, mock) and reference memory modules.
- Deterministic executor, reranker, validator, synthesizer behavior.
- Framework-level examples, tests, and documentation.
- Cross-domain improvements that benefit any agent built on `agent`.

## What does NOT belong here (out-of-scope — lives in the proprietary KLIMA product)

- Domain-specific tools (weather APIs, government agri/livelihood data feeds, crop-disease DBs).
- Localized prompt sets (e.g. Tagalog / Cebuano tuning) and domain knowledge.
- Trained models and weights (time-series / climate / recommendation).
- IoT sensor ingestion and device management.
- The farmer-facing mobile / web client.
- Managed deployment, hosting, billing, and customer data.

If your change targets a domain, keep it in the product repo and depend on
`agent` via `pip install agentic-rag`. If it generalizes cleanly, open a PR here.

## How to contribute

1. Fork and branch (`feat/...`, `fix/...`).
2. Add or update tests — the suite must pass offline (`pytest`).
3. Keep the public API stable; mark breaking changes clearly in the PR.
4. Open a pull request with a concise description of the intent and scope.

## License

MIT. By contributing, you agree your contributions are released under the same
license. This repo contains no proprietary KLIMA IP.
