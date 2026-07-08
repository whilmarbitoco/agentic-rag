"""OpenAI-compatible chat provider (works with OpenAI, Groq, NVIDIA,
OpenRouter, Together, vLLM, Ollama, etc.)."""
from __future__ import annotations

import json
from typing import Any

import httpx

from ..errors import LLMProviderError, ProviderRateLimitError
from .base import LLMProvider, extract_json


class OpenAICompatProvider(LLMProvider):
    def __init__(
        self,
        base_url: str = "https://api.openai.com/v1",
        api_key: str = "",
        model: str = "gpt-4o-mini",
        timeout: float = 60.0,
        json_mode: bool = False,
        extra_headers: dict[str, str] | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.json_mode = json_mode
        self.extra_headers = extra_headers or {}
        self._client = httpx.Client(base_url=self.base_url, timeout=timeout)

    def _headers(self) -> dict[str, str]:
        h = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        h.update(self.extra_headers)
        return h

    def _chat(self, messages: list[dict], system_prompt: str | None, want_json: bool) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.2,
        }
        if system_prompt:
            payload["messages"] = [
                {"role": "system", "content": system_prompt},
                *messages,
            ]
        if want_json and self.json_mode:
            payload["response_format"] = {"type": "json_object"}
        resp = self._client.post("/chat/completions", json=payload)
        if resp.status_code == 429:
            raise ProviderRateLimitError("rate limited")
        if resp.status_code >= 400:
            raise LLMProviderError(f"{resp.status_code}: {resp.text[:300]}")
        return resp.json()["choices"][0]["message"]["content"]

    def complete(self, messages: list[dict], system_prompt: str | None = None) -> str:
        return self._chat(messages, system_prompt, want_json=False)

    def complete_json(self, messages: list[dict], system_prompt: str | None = None) -> dict:
        raw = self._chat(messages, system_prompt, want_json=True)
        return extract_json(raw)

    def close(self):
        self._client.close()
