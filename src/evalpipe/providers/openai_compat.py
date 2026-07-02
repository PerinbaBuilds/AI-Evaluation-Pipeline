"""HTTP provider for OpenAI-compatible chat-completion endpoints.

The wire format is the de-facto standard implemented by vLLM, Ollama,
llama.cpp server, LM Studio and most hosted inference APIs, so one provider
class covers self-hosted and commercial backends alike.
"""

from __future__ import annotations

from typing import Any

import httpx

from evalpipe.exceptions import ProviderError
from evalpipe.providers.base import ModelProvider, ModelResponse


class OpenAICompatibleProvider(ModelProvider):
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
        timeout_s: float = 30.0,
        input_cost_per_1k_tokens: float = 0.0,
        output_cost_per_1k_tokens: float = 0.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.input_cost_per_1k_tokens = input_cost_per_1k_tokens
        self.output_cost_per_1k_tokens = output_cost_per_1k_tokens
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers=headers,
            timeout=timeout_s,
            transport=transport,
        )

    async def generate(self, prompt: str, *, reference: str | None = None) -> ModelResponse:
        # `reference` is part of the provider interface for simulation providers
        # only; it is intentionally unused here and never leaves the process.
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        try:
            response = await self._client.post("/chat/completions", json=payload)
        except httpx.HTTPError as exc:
            raise ProviderError(f"{self.model}: request failed: {exc}") from exc
        if response.status_code != 200:
            raise ProviderError(f"{self.model}: HTTP {response.status_code}: {response.text[:200]}")
        return self._parse(response)

    def _parse(self, response: httpx.Response) -> ModelResponse:
        try:
            body: Any = response.json()
            text = body["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise ProviderError(f"{self.model}: malformed response body") from exc
        if not isinstance(text, str):
            raise ProviderError(f"{self.model}: response content is not text")
        usage = body.get("usage") or {}
        return ModelResponse(
            text=text,
            input_tokens=int(usage.get("prompt_tokens") or 0),
            output_tokens=int(usage.get("completion_tokens") or 0),
        )

    async def aclose(self) -> None:
        await self._client.aclose()
