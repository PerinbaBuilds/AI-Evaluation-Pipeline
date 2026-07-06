"""Anthropic Messages API provider.

Implements the native ``/v1/messages`` wire format (distinct from the OpenAI
chat-completions shape). The API key is passed in by the factory from an
environment variable and sent as the ``x-api-key`` header — it never appears
in config files or reaches the browser.
"""

from __future__ import annotations

from typing import Any

import httpx

from evalpipe.exceptions import ProviderError
from evalpipe.providers.base import ModelProvider, ModelResponse


class AnthropicProvider(ModelProvider):
    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str = "https://api.anthropic.com",
        api_version: str = "2023-06-01",
        temperature: float = 0.0,
        max_tokens: int = 512,
        timeout_s: float = 60.0,
        input_cost_per_1k_tokens: float = 0.0,
        output_cost_per_1k_tokens: float = 0.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.input_cost_per_1k_tokens = input_cost_per_1k_tokens
        self.output_cost_per_1k_tokens = output_cost_per_1k_tokens
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers={
                "x-api-key": api_key,
                "anthropic-version": api_version,
                "content-type": "application/json",
            },
            timeout=timeout_s,
            transport=transport,
        )

    async def generate(self, prompt: str, *, reference: str | None = None) -> ModelResponse:
        payload = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        try:
            response = await self._client.post("/v1/messages", json=payload)
        except httpx.HTTPError as exc:
            raise ProviderError(f"{self.model}: request failed: {exc}") from exc
        if response.status_code != 200:
            raise ProviderError(f"{self.model}: HTTP {response.status_code}: {response.text[:200]}")
        return self._parse(response)

    def _parse(self, response: httpx.Response) -> ModelResponse:
        try:
            body: Any = response.json()
            blocks = body["content"]
            text = "".join(block.get("text", "") for block in blocks if block.get("type") == "text")
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            raise ProviderError(f"{self.model}: malformed response body") from exc
        usage = body.get("usage") or {}
        return ModelResponse(
            text=text,
            input_tokens=int(usage.get("input_tokens") or 0),
            output_tokens=int(usage.get("output_tokens") or 0),
        )

    async def aclose(self) -> None:
        await self._client.aclose()
