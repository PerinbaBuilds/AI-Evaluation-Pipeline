"""Google Gemini provider (generateContent API).

Implements the native ``/v1beta/models/{model}:generateContent`` wire format.
The API key is supplied by the factory from an environment variable and sent
as the ``key`` query parameter per the Gemini REST contract; it never appears
in config files or reaches the browser.
"""

from __future__ import annotations

from typing import Any

import httpx

from evalpipe.exceptions import ProviderError
from evalpipe.providers.base import ModelProvider, ModelResponse


class GeminiProvider(ModelProvider):
    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str = "https://generativelanguage.googleapis.com",
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
        self._api_key = api_key
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers={"content-type": "application/json"},
            timeout=timeout_s,
            transport=transport,
        )

    async def generate(self, prompt: str, *, reference: str | None = None) -> ModelResponse:
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": self.temperature,
                "maxOutputTokens": self.max_tokens,
            },
        }
        url = f"/v1beta/models/{self.model}:generateContent"
        try:
            response = await self._client.post(url, json=payload, params={"key": self._api_key})
        except httpx.HTTPError as exc:
            raise ProviderError(f"{self.model}: request failed: {exc}") from exc
        if response.status_code != 200:
            raise ProviderError(f"{self.model}: HTTP {response.status_code}: {response.text[:200]}")
        return self._parse(response)

    def _parse(self, response: httpx.Response) -> ModelResponse:
        try:
            body: Any = response.json()
            parts = body["candidates"][0]["content"]["parts"]
            text = "".join(part.get("text", "") for part in parts)
        except (ValueError, KeyError, IndexError, TypeError, AttributeError) as exc:
            raise ProviderError(f"{self.model}: malformed response body") from exc
        usage = body.get("usageMetadata") or {}
        return ModelResponse(
            text=text,
            input_tokens=int(usage.get("promptTokenCount") or 0),
            output_tokens=int(usage.get("candidatesTokenCount") or 0),
        )

    async def aclose(self) -> None:
        await self._client.aclose()
