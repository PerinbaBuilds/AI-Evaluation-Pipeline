"""Response caching.

A :class:`CachedProvider` wraps any provider and memoises completions in the
SQLite ``response_cache`` table, keyed by a hash of ``(model, prompt)``. The
point is workflow economics: once you have generated outputs for a dataset,
you can iterate on *evaluators and thresholds* and re-score for free, without
paying for inference again.

Caching is only sound for deterministic decoding (``temperature = 0``); it is
opt-in per run via ``cache_responses`` in the config, and cache hits are billed
at zero cost so the run's cost reflects real inference spend.
"""

from __future__ import annotations

import hashlib

from evalpipe.providers.base import ModelProvider, ModelResponse
from evalpipe.storage import Storage


def cache_key(model: str, prompt: str) -> str:
    """Stable content hash used as the cache key for a completion."""
    digest = hashlib.sha256(f"{model}\x00{prompt}".encode()).hexdigest()
    return f"{model}:{digest}"


class CachedProvider(ModelProvider):
    """Read-through/write-through cache around another provider."""

    def __init__(self, inner: ModelProvider, storage: Storage) -> None:
        self._inner = inner
        self._storage = storage
        self.model = inner.model
        self.input_cost_per_1k_tokens = inner.input_cost_per_1k_tokens
        self.output_cost_per_1k_tokens = inner.output_cost_per_1k_tokens
        self.hits = 0
        self.misses = 0

    async def generate(self, prompt: str, *, reference: str | None = None) -> ModelResponse:
        key = cache_key(self._inner.model, prompt)
        cached = self._storage.get_cached_response(key)
        if cached is not None:
            self.hits += 1
            output, input_tokens, output_tokens = cached
            return ModelResponse(
                text=output,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cached=True,
            )
        self.misses += 1
        response = await self._inner.generate(prompt, reference=reference)
        self._storage.put_cached_response(
            key,
            self._inner.model,
            response.text,
            response.input_tokens,
            response.output_tokens,
        )
        return response

    async def aclose(self) -> None:
        await self._inner.aclose()
