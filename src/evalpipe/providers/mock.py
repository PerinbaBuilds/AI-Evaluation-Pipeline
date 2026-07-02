"""Deterministic simulation provider.

``MockProvider`` behaves like a model of a configurable accuracy level. For a
given ``(seed, model, prompt)`` triple it always produces the same output, so
evaluation runs, tests and A/B demos are fully reproducible and run offline.

Given the item's reference answer it emits, per item:

- the correct answer (probability ``quality``),
- a partially-correct answer (half of the remaining probability), or
- a plausible-but-wrong canned answer (the rest),

which yields realistic score distributions across the metric suite instead of
a degenerate all-or-nothing split.

When the prompt is an LLM-judge grading request (it asks for a JSON
``{"score": ...}`` verdict), the provider simulates a grader instead: it
returns a deterministic JSON verdict centred around its ``quality`` level, so
judge-based metrics also work offline.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import random

from evalpipe.exceptions import ProviderError
from evalpipe.providers.base import ModelProvider, ModelResponse

_WRONG_ANSWERS = (
    "I do not have enough information to answer that question.",
    "The answer depends on several factors that are not specified here.",
    "That is outside the scope of what I can determine from the prompt.",
)


class MockProvider(ModelProvider):
    def __init__(
        self,
        model: str = "sim-model",
        quality: float = 0.8,
        seed: int = 42,
        failure_rate: float = 0.0,
        latency_ms: float = 0.0,
        input_cost_per_1k_tokens: float = 0.0,
        output_cost_per_1k_tokens: float = 0.0,
    ) -> None:
        if not 0.0 <= quality <= 1.0:
            raise ValueError("quality must be within [0, 1]")
        if not 0.0 <= failure_rate <= 1.0:
            raise ValueError("failure_rate must be within [0, 1]")
        self.model = model
        self.quality = quality
        self.seed = seed
        self.failure_rate = failure_rate
        self.latency_ms = latency_ms
        self.input_cost_per_1k_tokens = input_cost_per_1k_tokens
        self.output_cost_per_1k_tokens = output_cost_per_1k_tokens

    def _rng(self, prompt: str) -> random.Random:
        key = f"{self.seed}:{self.model}:{prompt}".encode()
        return random.Random(int.from_bytes(hashlib.sha256(key).digest()[:8], "big"))

    async def generate(self, prompt: str, *, reference: str | None = None) -> ModelResponse:
        rng = self._rng(prompt)
        if self.latency_ms > 0:
            await asyncio.sleep(rng.uniform(0.5, 1.5) * self.latency_ms / 1000.0)
        if rng.random() < self.failure_rate:
            raise ProviderError(f"{self.model}: simulated transient failure")

        text = self._answer(rng, prompt, reference)
        return ModelResponse(
            text=text,
            input_tokens=max(1, len(prompt.split())),
            output_tokens=max(1, len(text.split())),
        )

    def _answer(self, rng: random.Random, prompt: str, reference: str | None) -> str:
        if '"score"' in prompt:
            return self._judge_verdict(rng)
        if reference is None or not reference.strip():
            return f"Here is a general response to: {prompt.strip()[:80]}"
        roll = rng.random()
        if roll < self.quality:
            return reference
        if roll < self.quality + (1.0 - self.quality) / 2.0:
            return self._partial(rng, reference)
        return rng.choice(_WRONG_ANSWERS)

    def _judge_verdict(self, rng: random.Random) -> str:
        score = round(min(10.0, max(0.0, rng.gauss(2.0 + 8.0 * self.quality, 1.5))))
        return json.dumps({"score": int(score), "reasoning": "Simulated grading verdict."})

    @staticmethod
    def _partial(rng: random.Random, reference: str) -> str:
        """Keep roughly half of the reference tokens so overlap metrics land mid-range."""
        words = reference.split()
        if len(words) <= 1:
            return f"Possibly {reference}, though I am not certain."
        kept = [word for word in words if rng.random() < 0.6]
        if not kept or len(kept) == len(words):
            kept = words[: max(1, len(words) // 2)]
        return " ".join(kept)
