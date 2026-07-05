"""Response cache: keying, read-through/write-through, zero-cost hits."""

from __future__ import annotations

import pytest

from evalpipe.cache import CachedProvider, cache_key
from evalpipe.providers.base import ModelProvider, ModelResponse
from evalpipe.storage import Storage


class CountingProvider(ModelProvider):
    """Records how many times the model was actually invoked."""

    model = "counter"
    input_cost_per_1k_tokens = 1.0
    output_cost_per_1k_tokens = 2.0

    def __init__(self) -> None:
        self.calls = 0

    async def generate(self, prompt: str, *, reference: str | None = None) -> ModelResponse:
        self.calls += 1
        return ModelResponse(text=f"answer::{prompt}", input_tokens=1000, output_tokens=1000)


def test_cache_key_is_stable_and_model_scoped() -> None:
    assert cache_key("m1", "hi") == cache_key("m1", "hi")
    assert cache_key("m1", "hi") != cache_key("m2", "hi")
    assert cache_key("m1", "hi") != cache_key("m1", "bye")


async def test_miss_then_hit(tmp_path) -> None:
    storage = Storage(str(tmp_path / "c.db"))
    inner = CountingProvider()
    provider = CachedProvider(inner, storage)

    first = await provider.generate("what is 2+2?")
    assert first.cached is False
    assert inner.calls == 1
    assert provider.misses == 1 and provider.hits == 0

    second = await provider.generate("what is 2+2?")
    assert second.cached is True
    assert second.text == first.text
    assert inner.calls == 1  # not called again
    assert provider.hits == 1


async def test_cached_response_is_free(tmp_path) -> None:
    storage = Storage(str(tmp_path / "c.db"))
    inner = CountingProvider()
    provider = CachedProvider(inner, storage)

    fresh = await provider.generate("q")
    assert provider.estimate_cost_usd(fresh) == pytest.approx(1.0 + 2.0)  # 1000+1000 tokens

    hit = await provider.generate("q")
    assert hit.cached is True
    assert provider.estimate_cost_usd(hit) == 0.0


async def test_distinct_prompts_are_separate_entries(tmp_path) -> None:
    storage = Storage(str(tmp_path / "c.db"))
    inner = CountingProvider()
    provider = CachedProvider(inner, storage)
    await provider.generate("a")
    await provider.generate("b")
    assert inner.calls == 2
    assert storage.cache_size() == 2


async def test_cache_survives_new_provider_instance(tmp_path) -> None:
    db = str(tmp_path / "c.db")
    storage = Storage(db)
    inner1 = CountingProvider()
    await CachedProvider(inner1, storage).generate("shared")

    inner2 = CountingProvider()
    response = await CachedProvider(inner2, Storage(db)).generate("shared")
    assert response.cached is True
    assert inner2.calls == 0


async def test_pipeline_cache_zeroes_rerun_cost(tmp_path) -> None:
    import json

    from evalpipe.config import EvalConfig, ExactMatchConfig, MockProviderConfig
    from evalpipe.pipeline import execute_run

    dataset = tmp_path / "d.jsonl"
    dataset.write_text(
        "\n".join(
            json.dumps({"id": f"i{n}", "prompt": f"q{n}", "expected": f"a{n}"}) for n in range(5)
        ),
        encoding="utf-8",
    )
    config = EvalConfig(
        name="cached",
        dataset=str(dataset),
        provider=MockProviderConfig(
            model="m",
            quality=1.0,
            input_cost_per_1k_tokens=1.0,
            output_cost_per_1k_tokens=1.0,
        ),
        evaluators=[ExactMatchConfig()],
        cache_responses=True,
        retries=0,
        retry_backoff_s=0.0,
    )
    storage = Storage(str(tmp_path / "runs.db"))

    first = await execute_run(config, storage, run_id="r1")
    assert first.total_cost_usd > 0.0

    second = await execute_run(config, storage, run_id="r2")
    assert second.total_cost_usd == 0.0  # every item served from cache
    assert second.pass_rate == first.pass_rate
