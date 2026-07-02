"""Runner orchestration: concurrency limits, retries, per-item failure isolation."""

from __future__ import annotations

import asyncio

import pytest
from tests.conftest import make_config, make_items

from evalpipe.config import ExactMatchConfig, MockProviderConfig
from evalpipe.datasets import DatasetItem
from evalpipe.evaluators import build_evaluators
from evalpipe.evaluators.base import EvalScore, Evaluator
from evalpipe.exceptions import ProviderError
from evalpipe.providers.base import ModelProvider, ModelResponse
from evalpipe.providers.mock import MockProvider
from evalpipe.runner import run_evaluation


class FlakyProvider(ModelProvider):
    """Fails the first ``failures_per_item`` calls for every prompt."""

    model = "flaky"

    def __init__(self, failures_per_item: int) -> None:
        self.failures_per_item = failures_per_item
        self.calls: dict[str, int] = {}

    async def generate(self, prompt: str, *, reference: str | None = None) -> ModelResponse:
        count = self.calls.get(prompt, 0) + 1
        self.calls[prompt] = count
        if count <= self.failures_per_item:
            raise ProviderError("transient")
        return ModelResponse(text=reference or "ok")


class ConcurrencyProbe(ModelProvider):
    model = "probe"

    def __init__(self) -> None:
        self.active = 0
        self.max_active = 0

    async def generate(self, prompt: str, *, reference: str | None = None) -> ModelResponse:
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        await asyncio.sleep(0.01)
        self.active -= 1
        return ModelResponse(text=reference or "ok")


class ExplodingEvaluator(Evaluator):
    name = "exploding"
    threshold = 0.5

    async def evaluate(self, item: DatasetItem, output: str) -> EvalScore:
        raise RuntimeError("metric bug")


async def test_perfect_provider_passes_everything(dataset_file) -> None:
    config = make_config(str(dataset_file), quality=1.0)
    items = make_items()
    provider = MockProvider(quality=1.0)
    evaluators = build_evaluators(config.evaluators, provider)
    result = await run_evaluation(config, items, provider, evaluators)
    assert result.item_count == 6
    assert result.pass_rate == 1.0
    assert result.mean_score == 1.0
    assert result.error_count == 0
    assert all(item.attempts == 1 for item in result.items)


async def test_results_preserve_dataset_order(dataset_file) -> None:
    config = make_config(str(dataset_file))
    items = make_items()
    provider = MockProvider(quality=1.0)
    evaluators = build_evaluators(config.evaluators, provider)
    result = await run_evaluation(config, items, provider, evaluators)
    assert [item.item_id for item in result.items] == [item.id for item in items]


async def test_retries_recover_from_transient_failures(dataset_file) -> None:
    config = make_config(str(dataset_file)).model_copy(update={"retries": 2})
    items = make_items(3)
    provider = FlakyProvider(failures_per_item=2)
    evaluators = build_evaluators(config.evaluators, provider)
    result = await run_evaluation(config, items, provider, evaluators)
    assert result.error_count == 0
    assert all(item.attempts == 3 for item in result.items)
    assert result.pass_rate == 1.0


async def test_exhausted_retries_record_error_without_aborting(dataset_file) -> None:
    config = make_config(str(dataset_file)).model_copy(update={"retries": 1})
    items = make_items(4)
    provider = FlakyProvider(failures_per_item=99)
    evaluators = build_evaluators(config.evaluators, provider)
    result = await run_evaluation(config, items, provider, evaluators)
    assert result.item_count == 4
    assert result.error_count == 4
    assert result.pass_rate == 0.0
    for item_result in result.items:
        assert item_result.error is not None
        assert "2 attempt(s)" in item_result.error
        assert item_result.scores == ()


async def test_concurrency_is_bounded(dataset_file) -> None:
    config = make_config(str(dataset_file)).model_copy(update={"concurrency": 3})
    items = make_items(12)
    provider = ConcurrencyProbe()
    evaluators = build_evaluators(config.evaluators, provider)
    await run_evaluation(config, items, provider, evaluators)
    assert provider.max_active <= 3


async def test_prompt_template_is_rendered(dataset_file) -> None:
    config = make_config(str(dataset_file)).model_copy(
        update={"prompt_template": "SYSTEM RULES\n\nQ: {prompt}"}
    )
    items = make_items(2)
    provider = MockProvider(quality=1.0)
    evaluators = build_evaluators(config.evaluators, provider)
    result = await run_evaluation(config, items, provider, evaluators)
    assert all(item.prompt.startswith("SYSTEM RULES") for item in result.items)


async def test_broken_evaluator_scores_zero_without_sinking_item(dataset_file) -> None:
    config = make_config(str(dataset_file))
    items = make_items(2)
    provider = MockProvider(quality=1.0)
    evaluators = [*build_evaluators([ExactMatchConfig()], provider), ExplodingEvaluator()]
    result = await run_evaluation(config, items, provider, evaluators)
    assert result.error_count == 0
    for item_result in result.items:
        exploding = next(score for score in item_result.scores if score.name == "exploding")
        assert exploding.score == 0.0
        assert "RuntimeError" in exploding.detail
        assert not item_result.passed  # the broken metric fails the item, visibly


async def test_progress_callback_reports_all_items(dataset_file) -> None:
    config = make_config(str(dataset_file))
    items = make_items(5)
    provider = MockProvider(quality=1.0)
    evaluators = build_evaluators(config.evaluators, provider)
    seen: list[tuple[int, int]] = []
    await run_evaluation(
        config, items, provider, evaluators, progress=lambda done, total: seen.append((done, total))
    )
    assert len(seen) == 5
    assert seen[-1] == (5, 5)


async def test_empty_items_rejected(dataset_file) -> None:
    config = make_config(str(dataset_file))
    provider = MockProvider()
    evaluators = build_evaluators(config.evaluators, provider)
    with pytest.raises(ValueError, match="zero items"):
        await run_evaluation(config, [], provider, evaluators)


async def test_evaluator_means_aggregation(dataset_file) -> None:
    config = make_config(str(dataset_file))
    items = make_items(4)
    provider = MockProvider(quality=1.0)
    evaluators = build_evaluators(config.evaluators, provider)
    result = await run_evaluation(config, items, provider, evaluators)
    means = result.evaluator_means()
    assert set(means) == {"exact_match", "token_f1"}
    assert means["exact_match"] == 1.0


async def test_run_uses_config_provider_metadata(dataset_file) -> None:
    config = make_config(str(dataset_file), model="my-model")
    provider = MockProvider(model="my-model", quality=1.0)
    evaluators = build_evaluators(config.evaluators, provider)
    result = await run_evaluation(config, make_items(2), provider, evaluators, run_id="fixed-id")
    assert result.run_id == "fixed-id"
    assert result.model == "my-model"


def test_mock_config_roundtrip() -> None:
    # ensures the conftest factory builds a valid, typed config
    config = make_config("x.jsonl", quality=0.5)
    assert isinstance(config.provider, MockProviderConfig)
    assert config.retry_backoff_s == 0.0
