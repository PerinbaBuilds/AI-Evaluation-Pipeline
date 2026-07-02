"""Evaluation runner: bounded-concurrency orchestration with retries.

The runner is the pipeline's Map step. Each dataset item is processed
independently under a semaphore: render the prompt, call the provider (with
exponential backoff on transient failures), then apply every evaluator. One
item failing — even after all retries — records an errored result and never
aborts the run.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime

from evalpipe.config import EvalConfig
from evalpipe.datasets import DatasetItem
from evalpipe.evaluators.base import EvalScore, Evaluator
from evalpipe.exceptions import ProviderError
from evalpipe.providers.base import ModelProvider
from evalpipe.stats import mean, percentile


@dataclass(frozen=True)
class ItemResult:
    item_id: str
    prompt: str
    expected: str | None
    output: str
    scores: tuple[EvalScore, ...]
    passed: bool
    mean_score: float
    latency_ms: float
    cost_usd: float
    attempts: int
    error: str | None = None


@dataclass
class RunResult:
    run_id: str
    name: str
    model: str
    dataset: str
    started_at: datetime
    finished_at: datetime
    items: list[ItemResult] = field(default_factory=list)

    @property
    def item_count(self) -> int:
        return len(self.items)

    @property
    def error_count(self) -> int:
        return sum(1 for item in self.items if item.error is not None)

    @property
    def pass_rate(self) -> float:
        if not self.items:
            return 0.0
        return sum(item.passed for item in self.items) / len(self.items)

    @property
    def mean_score(self) -> float:
        if not self.items:
            return 0.0
        return mean([item.mean_score for item in self.items])

    @property
    def total_cost_usd(self) -> float:
        return sum(item.cost_usd for item in self.items)

    @property
    def latency_p50_ms(self) -> float:
        return self._latency_percentile(50.0)

    @property
    def latency_p95_ms(self) -> float:
        return self._latency_percentile(95.0)

    def _latency_percentile(self, q: float) -> float:
        latencies = [item.latency_ms for item in self.items if item.error is None]
        return percentile(latencies, q) if latencies else 0.0

    def evaluator_means(self) -> dict[str, float]:
        """Mean score per evaluator across all items (errored items count as 0)."""
        totals: dict[str, list[float]] = {}
        for item in self.items:
            for score in item.scores:
                totals.setdefault(score.name, []).append(score.score)
        return {name: mean(scores) for name, scores in sorted(totals.items())}


ProgressCallback = Callable[[int, int], None]


async def run_evaluation(
    config: EvalConfig,
    items: Sequence[DatasetItem],
    provider: ModelProvider,
    evaluators: Sequence[Evaluator],
    *,
    run_id: str | None = None,
    progress: ProgressCallback | None = None,
) -> RunResult:
    """Evaluate every item and return the aggregated run."""
    if not items:
        raise ValueError("cannot run an evaluation over zero items")
    if not evaluators:
        raise ValueError("cannot run an evaluation without evaluators")

    started_at = datetime.now(UTC)
    semaphore = asyncio.Semaphore(config.concurrency)
    completed = 0

    async def process(item: DatasetItem) -> ItemResult:
        nonlocal completed
        async with semaphore:
            result = await _process_item(config, item, provider, evaluators)
        completed += 1
        if progress is not None:
            progress(completed, len(items))
        return result

    results = await asyncio.gather(*(process(item) for item in items))
    return RunResult(
        run_id=run_id or uuid.uuid4().hex,
        name=config.name,
        model=provider.model,
        dataset=config.dataset,
        started_at=started_at,
        finished_at=datetime.now(UTC),
        items=list(results),
    )


async def _process_item(
    config: EvalConfig,
    item: DatasetItem,
    provider: ModelProvider,
    evaluators: Sequence[Evaluator],
) -> ItemResult:
    rendered_prompt = config.render_prompt(item.prompt)
    start = time.perf_counter()
    attempts = 0
    last_error: str | None = None
    response = None

    for attempt in range(config.retries + 1):
        attempts = attempt + 1
        try:
            response = await provider.generate(rendered_prompt, reference=item.expected)
            last_error = None
            break
        except ProviderError as exc:
            last_error = str(exc)
            if attempt < config.retries and config.retry_backoff_s > 0:
                await asyncio.sleep(config.retry_backoff_s * (2**attempt))
    latency_ms = (time.perf_counter() - start) * 1000.0

    if response is None:
        return ItemResult(
            item_id=item.id,
            prompt=rendered_prompt,
            expected=item.expected,
            output="",
            scores=(),
            passed=False,
            mean_score=0.0,
            latency_ms=latency_ms,
            cost_usd=0.0,
            attempts=attempts,
            error=f"provider failed after {attempts} attempt(s): {last_error}",
        )

    scores: list[EvalScore] = []
    for evaluator in evaluators:
        try:
            scores.append(await evaluator.evaluate(item, response.text))
        except Exception as exc:
            scores.append(
                EvalScore(
                    name=evaluator.name,
                    score=0.0,
                    passed=False,
                    detail=f"evaluator raised {type(exc).__name__}: {exc}",
                )
            )

    return ItemResult(
        item_id=item.id,
        prompt=rendered_prompt,
        expected=item.expected,
        output=response.text,
        scores=tuple(scores),
        passed=all(score.passed for score in scores),
        mean_score=mean([score.score for score in scores]),
        latency_ms=latency_ms,
        cost_usd=provider.estimate_cost_usd(response),
        attempts=attempts,
    )
