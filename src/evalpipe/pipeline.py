"""End-to-end pipeline: config → dataset → provider → evaluators → runner → storage.

This is the single entry point shared by the CLI and the API, so both
surfaces behave identically.
"""

from __future__ import annotations

import uuid

from evalpipe.cache import CachedProvider
from evalpipe.config import EvalConfig
from evalpipe.datasets import load_dataset
from evalpipe.evaluators import build_evaluators
from evalpipe.providers import build_provider
from evalpipe.providers.base import ModelProvider
from evalpipe.runner import ProgressCallback, RunResult, run_evaluation
from evalpipe.storage import Storage


def build_run_provider(config: EvalConfig, storage: Storage) -> ModelProvider:
    """Build the generation provider, wrapping it in the response cache when enabled."""
    provider = build_provider(config.provider)
    if config.cache_responses:
        return CachedProvider(provider, storage)
    return provider


async def execute_run(
    config: EvalConfig,
    storage: Storage,
    *,
    run_id: str | None = None,
    progress: ProgressCallback | None = None,
) -> RunResult:
    """Execute one evaluation run and persist it.

    Inputs are resolved *before* the run row is created, so a bad dataset path
    or missing API key fails fast without leaving a phantom run behind. Any
    failure after that point marks the stored run as failed.
    """
    items = load_dataset(config.dataset)
    provider = build_run_provider(config, storage)
    evaluators = build_evaluators(config.evaluators, provider)

    run_id = run_id or uuid.uuid4().hex
    storage.create_run(run_id, config.name, provider.model, config.dataset, config.model_dump())
    try:
        result = await run_evaluation(
            config, items, provider, evaluators, run_id=run_id, progress=progress
        )
        storage.complete_run(result)
        return result
    except Exception as exc:
        storage.fail_run(run_id, f"{type(exc).__name__}: {exc}")
        raise
    finally:
        await provider.aclose()
