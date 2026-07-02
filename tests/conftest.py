"""Shared fixtures and helpers for the test suite."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evalpipe.config import (
    EvalConfig,
    ExactMatchConfig,
    MockProviderConfig,
    TokenF1Config,
)
from evalpipe.datasets import DatasetItem


def make_items(count: int = 6) -> list[DatasetItem]:
    return [
        DatasetItem(
            id=f"item-{index:03d}",
            prompt=f"What is {index} plus {index}?",
            expected=f"The answer is {index + index}.",
        )
        for index in range(count)
    ]


@pytest.fixture
def items() -> list[DatasetItem]:
    return make_items()


@pytest.fixture
def dataset_file(tmp_path: Path) -> Path:
    path = tmp_path / "dataset.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for item in make_items():
            handle.write(
                json.dumps({"id": item.id, "prompt": item.prompt, "expected": item.expected}) + "\n"
            )
    return path


@pytest.fixture
def db_path(tmp_path: Path) -> str:
    return str(tmp_path / "test.db")


def make_config(
    dataset: str,
    *,
    name: str = "test-run",
    quality: float = 1.0,
    seed: int = 42,
    model: str = "sim-test",
    failure_rate: float = 0.0,
) -> EvalConfig:
    return EvalConfig(
        name=name,
        dataset=dataset,
        provider=MockProviderConfig(
            model=model, quality=quality, seed=seed, failure_rate=failure_rate
        ),
        evaluators=[ExactMatchConfig(), TokenF1Config()],
        concurrency=4,
        retries=1,
        retry_backoff_s=0.0,
    )
