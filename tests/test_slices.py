"""Subgroup (slice) analysis: key selection, grouping, ordering, persistence."""

from __future__ import annotations

import asyncio
import json

from evalpipe.config import EvalConfig, ExactMatchConfig, MockProviderConfig
from evalpipe.pipeline import execute_run
from evalpipe.runner import ItemResult
from evalpipe.slices import UNLABELED, slice_by, sliceable_keys
from evalpipe.storage import Storage


def item(item_id: str, passed: bool, score: float, metadata: dict[str, str]) -> ItemResult:
    return ItemResult(
        item_id=item_id,
        prompt="p",
        expected=None,
        output="o",
        scores=(),
        passed=passed,
        mean_score=score,
        latency_ms=0.0,
        cost_usd=0.0,
        attempts=1,
        metadata=metadata,
    )


class TestSliceableKeys:
    def test_offers_low_cardinality_categorical_keys(self) -> None:
        results = [
            item("a", True, 1.0, {"topic": "geo", "difficulty": "easy"}),
            item("b", False, 0.0, {"topic": "sci", "difficulty": "easy"}),
            item("c", True, 1.0, {"topic": "geo", "difficulty": "hard"}),
        ]
        assert sliceable_keys(results) == ["difficulty", "topic"]

    def test_excludes_single_value_keys(self) -> None:
        results = [item(str(i), True, 1.0, {"const": "x"}) for i in range(5)]
        assert sliceable_keys(results) == []

    def test_excludes_per_item_high_cardinality_keys(self) -> None:
        results = [item(str(i), True, 1.0, {"context": f"unique-{i}"}) for i in range(20)]
        assert "context" not in sliceable_keys(results)

    def test_ignores_empty_values(self) -> None:
        results = [
            item("a", True, 1.0, {"topic": "geo"}),
            item("b", True, 1.0, {"topic": ""}),
        ]
        assert sliceable_keys(results) == []  # only one non-empty value


class TestSliceBy:
    def test_groups_and_orders_weakest_first(self) -> None:
        results = [item(f"e{i}", True, 1.0, {"difficulty": "easy"}) for i in range(8)] + [
            item(f"h{i}", i < 2, 0.3, {"difficulty": "hard"}) for i in range(10)
        ]
        stats = slice_by(results, "difficulty")
        assert [s.value for s in stats] == ["hard", "easy"]  # weakest slice first
        hard, easy = stats
        assert hard.n == 10 and hard.passed == 2
        assert hard.pass_rate == 0.2
        assert easy.pass_rate == 1.0
        assert hard.pass_ci[0] <= hard.pass_rate <= hard.pass_ci[1]

    def test_missing_key_falls_into_unlabeled(self) -> None:
        results = [
            item("a", True, 1.0, {"topic": "geo"}),
            item("b", False, 0.0, {}),
        ]
        stats = slice_by(results, "topic")
        values = {s.value for s in stats}
        assert values == {"geo", UNLABELED}


def test_metadata_persists_and_slices_from_storage(tmp_path) -> None:
    dataset = tmp_path / "d.jsonl"
    rows = [
        json.dumps(
            {
                "id": f"i{n}",
                "prompt": f"q{n}",
                "expected": f"a{n}",
                "metadata": {"topic": "easy" if n < 6 else "hard"},
            }
        )
        for n in range(12)
    ]
    dataset.write_text("\n".join(rows), encoding="utf-8")

    config = EvalConfig(
        name="sliced",
        dataset=str(dataset),
        provider=MockProviderConfig(model="m", quality=1.0),
        evaluators=[ExactMatchConfig()],
        retries=0,
        retry_backoff_s=0.0,
    )
    storage = Storage(str(tmp_path / "runs.db"))
    run = asyncio.run(execute_run(config, storage))

    stored = storage.get_results(run.run_id)
    assert all(r.metadata.get("topic") in {"easy", "hard"} for r in stored)

    stats = slice_by(stored, "topic")
    assert {s.value for s in stats} == {"easy", "hard"}
    assert sum(s.n for s in stats) == 12
