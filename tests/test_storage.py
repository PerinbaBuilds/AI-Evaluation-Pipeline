"""SQLite persistence: run lifecycle, results, prompt versioning, error paths."""

from __future__ import annotations

import asyncio

import pytest
from tests.conftest import make_config, make_items

from evalpipe.evaluators import build_evaluators
from evalpipe.exceptions import StorageError
from evalpipe.pipeline import execute_run
from evalpipe.providers.mock import MockProvider
from evalpipe.runner import RunResult, run_evaluation
from evalpipe.storage import Storage


def run_result(config_dataset: str, *, run_id: str = "run-1", quality: float = 1.0) -> RunResult:
    config = make_config(config_dataset, quality=quality)
    provider = MockProvider(quality=quality)
    evaluators = build_evaluators(config.evaluators, provider)
    return asyncio.run(run_evaluation(config, make_items(), provider, evaluators, run_id=run_id))


class TestRunLifecycle:
    def test_create_complete_and_fetch(self, db_path: str, dataset_file) -> None:
        storage = Storage(db_path)
        result = run_result(str(dataset_file))
        storage.create_run(result.run_id, "test", "sim", str(dataset_file), {"a": 1})
        storage.complete_run(result)

        record = storage.get_run(result.run_id)
        assert record.status == "completed"
        assert record.item_count == 6
        assert record.pass_rate == 1.0
        assert record.evaluator_means["exact_match"] == 1.0
        assert record.finished_at is not None

    def test_fail_run(self, db_path: str) -> None:
        storage = Storage(db_path)
        storage.create_run("r1", "test", "sim", "d.jsonl", {})
        storage.fail_run("r1", "dataset went missing")
        record = storage.get_run("r1")
        assert record.status == "failed"
        assert record.error == "dataset went missing"

    def test_duplicate_run_id_rejected(self, db_path: str) -> None:
        storage = Storage(db_path)
        storage.create_run("r1", "test", "sim", "d.jsonl", {})
        with pytest.raises(StorageError, match="already exists"):
            storage.create_run("r1", "other", "sim", "d.jsonl", {})

    def test_unknown_run_operations_raise(self, db_path: str) -> None:
        storage = Storage(db_path)
        with pytest.raises(StorageError, match="does not exist"):
            storage.get_run("ghost")
        with pytest.raises(StorageError, match="does not exist"):
            storage.fail_run("ghost", "err")
        with pytest.raises(StorageError, match="does not exist"):
            storage.delete_run("ghost")

    def test_list_runs_most_recent_first(self, db_path: str) -> None:
        storage = Storage(db_path)
        for index in range(3):
            storage.create_run(f"r{index}", f"run-{index}", "sim", "d.jsonl", {})
        listed = storage.list_runs()
        assert len(listed) == 3
        assert storage.count_runs() == 3
        assert listed[0].started_at >= listed[-1].started_at

    def test_delete_run_cascades_to_results(self, db_path: str, dataset_file) -> None:
        storage = Storage(db_path)
        result = run_result(str(dataset_file))
        storage.create_run(result.run_id, "test", "sim", str(dataset_file), {})
        storage.complete_run(result)
        assert storage.get_results(result.run_id)
        storage.delete_run(result.run_id)
        with pytest.raises(StorageError):
            storage.get_run(result.run_id)


class TestResults:
    def test_roundtrip_preserves_scores(self, db_path: str, dataset_file) -> None:
        storage = Storage(db_path)
        result = run_result(str(dataset_file))
        storage.create_run(result.run_id, "test", "sim", str(dataset_file), {})
        storage.complete_run(result)

        stored = storage.get_results(result.run_id)
        assert len(stored) == 6
        original = {item.item_id: item for item in result.items}
        for row in stored:
            assert row.scores == original[row.item_id].scores
            assert row.output == original[row.item_id].output

    def test_passed_filter(self, db_path: str, dataset_file) -> None:
        storage = Storage(db_path)
        result = run_result(str(dataset_file), quality=0.5)
        storage.create_run(result.run_id, "test", "sim", str(dataset_file), {})
        storage.complete_run(result)

        passed = storage.get_results(result.run_id, passed=True)
        failed = storage.get_results(result.run_id, passed=False)
        assert len(passed) + len(failed) == 6
        assert all(row.passed for row in passed)
        assert all(not row.passed for row in failed)

    def test_outcomes_projection(self, db_path: str, dataset_file) -> None:
        storage = Storage(db_path)
        result = run_result(str(dataset_file))
        storage.create_run(result.run_id, "test", "sim", str(dataset_file), {})
        storage.complete_run(result)
        outcomes = storage.get_outcomes(result.run_id)
        assert len(outcomes) == 6
        assert all(outcome.passed for outcome in outcomes)

    def test_results_for_unknown_run(self, db_path: str) -> None:
        storage = Storage(db_path)
        with pytest.raises(StorageError):
            storage.get_results("ghost")


class TestPipelineIntegration:
    def test_execute_run_persists_everything(self, db_path: str, dataset_file) -> None:
        storage = Storage(db_path)
        config = make_config(str(dataset_file))
        result = asyncio.run(execute_run(config, storage))
        record = storage.get_run(result.run_id)
        assert record.status == "completed"
        assert record.name == "test-run"

    def test_execute_run_marks_failure(self, db_path: str, dataset_file, monkeypatch) -> None:
        storage = Storage(db_path)
        config = make_config(str(dataset_file))

        async def boom(*args: object, **kwargs: object) -> None:
            raise RuntimeError("mid-run explosion")

        monkeypatch.setattr("evalpipe.pipeline.run_evaluation", boom)
        with pytest.raises(RuntimeError):
            asyncio.run(execute_run(config, storage, run_id="doomed"))
        record = storage.get_run("doomed")
        assert record.status == "failed"
        assert "mid-run explosion" in (record.error or "")

    def test_execute_run_bad_dataset_leaves_no_row(self, db_path: str) -> None:
        storage = Storage(db_path)
        config = make_config("missing-dataset.jsonl")
        from evalpipe.exceptions import DatasetError

        with pytest.raises(DatasetError):
            asyncio.run(execute_run(config, storage))
        assert storage.count_runs() == 0


class TestPrompts:
    def test_versioning_increments(self, db_path: str) -> None:
        storage = Storage(db_path)
        first = storage.save_prompt("qa", "v1: {prompt}")
        second = storage.save_prompt("qa", "v2: {prompt}")
        assert (first.version, second.version) == (1, 2)
        assert storage.get_prompt("qa").content == "v2: {prompt}"
        assert storage.get_prompt("qa", version=1).content == "v1: {prompt}"

    def test_list_returns_latest_only(self, db_path: str) -> None:
        storage = Storage(db_path)
        storage.save_prompt("a", "one {prompt}")
        storage.save_prompt("a", "two {prompt}")
        storage.save_prompt("b", "solo {prompt}")
        latest = storage.list_prompts()
        assert [(record.name, record.version) for record in latest] == [("a", 2), ("b", 1)]

    def test_placeholder_required(self, db_path: str) -> None:
        storage = Storage(db_path)
        with pytest.raises(StorageError, match="placeholder"):
            storage.save_prompt("bad", "no placeholder")

    def test_unknown_prompt(self, db_path: str) -> None:
        storage = Storage(db_path)
        with pytest.raises(StorageError, match="does not exist"):
            storage.get_prompt("ghost")
        with pytest.raises(StorageError, match="version 9"):
            storage.save_prompt("qa", "x {prompt}")
            storage.get_prompt("qa", version=9)
