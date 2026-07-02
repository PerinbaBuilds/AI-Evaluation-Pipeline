"""CLI behaviour, including the CI/CD quality-gate exit codes."""

from __future__ import annotations

import re
from pathlib import Path

import yaml
from tests.conftest import make_config
from typer.testing import CliRunner

from evalpipe.cli import app
from evalpipe.storage import Storage

runner = CliRunner()


def write_config(path: Path, dataset: Path, *, quality: float, name: str = "cli-run") -> Path:
    config = make_config(str(dataset), quality=quality, name=name)
    path.write_text(yaml.safe_dump(config.model_dump()), encoding="utf-8")
    return path


def test_version() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert re.match(r"evalpipe \d+\.\d+\.\d+", result.output)


def test_run_and_list(tmp_path: Path, dataset_file: Path, db_path: str) -> None:
    config_path = write_config(tmp_path / "c.yaml", dataset_file, quality=1.0)
    result = runner.invoke(app, ["run", str(config_path), "--db", db_path])
    assert result.exit_code == 0
    assert "pass rate:   100.0%" in result.output

    listing = runner.invoke(app, ["runs", "--db", db_path])
    assert listing.exit_code == 0
    assert "cli-run" in listing.output


def test_quality_gate_passes(tmp_path: Path, dataset_file: Path, db_path: str) -> None:
    config_path = write_config(tmp_path / "c.yaml", dataset_file, quality=1.0)
    result = runner.invoke(
        app, ["run", str(config_path), "--db", db_path, "--min-pass-rate", "0.9"]
    )
    assert result.exit_code == 0
    assert "Quality gate passed" in result.output


def test_quality_gate_fails_with_exit_code_1(
    tmp_path: Path, dataset_file: Path, db_path: str
) -> None:
    config_path = write_config(tmp_path / "c.yaml", dataset_file, quality=0.0)
    result = runner.invoke(
        app, ["run", str(config_path), "--db", db_path, "--min-pass-rate", "0.5"]
    )
    assert result.exit_code == 1
    assert "Quality gate FAILED" in result.output


def test_run_with_missing_config_exits_2(db_path: str) -> None:
    result = runner.invoke(app, ["run", "missing.yaml", "--db", db_path])
    assert result.exit_code == 2


def test_validate_ok(dataset_file: Path) -> None:
    result = runner.invoke(app, ["validate", str(dataset_file)])
    assert result.exit_code == 0
    assert "OK" in result.output


def test_validate_broken_dataset(tmp_path: Path) -> None:
    bad = tmp_path / "bad.jsonl"
    bad.write_text("{not json}\n", encoding="utf-8")
    result = runner.invoke(app, ["validate", str(bad)])
    assert result.exit_code == 2
    assert "ERROR" in result.output


def test_compare_flow(tmp_path: Path, dataset_file: Path, db_path: str) -> None:
    weak = write_config(tmp_path / "weak.yaml", dataset_file, quality=0.0, name="weak")
    strong = write_config(tmp_path / "strong.yaml", dataset_file, quality=1.0, name="strong")
    assert runner.invoke(app, ["run", str(weak), "--db", db_path]).exit_code == 0
    assert runner.invoke(app, ["run", str(strong), "--db", db_path]).exit_code == 0

    records = Storage(db_path).list_runs()
    ids = {record.name: record.id for record in records}
    result = runner.invoke(app, ["compare", ids["weak"], ids["strong"], "--db", db_path])
    assert result.exit_code == 0
    assert "verdict:" in result.output.lower()
    assert "pass rate" in result.output


def test_compare_unknown_runs_exits_2(db_path: str) -> None:
    result = runner.invoke(app, ["compare", "a", "b", "--db", db_path])
    assert result.exit_code == 2


def test_prompt_save_and_list(tmp_path: Path, db_path: str) -> None:
    template = tmp_path / "prompt.txt"
    template.write_text("Be concise.\n\nQ: {prompt}", encoding="utf-8")
    saved = runner.invoke(app, ["prompt", "save", "concise", str(template), "--db", db_path])
    assert saved.exit_code == 0
    assert "version 1" in saved.output

    listed = runner.invoke(app, ["prompt", "list", "--db", db_path])
    assert listed.exit_code == 0
    assert "concise" in listed.output


def test_prompt_save_missing_file_exits_2(db_path: str) -> None:
    result = runner.invoke(app, ["prompt", "save", "x", "nope.txt", "--db", db_path])
    assert result.exit_code == 2


def test_demo_seeds_history(tmp_path: Path, db_path: str) -> None:
    result = runner.invoke(
        app,
        ["demo", "--db", db_path, "--dataset-out", str(tmp_path / "demo.jsonl")],
    )
    assert result.exit_code == 0
    storage = Storage(db_path)
    assert storage.count_runs() == 7
    assert all(record.status == "completed" for record in storage.list_runs())
