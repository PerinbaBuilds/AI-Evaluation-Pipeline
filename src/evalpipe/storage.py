"""SQLite persistence for runs, per-item results and versioned prompt templates.

A fresh connection is opened per operation (cheap for SQLite, and safe across
threads and event loops), with WAL journaling and foreign keys enforced. All
queries are parameterised.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evalpipe.ab import ItemOutcome
from evalpipe.evaluators.base import EvalScore
from evalpipe.exceptions import StorageError
from evalpipe.runner import ItemResult, RunResult

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id                TEXT PRIMARY KEY,
    name              TEXT NOT NULL,
    model             TEXT NOT NULL,
    dataset           TEXT NOT NULL,
    status            TEXT NOT NULL CHECK (status IN ('running', 'completed', 'failed')),
    config_json       TEXT NOT NULL,
    started_at        TEXT NOT NULL,
    finished_at       TEXT,
    error             TEXT,
    item_count        INTEGER NOT NULL DEFAULT 0,
    error_count       INTEGER NOT NULL DEFAULT 0,
    pass_rate         REAL NOT NULL DEFAULT 0,
    mean_score        REAL NOT NULL DEFAULT 0,
    total_cost_usd    REAL NOT NULL DEFAULT 0,
    latency_p50_ms    REAL NOT NULL DEFAULT 0,
    latency_p95_ms    REAL NOT NULL DEFAULT 0,
    evaluator_means   TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS results (
    run_id     TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    item_id    TEXT NOT NULL,
    prompt     TEXT NOT NULL,
    expected   TEXT,
    output     TEXT NOT NULL,
    passed     INTEGER NOT NULL,
    mean_score REAL NOT NULL,
    scores     TEXT NOT NULL,
    latency_ms REAL NOT NULL,
    cost_usd   REAL NOT NULL,
    attempts   INTEGER NOT NULL,
    error      TEXT,
    metadata   TEXT NOT NULL DEFAULT '{}',
    PRIMARY KEY (run_id, item_id)
);

CREATE INDEX IF NOT EXISTS idx_results_run ON results(run_id);

CREATE TABLE IF NOT EXISTS prompts (
    name       TEXT NOT NULL,
    version    INTEGER NOT NULL,
    content    TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (name, version)
);

CREATE TABLE IF NOT EXISTS response_cache (
    key           TEXT PRIMARY KEY,
    model         TEXT NOT NULL,
    output        TEXT NOT NULL,
    input_tokens  INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    created_at    TEXT NOT NULL
);
"""


@dataclass(frozen=True)
class RunRecord:
    id: str
    name: str
    model: str
    dataset: str
    status: str
    started_at: str
    finished_at: str | None
    error: str | None
    item_count: int
    error_count: int
    pass_rate: float
    mean_score: float
    total_cost_usd: float
    latency_p50_ms: float
    latency_p95_ms: float
    evaluator_means: dict[str, float]


@dataclass(frozen=True)
class PromptRecord:
    name: str
    version: int
    content: str
    created_at: str


class Storage:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if self.path.parent != Path():
            self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("PRAGMA journal_mode = WAL")
            yield conn
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ------------------------------------------------------------------------- runs

    def create_run(
        self, run_id: str, name: str, model: str, dataset: str, config: dict[str, Any]
    ) -> None:
        with self._connect() as conn:
            try:
                conn.execute(
                    "INSERT INTO runs (id, name, model, dataset, status, config_json, started_at)"
                    " VALUES (?, ?, ?, ?, 'running', ?, ?)",
                    (run_id, name, model, dataset, json.dumps(config), _now()),
                )
            except sqlite3.IntegrityError as exc:
                raise StorageError(f"run {run_id!r} already exists") from exc

    def complete_run(self, run: RunResult) -> None:
        rows = [
            (
                run.run_id,
                item.item_id,
                item.prompt,
                item.expected,
                item.output,
                int(item.passed),
                item.mean_score,
                json.dumps([asdict(score) for score in item.scores]),
                item.latency_ms,
                item.cost_usd,
                item.attempts,
                item.error,
                json.dumps(item.metadata),
            )
            for item in run.items
        ]
        with self._connect() as conn:
            updated = conn.execute(
                "UPDATE runs SET status = 'completed', finished_at = ?, item_count = ?,"
                " error_count = ?, pass_rate = ?, mean_score = ?, total_cost_usd = ?,"
                " latency_p50_ms = ?, latency_p95_ms = ?, evaluator_means = ? WHERE id = ?",
                (
                    run.finished_at.isoformat(),
                    run.item_count,
                    run.error_count,
                    run.pass_rate,
                    run.mean_score,
                    run.total_cost_usd,
                    run.latency_p50_ms,
                    run.latency_p95_ms,
                    json.dumps(run.evaluator_means()),
                    run.run_id,
                ),
            ).rowcount
            if updated == 0:
                raise StorageError(f"run {run.run_id!r} does not exist")
            conn.executemany(
                "INSERT INTO results (run_id, item_id, prompt, expected, output, passed,"
                " mean_score, scores, latency_ms, cost_usd, attempts, error, metadata)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                rows,
            )

    def fail_run(self, run_id: str, error: str) -> None:
        with self._connect() as conn:
            updated = conn.execute(
                "UPDATE runs SET status = 'failed', finished_at = ?, error = ? WHERE id = ?",
                (_now(), error, run_id),
            ).rowcount
        if updated == 0:
            raise StorageError(f"run {run_id!r} does not exist")

    def get_run(self, run_id: str) -> RunRecord:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        if row is None:
            raise StorageError(f"run {run_id!r} does not exist")
        return _run_record(row)

    def list_runs(self, limit: int = 50, offset: int = 0) -> list[RunRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM runs ORDER BY started_at DESC, id LIMIT ? OFFSET ?",
                (max(1, min(limit, 500)), max(0, offset)),
            ).fetchall()
        return [_run_record(row) for row in rows]

    def count_runs(self) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS n FROM runs").fetchone()
        return int(row["n"])

    def delete_run(self, run_id: str) -> None:
        with self._connect() as conn:
            deleted = conn.execute("DELETE FROM runs WHERE id = ?", (run_id,)).rowcount
        if deleted == 0:
            raise StorageError(f"run {run_id!r} does not exist")

    # ---------------------------------------------------------------------- results

    def get_results(
        self,
        run_id: str,
        *,
        passed: bool | None = None,
        limit: int = 200,
        offset: int = 0,
    ) -> list[ItemResult]:
        self.get_run(run_id)  # raise cleanly on unknown run
        query = "SELECT * FROM results WHERE run_id = ?"
        params: list[Any] = [run_id]
        if passed is not None:
            query += " AND passed = ?"
            params.append(int(passed))
        query += " ORDER BY item_id LIMIT ? OFFSET ?"
        params.extend([max(1, min(limit, 1000)), max(0, offset)])
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [_item_result(row) for row in rows]

    def get_outcomes(self, run_id: str) -> list[ItemOutcome]:
        """Slim projection used by the A/B comparison."""
        self.get_run(run_id)
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT item_id, passed, mean_score FROM results WHERE run_id = ? ORDER BY item_id",
                (run_id,),
            ).fetchall()
        return [
            ItemOutcome(item_id=row["item_id"], passed=bool(row["passed"]), score=row["mean_score"])
            for row in rows
        ]

    # ---------------------------------------------------------------------- prompts

    def save_prompt(self, name: str, content: str) -> PromptRecord:
        """Store a new version of a named prompt template (versions are append-only)."""
        if not name.strip():
            raise StorageError("prompt name must not be blank")
        if "{prompt}" not in content:
            raise StorageError("prompt template must contain the {prompt} placeholder")
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COALESCE(MAX(version), 0) AS v FROM prompts WHERE name = ?", (name,)
            ).fetchone()
            version = int(row["v"]) + 1
            created_at = _now()
            conn.execute(
                "INSERT INTO prompts (name, version, content, created_at) VALUES (?, ?, ?, ?)",
                (name, version, content, created_at),
            )
        return PromptRecord(name=name, version=version, content=content, created_at=created_at)

    def get_prompt(self, name: str, version: int | None = None) -> PromptRecord:
        """Fetch a specific version, or the latest when ``version`` is None."""
        query = "SELECT * FROM prompts WHERE name = ?"
        params: list[Any] = [name]
        if version is not None:
            query += " AND version = ?"
            params.append(version)
        query += " ORDER BY version DESC LIMIT 1"
        with self._connect() as conn:
            row = conn.execute(query, params).fetchone()
        if row is None:
            suffix = f" (version {version})" if version is not None else ""
            raise StorageError(f"prompt {name!r}{suffix} does not exist")
        return PromptRecord(
            name=row["name"],
            version=row["version"],
            content=row["content"],
            created_at=row["created_at"],
        )

    def list_prompts(self) -> list[PromptRecord]:
        """Latest version of every named prompt."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT p.* FROM prompts p"
                " JOIN (SELECT name, MAX(version) AS v FROM prompts GROUP BY name) latest"
                " ON p.name = latest.name AND p.version = latest.v ORDER BY p.name"
            ).fetchall()
        return [
            PromptRecord(
                name=row["name"],
                version=row["version"],
                content=row["content"],
                created_at=row["created_at"],
            )
            for row in rows
        ]

    # ---------------------------------------------------------------- response cache

    def get_cached_response(self, key: str) -> tuple[str, int, int] | None:
        """Return ``(output, input_tokens, output_tokens)`` for a cache key, or None."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT output, input_tokens, output_tokens FROM response_cache WHERE key = ?",
                (key,),
            ).fetchone()
        if row is None:
            return None
        return (row["output"], row["input_tokens"], row["output_tokens"])

    def put_cached_response(
        self, key: str, model: str, output: str, input_tokens: int, output_tokens: int
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO response_cache"
                " (key, model, output, input_tokens, output_tokens, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (key, model, output, input_tokens, output_tokens, _now()),
            )

    def cache_size(self) -> int:
        with self._connect() as conn:
            return int(conn.execute("SELECT COUNT(*) AS n FROM response_cache").fetchone()["n"])

    # ------------------------------------------------------------------------ metrics

    def metrics(self) -> dict[str, float]:
        """Aggregate counters for the Prometheus exposition endpoint."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT status, COUNT(*) AS n, COALESCE(SUM(item_count), 0) AS items,"
                " COALESCE(SUM(total_cost_usd), 0) AS cost FROM runs GROUP BY status"
            ).fetchall()
            prompts = conn.execute("SELECT COUNT(*) AS n FROM prompts").fetchone()["n"]
            cache = conn.execute("SELECT COUNT(*) AS n FROM response_cache").fetchone()["n"]
        by_status = {row["status"]: row for row in rows}
        completed = by_status.get("completed")
        return {
            "runs_total": float(sum(row["n"] for row in rows)),
            "runs_completed": float(by_status["completed"]["n"])
            if "completed" in by_status
            else 0.0,
            "runs_failed": float(by_status["failed"]["n"]) if "failed" in by_status else 0.0,
            "runs_running": float(by_status["running"]["n"]) if "running" in by_status else 0.0,
            "items_evaluated_total": float(completed["items"]) if completed else 0.0,
            "estimated_cost_usd_total": float(completed["cost"]) if completed else 0.0,
            "prompts_total": float(prompts),
            "response_cache_size": float(cache),
        }


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _run_record(row: sqlite3.Row) -> RunRecord:
    return RunRecord(
        id=row["id"],
        name=row["name"],
        model=row["model"],
        dataset=row["dataset"],
        status=row["status"],
        started_at=row["started_at"],
        finished_at=row["finished_at"],
        error=row["error"],
        item_count=row["item_count"],
        error_count=row["error_count"],
        pass_rate=row["pass_rate"],
        mean_score=row["mean_score"],
        total_cost_usd=row["total_cost_usd"],
        latency_p50_ms=row["latency_p50_ms"],
        latency_p95_ms=row["latency_p95_ms"],
        evaluator_means=json.loads(row["evaluator_means"]),
    )


def _item_result(row: sqlite3.Row) -> ItemResult:
    scores = tuple(EvalScore(**payload) for payload in json.loads(row["scores"]))
    return ItemResult(
        item_id=row["item_id"],
        prompt=row["prompt"],
        expected=row["expected"],
        output=row["output"],
        scores=scores,
        passed=bool(row["passed"]),
        mean_score=row["mean_score"],
        latency_ms=row["latency_ms"],
        cost_usd=row["cost_usd"],
        attempts=row["attempts"],
        error=row["error"],
        metadata=json.loads(row["metadata"]),
    )
