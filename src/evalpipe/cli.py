"""Command-line interface.

Exit codes are part of the contract so runs can gate CI/CD pipelines:

- ``0`` success (and quality gate met, when one is set)
- ``1`` the run finished but the ``--min-pass-rate`` quality gate failed
- ``2`` the command could not complete (bad config, missing dataset, ...)
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import typer

from evalpipe import __version__
from evalpipe.ab import ABReport
from evalpipe.config import load_config
from evalpipe.datasets import validate_dataset
from evalpipe.exceptions import EvalPipeError
from evalpipe.pipeline import execute_run
from evalpipe.runner import RunResult
from evalpipe.storage import Storage

app = typer.Typer(
    name="evalpipe",
    help="Evaluation pipeline for LLM applications: run benchmarks, compare models, serve reports.",
    no_args_is_help=True,
    add_completion=False,
)

_DEFAULT_DB = "evalpipe.db"


def _db_path(db: str | None) -> str:
    return db or os.environ.get("EVALPIPE_DB", _DEFAULT_DB)


def _fail(message: str) -> None:
    typer.secho(f"error: {message}", fg=typer.colors.RED, err=True)
    raise typer.Exit(code=2)


@app.callback()
def _main() -> None:
    """EvalPipe — provider-agnostic LLM evaluation."""


@app.command()
def version() -> None:
    """Print the installed version."""
    typer.echo(f"evalpipe {__version__}")


@app.command()
def run(
    config_path: Path = typer.Argument(..., help="Path to an evaluation config (YAML)."),
    db: str | None = typer.Option(
        None, "--db", help="SQLite database path (default: $EVALPIPE_DB or ./evalpipe.db)."
    ),
    min_pass_rate: float | None = typer.Option(
        None,
        "--min-pass-rate",
        min=0.0,
        max=1.0,
        help="Quality gate: exit 1 when the run's pass rate is below this value.",
    ),
) -> None:
    """Run an evaluation described by CONFIG_PATH and persist the results."""
    try:
        config = load_config(config_path)
        storage = Storage(_db_path(db))

        def progress(done: int, total: int) -> None:
            typer.echo(f"\r  evaluating {done}/{total} items", nl=False)

        result = asyncio.run(execute_run(config, storage, progress=progress))
    except EvalPipeError as exc:
        _fail(str(exc))
        return
    typer.echo()  # newline after the progress line
    _print_run_summary(result)
    typer.echo(f"\nStored as run {result.run_id} in {_db_path(db)}")

    if min_pass_rate is not None and result.pass_rate < min_pass_rate:
        typer.secho(
            f"\nQuality gate FAILED: pass rate {result.pass_rate:.1%}"
            f" < required {min_pass_rate:.1%}",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1)
    if min_pass_rate is not None:
        typer.secho(
            f"\nQuality gate passed: pass rate {result.pass_rate:.1%} >= {min_pass_rate:.1%}",
            fg=typer.colors.GREEN,
        )


@app.command()
def compare(
    baseline_run_id: str = typer.Argument(..., help="Run id of the baseline (A)."),
    candidate_run_id: str = typer.Argument(..., help="Run id of the candidate (B)."),
    db: str | None = typer.Option(None, "--db"),
    alpha: float = typer.Option(0.05, "--alpha", min=0.001, max=0.5, help="Significance level."),
) -> None:
    """A/B comparison of two stored runs with significance testing."""
    from evalpipe import ab

    try:
        storage = Storage(_db_path(db))
        baseline_run = storage.get_run(baseline_run_id)
        candidate_run = storage.get_run(candidate_run_id)
        report = ab.compare(
            storage.get_outcomes(baseline_run_id),
            storage.get_outcomes(candidate_run_id),
            baseline_name=f"{baseline_run.name} ({baseline_run.model})",
            candidate_name=f"{candidate_run.name} ({candidate_run.model})",
            alpha=alpha,
        )
    except (EvalPipeError, ValueError) as exc:
        _fail(str(exc))
        return
    _print_ab_report(report)


@app.command()
def validate(
    dataset_path: Path = typer.Argument(..., help="Dataset file (.jsonl or .csv) to validate."),
) -> None:
    """Validate a dataset file and report problems without running anything."""
    report = validate_dataset(dataset_path)
    for error in report.errors:
        typer.secho(f"ERROR   {error}", fg=typer.colors.RED)
    for warning in report.warnings:
        typer.secho(f"WARNING {warning}", fg=typer.colors.YELLOW)
    if report.ok:
        typer.secho(f"OK      {report.item_count} valid item(s)", fg=typer.colors.GREEN)
    else:
        raise typer.Exit(code=2)


@app.command()
def runs(
    db: str | None = typer.Option(None, "--db"),
    limit: int = typer.Option(20, "--limit", min=1, max=500),
) -> None:
    """List stored runs, most recent first."""
    storage = Storage(_db_path(db))
    records = storage.list_runs(limit=limit)
    if not records:
        typer.echo("no runs stored yet — try `evalpipe demo` or `evalpipe run <config>`")
        return
    header = f"{'RUN ID':<34}{'NAME':<22}{'MODEL':<16}{'STATUS':<11}{'PASS':>7}{'SCORE':>8}"
    typer.echo(header)
    typer.echo("-" * len(header))
    for record in records:
        typer.echo(
            f"{record.id:<34}{record.name[:20]:<22}{record.model[:14]:<16}"
            f"{record.status:<11}{record.pass_rate:>6.1%}{record.mean_score:>8.3f}"
        )


@app.command()
def demo(
    db: str | None = typer.Option(None, "--db"),
    dataset_out: str = typer.Option(
        "demo_dataset.jsonl", "--dataset-out", help="Where to write the generated demo dataset."
    ),
) -> None:
    """Seed the database with a deterministic offline demo history (no API keys needed)."""
    from evalpipe.demo import seed_demo

    storage = Storage(_db_path(db))
    typer.echo("Seeding demo evaluation history (fully offline, deterministic)...")
    run_ids = asyncio.run(seed_demo(storage, dataset_out))
    typer.secho(f"Created {len(run_ids)} runs in {_db_path(db)}", fg=typer.colors.GREEN)
    typer.echo(f"A/B pair for comparison:\n  baseline:  {run_ids[-2]}\n  candidate: {run_ids[-1]}")
    typer.echo("Start the dashboard with: evalpipe serve")


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8000, "--port", min=1, max=65535),
    db: str | None = typer.Option(None, "--db"),
    seed_demo: bool = typer.Option(
        False,
        "--seed-demo/--no-seed-demo",
        envvar="EVALPIPE_SEED_DEMO",
        help="Seed the offline demo history on startup if the database is empty "
        "(useful for a fresh cloud deployment so the dashboard shows data).",
    ),
) -> None:
    """Serve the REST API and reporting dashboard."""
    import uvicorn

    from evalpipe.demo import seed_demo as seed_demo_history
    from evalpipe.server.app import create_app

    resolved_db = _db_path(db)
    if seed_demo:
        storage = Storage(resolved_db)
        if storage.count_runs() == 0:
            dataset_path = str(Path(resolved_db).resolve().parent / "demo_dataset.jsonl")
            typer.echo("Seeding demo history on first start...")
            asyncio.run(seed_demo_history(storage, dataset_path))

    uvicorn.run(create_app(resolved_db), host=host, port=port, log_level="info")


prompt_app = typer.Typer(help="Manage versioned prompt templates.", no_args_is_help=True)
app.add_typer(prompt_app, name="prompt")


@prompt_app.command("save")
def prompt_save(
    name: str = typer.Argument(..., help="Prompt template name."),
    file: Path = typer.Argument(..., help="File containing the template (must include {prompt})."),
    db: str | None = typer.Option(None, "--db"),
) -> None:
    """Save a new version of a named prompt template."""
    if not file.exists():
        _fail(f"file not found: {file}")
    try:
        record = Storage(_db_path(db)).save_prompt(name, file.read_text(encoding="utf-8"))
    except EvalPipeError as exc:
        _fail(str(exc))
        return
    typer.secho(f"saved prompt {record.name!r} version {record.version}", fg=typer.colors.GREEN)


@prompt_app.command("list")
def prompt_list(db: str | None = typer.Option(None, "--db")) -> None:
    """List the latest version of every stored prompt template."""
    records = Storage(_db_path(db)).list_prompts()
    if not records:
        typer.echo("no prompts stored yet — try `evalpipe prompt save <name> <file>`")
        return
    for record in records:
        preview = record.content.replace("\n", " ")[:60]
        typer.echo(f"{record.name}  v{record.version}  {record.created_at}  {preview!r}")


def _print_run_summary(result: RunResult) -> None:
    typer.echo(f"\nRun: {result.name}  |  model: {result.model}  |  items: {result.item_count}")
    typer.echo(f"  pass rate:   {result.pass_rate:.1%}")
    typer.echo(f"  mean score:  {result.mean_score:.3f}")
    typer.echo(f"  errors:      {result.error_count}")
    typer.echo(
        f"  latency:     p50 {result.latency_p50_ms:.0f} ms, p95 {result.latency_p95_ms:.0f} ms"
    )
    typer.echo(f"  est. cost:   ${result.total_cost_usd:.4f}")
    typer.echo("  per-evaluator mean scores:")
    for name, value in result.evaluator_means().items():
        typer.echo(f"    {name:<28}{value:.3f}")


def _print_ab_report(report: ABReport) -> None:
    z = report.pass_rate_test
    t = report.score_test
    typer.echo(f"\nA/B comparison over {report.n_common} shared items (alpha={report.alpha})")
    typer.echo(f"  A (baseline):  {report.baseline_name}")
    typer.echo(f"  B (candidate): {report.candidate_name}\n")
    typer.echo(
        f"  pass rate   A {z.proportion_a:.1%}  ->  B {z.proportion_b:.1%}"
        f"   diff {z.diff:+.1%}   z={z.z_statistic:.3f}  p={z.p_value:.4f}"
        f"   CI [{z.ci_low:+.1%}, {z.ci_high:+.1%}]   Cohen's h={z.cohen_h:.3f}"
    )
    typer.echo(
        f"  mean score  A {t.mean_a:.3f}  ->  B {t.mean_b:.3f}"
        f"   diff {t.diff:+.3f}   t={t.t_statistic:.3f}  p={t.p_value:.4f}"
        f"   CI [{t.ci_low:+.3f}, {t.ci_high:+.3f}]   Cohen's d={t.cohen_d:.3f}"
    )
    low, high = report.score_bootstrap_ci
    typer.echo(f"  bootstrap score-diff CI [{low:+.3f}, {high:+.3f}]")
    for warning in report.warnings:
        typer.secho(f"  warning: {warning}", fg=typer.colors.YELLOW)
    color = {
        "candidate_better": typer.colors.GREEN,
        "baseline_better": typer.colors.RED,
        "inconclusive": typer.colors.YELLOW,
    }[report.verdict]
    typer.secho(f"\n  verdict: {report.verdict.replace('_', ' ').upper()}", fg=color, bold=True)


if __name__ == "__main__":  # pragma: no cover
    app()
