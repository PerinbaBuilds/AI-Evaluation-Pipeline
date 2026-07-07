"""FastAPI application factory.

``create_app(db_path)`` wires storage into the REST API and the server-rendered
dashboard. Evaluation runs triggered over HTTP execute as background tasks; the
run row is created immediately so the dashboard shows it as *running*.
"""

from __future__ import annotations

import asyncio
import contextlib
import csv
import io
import json
import math
import os
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from evalpipe import __version__, ab, slices
from evalpipe.config import EvalConfig
from evalpipe.datasets import DatasetItem, load_dataset
from evalpipe.evaluators import build_evaluators
from evalpipe.exceptions import EvalPipeError, ProviderError, StorageError
from evalpipe.pipeline import build_run_provider
from evalpipe.providers import build_provider
from evalpipe.runner import ItemResult
from evalpipe.server.schemas import (
    HealthResponse,
    PlaygroundRequest,
    PlaygroundResponse,
    PlaygroundResult,
    PlaygroundScore,
    PromptSaveRequest,
    RunCreatedResponse,
)
from evalpipe.stats import mean, sample_variance, wilson_interval
from evalpipe.storage import RunRecord, Storage

_BASE_DIR = Path(__file__).parent


def create_app(db_path: str = "evalpipe.db") -> FastAPI:
    app = FastAPI(
        title="EvalPipe",
        version=__version__,
        description="Provider-agnostic evaluation pipeline for LLM applications.",
    )
    storage = Storage(db_path)
    templates = Jinja2Templates(directory=str(_BASE_DIR / "templates"))
    app.mount("/static", StaticFiles(directory=str(_BASE_DIR / "static")), name="static")
    background_tasks: set[asyncio.Task[Any]] = set()

    # ------------------------------------------------------------------------ API

    @app.get("/api/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse(status="ok", version=__version__)

    @app.get("/metrics", response_class=PlainTextResponse, include_in_schema=False)
    async def metrics() -> str:
        return _prometheus_exposition(storage.metrics())

    @app.get("/api/runs")
    async def list_runs(
        limit: int = Query(50, ge=1, le=500), offset: int = Query(0, ge=0)
    ) -> dict[str, Any]:
        records = storage.list_runs(limit=limit, offset=offset)
        return {"total": storage.count_runs(), "runs": [asdict(record) for record in records]}

    @app.get("/api/runs/{run_id}")
    async def get_run(run_id: str) -> dict[str, Any]:
        return asdict(_run_or_404(storage, run_id))

    @app.get("/api/runs/{run_id}/results")
    async def get_results(
        run_id: str,
        passed: bool | None = Query(None),
        limit: int = Query(200, ge=1, le=1000),
        offset: int = Query(0, ge=0),
    ) -> dict[str, Any]:
        _run_or_404(storage, run_id)
        results = storage.get_results(run_id, passed=passed, limit=limit, offset=offset)
        return {"run_id": run_id, "results": [asdict(result) for result in results]}

    @app.get("/api/runs/{run_id}/export", include_in_schema=True)
    async def export_results(
        run_id: str, format: str = Query("csv", pattern="^(csv|json)$")
    ) -> Response:
        record = _run_or_404(storage, run_id)
        results = storage.get_results(run_id, limit=1000)
        if format == "json":
            payload = {"run": asdict(record), "results": [asdict(r) for r in results]}
            return Response(
                content=json.dumps(payload, indent=2),
                media_type="application/json",
                headers={"Content-Disposition": f'attachment; filename="{run_id}.json"'},
            )
        return Response(
            content=_results_to_csv(results),
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="{run_id}.csv"'},
        )

    @app.get("/api/runs/{run_id}/slices")
    async def get_slices(run_id: str, key: str = Query("")) -> dict[str, Any]:
        _run_or_404(storage, run_id)
        results = storage.get_results(run_id, limit=1000)
        keys = slices.sliceable_keys(results)
        active = key if key in keys else (keys[0] if keys else "")
        stats = slices.slice_by(results, active) if active else []
        return {
            "run_id": run_id,
            "keys": keys,
            "key": active,
            "slices": [asdict(stat) for stat in stats],
        }

    @app.post("/api/runs", response_model=RunCreatedResponse, status_code=202)
    async def create_run(config: EvalConfig) -> RunCreatedResponse:
        # Resolve inputs up front so obvious mistakes fail with 400, not silently
        # inside a background task.
        try:
            load_dataset(config.dataset)
            provider = build_provider(config.provider)
            build_evaluators(config.evaluators, provider)
            await provider.aclose()
        except EvalPipeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        run_id = uuid.uuid4().hex
        storage.create_run(
            run_id, config.name, config.provider.model, config.dataset, config.model_dump()
        )
        task = asyncio.create_task(_execute_prepared(config, storage, run_id))
        background_tasks.add(task)
        task.add_done_callback(background_tasks.discard)
        return RunCreatedResponse(run_id=run_id)

    @app.get("/api/compare")
    async def compare_runs(
        baseline: str, candidate: str, alpha: float = Query(0.05, gt=0.0, lt=0.5)
    ) -> dict[str, Any]:
        report = _build_ab_report(storage, baseline, candidate, alpha)
        return asdict(report)

    @app.post("/api/playground", response_model=PlaygroundResponse)
    async def playground(request: PlaygroundRequest) -> PlaygroundResponse:
        item = DatasetItem(
            id="playground", prompt=request.prompt, expected=request.reference or None
        )

        async def invoke(provider_config: Any) -> PlaygroundResult:
            ptype = getattr(provider_config, "type", "")
            try:
                provider = build_provider(provider_config)
            except EvalPipeError as exc:
                return PlaygroundResult(
                    model=provider_config.model, provider_type=ptype, error=str(exc)
                )
            start = time.perf_counter()
            try:
                response = await provider.generate(request.prompt, reference=request.reference)
            except ProviderError as exc:
                await provider.aclose()
                return PlaygroundResult(model=provider.model, provider_type=ptype, error=str(exc))
            latency_ms = (time.perf_counter() - start) * 1000.0
            cost = provider.estimate_cost_usd(response)

            scores: list[PlaygroundScore] = []
            if request.evaluators:
                evaluators = build_evaluators(request.evaluators, provider)
                for evaluator in evaluators:
                    verdict = await evaluator.evaluate(item, response.text)
                    scores.append(
                        PlaygroundScore(
                            name=verdict.name,
                            score=verdict.score,
                            passed=verdict.passed,
                            detail=verdict.detail,
                        )
                    )
            await provider.aclose()

            return PlaygroundResult(
                model=provider.model,
                provider_type=ptype,
                output=response.text,
                latency_ms=latency_ms,
                cost_usd=cost,
                passed=all(s.passed for s in scores) if scores else None,
                mean_score=(sum(s.score for s in scores) / len(scores)) if scores else None,
                scores=scores,
            )

        results = await asyncio.gather(*(invoke(config) for config in request.providers))
        return PlaygroundResponse(results=list(results))

    @app.get("/api/integrations")
    async def integrations() -> dict[str, Any]:
        return {"providers": _integration_status()}

    @app.get("/api/prompts")
    async def list_prompts() -> dict[str, Any]:
        return {"prompts": [asdict(record) for record in storage.list_prompts()]}

    @app.post("/api/prompts", status_code=201)
    async def save_prompt(request: PromptSaveRequest) -> dict[str, Any]:
        try:
            return asdict(storage.save_prompt(request.name, request.content))
        except StorageError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/prompts/{name}")
    async def get_prompt(name: str, version: int | None = Query(None, ge=1)) -> dict[str, Any]:
        try:
            return asdict(storage.get_prompt(name, version))
        except StorageError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    # ---------------------------------------------------------------------- pages

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def dashboard_page(request: Request) -> HTMLResponse:
        records = storage.list_runs(limit=100)
        completed = [record for record in records if record.status == "completed"]
        chronological = list(reversed(completed))
        latest = completed[0] if completed else None
        previous = completed[1] if len(completed) > 1 else None
        context = {
            "request": request,
            "records": records[:20],
            "latest": latest,
            "delta": (latest.pass_rate - previous.pass_rate) if latest and previous else None,
            "total_runs": storage.count_runs(),
            "total_items": sum(record.item_count for record in completed),
            "total_cost": sum(record.total_cost_usd for record in completed),
            "integrations": _integration_status(),
            "trend": [
                {
                    "label": f"{record.name} · {record.model}",
                    "value": round(record.pass_rate, 4),
                    "run_id": record.id,
                }
                for record in chronological
            ],
        }
        return templates.TemplateResponse(request, "dashboard.html", context)

    @app.get("/runs/{run_id}", response_class=HTMLResponse, include_in_schema=False)
    async def run_page(
        request: Request,
        run_id: str,
        outcome: str = Query("all"),
        slice_key: str = Query("", alias="slice"),
    ) -> HTMLResponse:
        record = _run_or_404(storage, run_id)
        passed_filter = {"passed": True, "failed": False}.get(outcome)
        results = storage.get_results(run_id, passed=passed_filter, limit=500)
        all_results = storage.get_results(run_id, limit=1000)
        scores = [result.mean_score for result in all_results]
        histogram = _histogram(scores, bins=10)
        pass_ci = (
            wilson_interval(sum(r.passed for r in all_results), len(all_results))
            if all_results
            else (0.0, 0.0)
        )

        keys = slices.sliceable_keys(all_results)
        active_key = slice_key if slice_key in keys else (keys[0] if keys else "")
        slice_stats = slices.slice_by(all_results, active_key) if active_key else []

        context = {
            "request": request,
            "run": record,
            "results": results,
            "outcome": outcome,
            "pass_ci": pass_ci,
            "histogram": histogram,
            "evaluator_means": [
                {"label": name, "value": round(value, 4)}
                for name, value in record.evaluator_means.items()
            ],
            "slice_keys": keys,
            "active_slice": active_key,
            "slice_stats": slice_stats,
            "slice_chart": {
                "bars": [{"label": s.value, "value": round(s.pass_rate, 4)} for s in slice_stats],
                "format": "percent",
                "max": 1,
            },
        }
        return templates.TemplateResponse(request, "run_detail.html", context)

    @app.get("/compare", response_class=HTMLResponse, include_in_schema=False)
    async def compare_page(
        request: Request, baseline: str | None = None, candidate: str | None = None
    ) -> HTMLResponse:
        completed = [
            record for record in storage.list_runs(limit=200) if record.status == "completed"
        ]
        context: dict[str, Any] = {
            "request": request,
            "completed_runs": completed,
            "baseline_id": baseline,
            "candidate_id": candidate,
            "report": None,
            "error": None,
        }
        if baseline and candidate:
            try:
                report = _build_ab_report(storage, baseline, candidate, alpha=0.05)
                context["report"] = report
                context["chart"] = _compare_chart_data(storage, baseline, candidate, report)
            except HTTPException as exc:
                context["error"] = str(exc.detail)
        return templates.TemplateResponse(request, "compare.html", context)

    @app.get("/playground", response_class=HTMLResponse, include_in_schema=False)
    async def playground_page(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(
            request, "playground.html", {"integrations": _integration_status()}
        )

    return app


async def _execute_prepared(config: EvalConfig, storage: Storage, run_id: str) -> None:
    """Background wrapper: failures are recorded on the run row, never raised."""
    with contextlib.suppress(Exception):
        await _run_without_recreating(config, storage, run_id)


async def _run_without_recreating(config: EvalConfig, storage: Storage, run_id: str) -> None:
    from evalpipe.runner import run_evaluation

    provider = build_run_provider(config, storage)
    try:
        items = load_dataset(config.dataset)
        evaluators = build_evaluators(config.evaluators, provider)
        result = await run_evaluation(config, items, provider, evaluators, run_id=run_id)
        storage.complete_run(result)
    except Exception as exc:
        storage.fail_run(run_id, f"{type(exc).__name__}: {exc}")
        raise
    finally:
        await provider.aclose()


def _run_or_404(storage: Storage, run_id: str) -> RunRecord:
    try:
        return storage.get_run(run_id)
    except StorageError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


def _build_ab_report(storage: Storage, baseline: str, candidate: str, alpha: float) -> ab.ABReport:
    baseline_run = _run_or_404(storage, baseline)
    candidate_run = _run_or_404(storage, candidate)
    try:
        return ab.compare(
            storage.get_outcomes(baseline),
            storage.get_outcomes(candidate),
            baseline_name=f"{baseline_run.name} ({baseline_run.model})",
            candidate_name=f"{candidate_run.name} ({candidate_run.model})",
            alpha=alpha,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


_CSV_COLUMNS = (
    "item_id",
    "passed",
    "mean_score",
    "latency_ms",
    "cost_usd",
    "attempts",
    "error",
    "prompt",
    "expected",
    "output",
    "scores",
)


def _results_to_csv(results: list[ItemResult]) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(_CSV_COLUMNS)
    for r in results:
        writer.writerow(
            [
                r.item_id,
                int(r.passed),
                f"{r.mean_score:.6f}",
                f"{r.latency_ms:.3f}",
                f"{r.cost_usd:.6f}",
                r.attempts,
                r.error or "",
                r.prompt,
                r.expected or "",
                r.output,
                json.dumps([asdict(s) for s in r.scores]),
            ]
        )
    return buffer.getvalue()


_INTEGRATIONS = (
    ("openai", "OpenAI · ChatGPT", "OPENAI_API_KEY"),
    ("anthropic", "Anthropic", "ANTHROPIC_API_KEY"),
    ("gemini", "Google Gemini · free tier", "GEMINI_API_KEY"),
    ("groq", "Groq · free tier", "GROQ_API_KEY"),
    ("openrouter", "OpenRouter · free models", "OPENROUTER_API_KEY"),
    ("ollama", "Ollama · local, free", None),
    ("openai_compatible", "OpenAI-compatible · proxy", None),
    ("mock", "Simulation · offline", None),
)


def _integration_status() -> list[dict[str, Any]]:
    """Which provider integrations are usable right now (env key present)."""
    status = []
    for ptype, label, env in _INTEGRATIONS:
        configured = env is None or bool(os.environ.get(env))
        status.append({"type": ptype, "label": label, "env": env, "configured": configured})
    return status


def _prometheus_exposition(metrics: dict[str, float]) -> str:
    """Render aggregate metrics in the Prometheus text exposition format."""
    lines: list[str] = []
    for name, value in metrics.items():
        metric = f"evalpipe_{name}"
        lines.append(f"# TYPE {metric} gauge")
        lines.append(f"{metric} {value:g}")
    return "\n".join(lines) + "\n"


def _histogram(scores: list[float], bins: int) -> list[dict[str, Any]]:
    counts = [0] * bins
    for score in scores:
        index = min(bins - 1, int(score * bins))
        counts[index] += 1
    return [
        {
            "label": f"{i / bins:.1f}-{(i + 1) / bins:.1f}",
            "value": count,
        }
        for i, count in enumerate(counts)
    ]


def _compare_chart_data(
    storage: Storage, baseline: str, candidate: str, report: ab.ABReport
) -> dict[str, Any]:
    """Grouped-bar data: pass rate and mean score per run, with CI whiskers."""

    def score_ci(run_id: str) -> tuple[float, float]:
        scores = [outcome.score for outcome in storage.get_outcomes(run_id)]
        if len(scores) < 2:
            value = scores[0] if scores else 0.0
            return (value, value)
        half_width = 1.96 * math.sqrt(sample_variance(scores) / len(scores))
        center = mean(scores)
        return (max(0.0, center - half_width), min(1.0, center + half_width))

    base_score_ci = score_ci(baseline)
    cand_score_ci = score_ci(candidate)
    z, t = report.pass_rate_test, report.score_test
    return {
        "series": ["Baseline", "Candidate"],
        "groups": [
            {
                "label": "Pass rate",
                "values": [round(z.proportion_a, 4), round(z.proportion_b, 4)],
                "ci": [
                    [round(v, 4) for v in report.baseline_pass_ci],
                    [round(v, 4) for v in report.candidate_pass_ci],
                ],
            },
            {
                "label": "Mean score",
                "values": [round(t.mean_a, 4), round(t.mean_b, 4)],
                "ci": [
                    [round(v, 4) for v in base_score_ci],
                    [round(v, 4) for v in cand_score_ci],
                ],
            },
        ],
    }
