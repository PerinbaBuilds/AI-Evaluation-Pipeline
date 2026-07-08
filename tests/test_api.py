"""REST API and dashboard pages, exercised through the ASGI test client."""

from __future__ import annotations

import asyncio
import json
import time

import pytest
from fastapi.testclient import TestClient

from evalpipe.pipeline import execute_run
from evalpipe.server.app import create_app
from evalpipe.storage import Storage
from tests.conftest import make_config


@pytest.fixture
def seeded(db_path: str, dataset_file) -> dict[str, str]:
    """Two completed runs (a weaker baseline and a stronger candidate) plus the app."""
    storage = Storage(db_path)
    baseline = asyncio.run(
        execute_run(make_config(str(dataset_file), name="base", quality=0.3, seed=11), storage)
    )
    candidate = asyncio.run(
        execute_run(make_config(str(dataset_file), name="cand", quality=1.0, seed=12), storage)
    )
    return {"db": db_path, "baseline": baseline.run_id, "candidate": candidate.run_id}


@pytest.fixture
def client(seeded: dict[str, str]):
    app = create_app(seeded["db"])
    with TestClient(app) as test_client:
        yield test_client


class TestApi:
    def test_health(self, client: TestClient) -> None:
        body = client.get("/api/health").json()
        assert body["status"] == "ok"
        assert body["version"]

    def test_list_runs(self, client: TestClient) -> None:
        body = client.get("/api/runs").json()
        assert body["total"] == 2
        assert {run["name"] for run in body["runs"]} == {"base", "cand"}

    def test_get_run(self, client: TestClient, seeded: dict[str, str]) -> None:
        body = client.get(f"/api/runs/{seeded['candidate']}").json()
        assert body["status"] == "completed"
        assert body["pass_rate"] == 1.0

    def test_runs_expose_provider_type(self, client: TestClient) -> None:
        # runs carry the provider kind (derived from stored config) so the UI can
        # flag simulated (offline) runs; the seeded runs use the mock provider.
        body = client.get("/api/runs").json()
        assert body["runs"]
        assert all(run["provider_type"] == "mock" for run in body["runs"])

    def test_get_unknown_run_is_404(self, client: TestClient) -> None:
        assert client.get("/api/runs/ghost").status_code == 404

    def test_get_results_with_filter(self, client: TestClient, seeded: dict[str, str]) -> None:
        all_rows = client.get(f"/api/runs/{seeded['baseline']}/results").json()["results"]
        failed = client.get(
            f"/api/runs/{seeded['baseline']}/results", params={"passed": False}
        ).json()["results"]
        assert len(all_rows) == 6
        assert all(not row["passed"] for row in failed)

    def test_compare_endpoint(self, client: TestClient, seeded: dict[str, str]) -> None:
        body = client.get(
            "/api/compare",
            params={"baseline": seeded["baseline"], "candidate": seeded["candidate"]},
        ).json()
        assert body["n_common"] == 6
        assert body["verdict"] in {"candidate_better", "inconclusive"}
        assert 0.0 <= body["pass_rate_test"]["p_value"] <= 1.0

    def test_compare_unknown_run(self, client: TestClient, seeded: dict[str, str]) -> None:
        response = client.get(
            "/api/compare", params={"baseline": "ghost", "candidate": seeded["candidate"]}
        )
        assert response.status_code == 404

    def test_compare_includes_regression_diff(
        self, client: TestClient, seeded: dict[str, str]
    ) -> None:
        body = client.get(
            "/api/compare",
            params={"baseline": seeded["baseline"], "candidate": seeded["candidate"]},
        ).json()
        # weak baseline (quality 0.3) -> strong candidate (quality 1.0): net improvements
        assert body["n_improvements"] >= body["n_regressions"]
        assert isinstance(body["regressed_ids"], list)
        assert isinstance(body["improved_ids"], list)

    def test_metrics_endpoint_prometheus_format(self, client: TestClient) -> None:
        response = client.get("/metrics")
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/plain")
        body = response.text
        assert "evalpipe_runs_total" in body
        assert "evalpipe_runs_completed 2" in body
        assert "# TYPE evalpipe_runs_total gauge" in body

    def test_export_csv(self, client: TestClient, seeded: dict[str, str]) -> None:
        response = client.get(f"/api/runs/{seeded['candidate']}/export", params={"format": "csv"})
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/csv")
        assert "attachment" in response.headers["content-disposition"]
        lines = response.text.strip().splitlines()
        assert lines[0].startswith("item_id,passed,mean_score")
        assert len(lines) == 7  # header + 6 items

    def test_export_json(self, client: TestClient, seeded: dict[str, str]) -> None:
        response = client.get(f"/api/runs/{seeded['candidate']}/export", params={"format": "json"})
        assert response.status_code == 200
        body = response.json()
        assert body["run"]["status"] == "completed"
        assert len(body["results"]) == 6

    def test_export_bad_format_is_422(self, client: TestClient, seeded: dict[str, str]) -> None:
        response = client.get(f"/api/runs/{seeded['candidate']}/export", params={"format": "xml"})
        assert response.status_code == 422

    def test_export_unknown_run_is_404(self, client: TestClient) -> None:
        assert client.get("/api/runs/ghost/export").status_code == 404

    def test_slices_unknown_run_is_404(self, client: TestClient) -> None:
        assert client.get("/api/runs/ghost/slices").status_code == 404


def test_slices_endpoint_groups_by_metadata(tmp_path) -> None:
    dataset = tmp_path / "d.jsonl"
    rows = [
        json.dumps(
            {
                "id": f"i{n}",
                "prompt": f"q{n}",
                "expected": f"a{n}",
                "metadata": {"topic": "easy" if n < 5 else "hard"},
            }
        )
        for n in range(10)
    ]
    dataset.write_text("\n".join(rows), encoding="utf-8")

    db = str(tmp_path / "s.db")
    config = make_config(str(dataset), quality=1.0)
    run = asyncio.run(execute_run(config, Storage(db)))

    with TestClient(create_app(db)) as client:
        body = client.get(f"/api/runs/{run.run_id}/slices").json()
        assert "topic" in body["keys"]
        assert body["key"] == "topic"
        assert {s["value"] for s in body["slices"]} == {"easy", "hard"}
        assert all("pass_rate" in s and "pass_ci" in s for s in body["slices"])

    def test_create_run_executes_in_background(self, client: TestClient, dataset_file) -> None:
        config = make_config(str(dataset_file), name="via-api").model_dump()
        response = client.post("/api/runs", json=config)
        assert response.status_code == 202
        run_id = response.json()["run_id"]

        deadline = time.monotonic() + 10
        status = "running"
        while time.monotonic() < deadline:
            status = client.get(f"/api/runs/{run_id}").json()["status"]
            if status != "running":
                break
            time.sleep(0.05)
        assert status == "completed"
        results = client.get(f"/api/runs/{run_id}/results").json()["results"]
        assert len(results) == 6

    def test_create_run_with_bad_dataset_is_400(self, client: TestClient) -> None:
        config = make_config("missing.jsonl").model_dump()
        response = client.post("/api/runs", json=config)
        assert response.status_code == 400
        assert "not found" in response.json()["detail"]

    def test_create_run_with_invalid_config_is_422(self, client: TestClient) -> None:
        response = client.post("/api/runs", json={"name": "x"})
        assert response.status_code == 422

    def test_playground_side_by_side(self, client: TestClient) -> None:
        response = client.post(
            "/api/playground",
            json={
                "prompt": "Capital of France?",
                "reference": "Paris.",
                "providers": [
                    {"type": "mock", "model": "sim-a", "quality": 1.0},
                    {"type": "mock", "model": "sim-b", "quality": 1.0},
                ],
            },
        )
        assert response.status_code == 200
        results = response.json()["results"]
        assert len(results) == 2
        assert results[0]["output"] == "Paris."
        assert results[0]["latency_ms"] >= 0.0

    def test_playground_scores_outputs_with_metrics(self, client: TestClient) -> None:
        response = client.post(
            "/api/playground",
            json={
                "prompt": "Capital of France?",
                "reference": "Paris.",
                "providers": [
                    {"type": "mock", "model": "sim-a", "quality": 1.0},
                    {"type": "mock", "model": "sim-b", "quality": 0.0, "seed": 5},
                ],
                "evaluators": [
                    {"type": "exact_match", "strip_punctuation": True},
                    {"type": "token_f1"},
                ],
            },
        )
        assert response.status_code == 200
        results = response.json()["results"]
        assert len(results) == 2
        for result in results:
            assert len(result["scores"]) == 2
            assert result["passed"] is not None
            assert result["mean_score"] is not None
        # quality 1.0 returns the reference verbatim -> exact match passes
        assert results[0]["passed"] is True
        assert results[0]["provider_type"] == "mock"

    def test_playground_without_metrics_has_no_scores(self, client: TestClient) -> None:
        response = client.post(
            "/api/playground",
            json={
                "prompt": "Q?",
                "providers": [{"type": "mock", "model": "m", "quality": 1.0}],
            },
        )
        result = response.json()["results"][0]
        assert result["scores"] == []
        assert result["passed"] is None

    def test_integrations_status(self, client: TestClient, monkeypatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-x")
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        body = client.get("/api/integrations").json()
        by_type = {p["type"]: p for p in body["providers"]}
        assert by_type["openai"]["configured"] is True
        assert by_type["anthropic"]["configured"] is False
        assert by_type["mock"]["configured"] is True  # no key needed
        # free options are listed; local ones need no key
        assert by_type["ollama"]["configured"] is True
        assert {"groq", "openrouter", "gemini"} <= set(by_type)

    def test_playground_provider_error_is_isolated(self, client: TestClient) -> None:
        response = client.post(
            "/api/playground",
            json={
                "prompt": "Q?",
                "providers": [
                    {"type": "mock", "model": "ok", "quality": 1.0},
                    {"type": "mock", "model": "broken", "failure_rate": 1.0},
                ],
            },
        )
        results = response.json()["results"]
        assert results[0]["error"] is None
        assert "simulated" in results[1]["error"]

    def test_prompt_crud(self, client: TestClient) -> None:
        created = client.post("/api/prompts", json={"name": "qa", "content": "Answer: {prompt}"})
        assert created.status_code == 201
        assert created.json()["version"] == 1
        listed = client.get("/api/prompts").json()["prompts"]
        assert listed[0]["name"] == "qa"
        fetched = client.get("/api/prompts/qa").json()
        assert fetched["content"] == "Answer: {prompt}"
        assert client.get("/api/prompts/ghost").status_code == 404

    def test_prompt_without_placeholder_is_400(self, client: TestClient) -> None:
        response = client.post("/api/prompts", json={"name": "bad", "content": "static"})
        assert response.status_code == 400


class TestPages:
    def test_dashboard_renders(self, client: TestClient) -> None:
        response = client.get("/")
        assert response.status_code == 200
        assert "Evaluation dashboard" in response.text
        assert "Pass rate over runs" in response.text
        # the real-time comparison widget is embedded on the dashboard itself
        assert "Compare models in real time" in response.text
        assert 'id="pg-form"' in response.text
        # static assets are cache-busted so a new build is never served stale
        assert "styles.css?v=" in response.text
        assert "playground.js?v=" in response.text
        # simulated (mock-provider) runs are flagged in the history
        assert 'class="tag-sim"' in response.text
        # the overview carries per-run charts, not just the trend line
        assert "Score distribution — latest run" in response.text
        assert "Mean score by evaluator — latest run" in response.text
        # light/dark theme toggle is present and initialised before paint
        assert 'id="theme-toggle"' in response.text
        assert "evalpipe-theme" in response.text
        assert 'data-theme' in response.text

    def test_self_hosted_font_is_served(self, client: TestClient) -> None:
        response = client.get("/static/fonts/ibm-plex-sans-latin-400-normal.woff2")
        assert response.status_code == 200
        assert response.headers["content-type"] == "font/woff2"

    def test_dashboard_comparison_available_when_empty(self, tmp_path) -> None:
        app = create_app(str(tmp_path / "empty.db"))
        with TestClient(app) as empty_client:
            response = empty_client.get("/")
        assert "No completed runs yet" in response.text
        assert "Compare models in real time" in response.text  # usable with zero runs

    def test_dashboard_empty_state(self, tmp_path) -> None:
        app = create_app(str(tmp_path / "empty.db"))
        with TestClient(app) as empty_client:
            response = empty_client.get("/")
        assert "No completed runs yet" in response.text

    def test_run_detail_renders(self, client: TestClient, seeded: dict[str, str]) -> None:
        response = client.get(f"/runs/{seeded['candidate']}")
        assert response.status_code == 200
        assert "Score distribution" in response.text
        assert "Mean score by evaluator" in response.text
        assert 'class="tag-sim"' in response.text  # mock-provider run is flagged

    def test_run_detail_outcome_filter(self, client: TestClient, seeded: dict[str, str]) -> None:
        response = client.get(f"/runs/{seeded['baseline']}", params={"outcome": "failed"})
        assert response.status_code == 200

    def test_run_detail_404(self, client: TestClient) -> None:
        assert client.get("/runs/ghost").status_code == 404

    def test_compare_page_with_report(self, client: TestClient, seeded: dict[str, str]) -> None:
        response = client.get(
            "/compare",
            params={"baseline": seeded["baseline"], "candidate": seeded["candidate"]},
        )
        assert response.status_code == 200
        assert "Statistical detail" in response.text

    def test_compare_page_without_selection(self, client: TestClient) -> None:
        response = client.get("/compare")
        assert response.status_code == 200
        assert "Select a completed run" in response.text

    def test_playground_page(self, client: TestClient) -> None:
        response = client.get("/playground")
        assert response.status_code == 200
        assert "Run comparison" in response.text
        assert "Anthropic" in response.text  # integrations strip lists the providers

    def test_static_assets_served(self, client: TestClient) -> None:
        assert client.get("/static/styles.css").status_code == 200
        assert client.get("/static/charts.js").status_code == 200
