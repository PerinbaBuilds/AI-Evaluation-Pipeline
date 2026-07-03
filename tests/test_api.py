"""REST API and dashboard pages, exercised through the ASGI test client."""

from __future__ import annotations

import asyncio
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
        assert "Run side by side" in response.text

    def test_static_assets_served(self, client: TestClient) -> None:
        assert client.get("/static/styles.css").status_code == 200
        assert client.get("/static/charts.js").status_code == 200
