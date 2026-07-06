"""Request/response models for the REST API."""

from __future__ import annotations

from pydantic import BaseModel, Field

from evalpipe.config import EvaluatorConfig, ProviderConfig


class HealthResponse(BaseModel):
    status: str
    version: str


class RunCreatedResponse(BaseModel):
    run_id: str
    status: str = "running"


class PromptSaveRequest(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    content: str = Field(min_length=1)


class PlaygroundScore(BaseModel):
    name: str
    score: float
    passed: bool
    detail: str = ""


class PlaygroundRequest(BaseModel):
    """Online evaluation: one prompt across providers, scored by our metric suite.

    When ``evaluators`` is provided, every provider's output is graded by the
    same evaluators (against ``reference`` where the metric needs one), so the
    real-time comparison shows output, latency and cost *and* per-metric scores.
    """

    prompt: str = Field(min_length=1, max_length=20_000)
    reference: str | None = None
    providers: list[ProviderConfig] = Field(min_length=1, max_length=4)
    evaluators: list[EvaluatorConfig] = Field(default_factory=list, max_length=8)


class PlaygroundResult(BaseModel):
    model: str
    provider_type: str = ""
    output: str = ""
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    passed: bool | None = None
    mean_score: float | None = None
    scores: list[PlaygroundScore] = Field(default_factory=list)
    error: str | None = None


class PlaygroundResponse(BaseModel):
    results: list[PlaygroundResult]
