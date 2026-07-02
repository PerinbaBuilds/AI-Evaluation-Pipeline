"""Request/response models for the REST API."""

from __future__ import annotations

from pydantic import BaseModel, Field

from evalpipe.config import ProviderConfig


class HealthResponse(BaseModel):
    status: str
    version: str


class RunCreatedResponse(BaseModel):
    run_id: str
    status: str = "running"


class PromptSaveRequest(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    content: str = Field(min_length=1)


class PlaygroundRequest(BaseModel):
    """Online evaluation: one prompt against up to four providers, side by side."""

    prompt: str = Field(min_length=1, max_length=20_000)
    reference: str | None = None
    providers: list[ProviderConfig] = Field(min_length=1, max_length=4)


class PlaygroundResult(BaseModel):
    model: str
    output: str = ""
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    error: str | None = None


class PlaygroundResponse(BaseModel):
    results: list[PlaygroundResult]
