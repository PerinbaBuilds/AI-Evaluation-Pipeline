"""Evaluation run configuration.

A run is described declaratively (YAML or JSON): which dataset to load, which
provider to call, an optional prompt template, and the list of evaluators with
their thresholds. Everything is validated up front with pydantic so a typo
fails before any model call is made, not after item 4,000.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from evalpipe.exceptions import ConfigError

# --------------------------------------------------------------------------- providers


class MockProviderConfig(BaseModel):
    """Deterministic simulation provider — used for demos, CI and A/B pipeline tests."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["mock"] = "mock"
    model: str = "sim-model"
    quality: float = Field(default=0.8, ge=0.0, le=1.0)
    seed: int = 42
    failure_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    latency_ms: float = Field(default=0.0, ge=0.0)
    input_cost_per_1k_tokens: float = Field(default=0.0, ge=0.0)
    output_cost_per_1k_tokens: float = Field(default=0.0, ge=0.0)


class OpenAICompatibleProviderConfig(BaseModel):
    """Any endpoint speaking the OpenAI chat-completions wire format.

    Works with self-hosted runtimes (vLLM, Ollama, llama.cpp server, ...) and
    hosted APIs alike. The API key is referenced by *environment variable
    name*, never stored in config files.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["openai_compatible"] = "openai_compatible"
    model: str
    base_url: str
    api_key_env: str | None = None
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=512, ge=1)
    timeout_s: float = Field(default=30.0, gt=0.0)
    input_cost_per_1k_tokens: float = Field(default=0.0, ge=0.0)
    output_cost_per_1k_tokens: float = Field(default=0.0, ge=0.0)

    @field_validator("base_url")
    @classmethod
    def _valid_url(cls, value: str) -> str:
        if not value.startswith(("http://", "https://")):
            raise ValueError("base_url must start with http:// or https://")
        return value.rstrip("/")


class OpenAIProviderConfig(BaseModel):
    """OpenAI Chat Completions API (ChatGPT models)."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["openai"] = "openai"
    model: str = "gpt-4o-mini"
    base_url: str = "https://api.openai.com/v1"
    api_key_env: str = "OPENAI_API_KEY"
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=512, ge=1)
    timeout_s: float = Field(default=60.0, gt=0.0)
    input_cost_per_1k_tokens: float = Field(default=0.0, ge=0.0)
    output_cost_per_1k_tokens: float = Field(default=0.0, ge=0.0)


class AnthropicProviderConfig(BaseModel):
    """Anthropic Messages API."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["anthropic"] = "anthropic"
    model: str
    base_url: str = "https://api.anthropic.com"
    api_key_env: str = "ANTHROPIC_API_KEY"
    api_version: str = "2023-06-01"
    temperature: float = Field(default=0.0, ge=0.0, le=1.0)
    max_tokens: int = Field(default=512, ge=1)
    timeout_s: float = Field(default=60.0, gt=0.0)
    input_cost_per_1k_tokens: float = Field(default=0.0, ge=0.0)
    output_cost_per_1k_tokens: float = Field(default=0.0, ge=0.0)


class GeminiProviderConfig(BaseModel):
    """Google Gemini generateContent API (free tier available)."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["gemini"] = "gemini"
    model: str = "gemini-1.5-flash"
    base_url: str = "https://generativelanguage.googleapis.com"
    api_key_env: str = "GEMINI_API_KEY"
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=512, ge=1)
    timeout_s: float = Field(default=60.0, gt=0.0)
    input_cost_per_1k_tokens: float = Field(default=0.0, ge=0.0)
    output_cost_per_1k_tokens: float = Field(default=0.0, ge=0.0)


class GroqProviderConfig(BaseModel):
    """Groq (OpenAI-compatible, free tier) — fast hosted open models."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["groq"] = "groq"
    model: str = "llama-3.3-70b-versatile"
    base_url: str = "https://api.groq.com/openai/v1"
    api_key_env: str = "GROQ_API_KEY"
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=512, ge=1)
    timeout_s: float = Field(default=60.0, gt=0.0)
    input_cost_per_1k_tokens: float = Field(default=0.0, ge=0.0)
    output_cost_per_1k_tokens: float = Field(default=0.0, ge=0.0)


class OpenRouterProviderConfig(BaseModel):
    """OpenRouter (OpenAI-compatible) — includes free ``:free`` models."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["openrouter"] = "openrouter"
    model: str = "meta-llama/llama-3.3-70b-instruct:free"
    base_url: str = "https://openrouter.ai/api/v1"
    api_key_env: str = "OPENROUTER_API_KEY"
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=512, ge=1)
    timeout_s: float = Field(default=60.0, gt=0.0)
    input_cost_per_1k_tokens: float = Field(default=0.0, ge=0.0)
    output_cost_per_1k_tokens: float = Field(default=0.0, ge=0.0)


class OllamaProviderConfig(BaseModel):
    """Ollama (OpenAI-compatible) — fully free, runs models locally, no API key."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["ollama"] = "ollama"
    model: str = "llama3.2"
    base_url: str = "http://localhost:11434/v1"
    api_key_env: str | None = None
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=512, ge=1)
    timeout_s: float = Field(default=120.0, gt=0.0)
    input_cost_per_1k_tokens: float = Field(default=0.0, ge=0.0)
    output_cost_per_1k_tokens: float = Field(default=0.0, ge=0.0)


ProviderConfig = Annotated[
    MockProviderConfig
    | OpenAICompatibleProviderConfig
    | OpenAIProviderConfig
    | AnthropicProviderConfig
    | GeminiProviderConfig
    | GroqProviderConfig
    | OpenRouterProviderConfig
    | OllamaProviderConfig,
    Field(discriminator="type"),
]

# -------------------------------------------------------------------------- evaluators


class ExactMatchConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["exact_match"] = "exact_match"
    case_sensitive: bool = False
    strip_punctuation: bool = False
    threshold: float = Field(default=1.0, ge=0.0, le=1.0)


class TokenF1Config(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["token_f1"] = "token_f1"
    threshold: float = Field(default=0.6, ge=0.0, le=1.0)


class ContainsConfig(BaseModel):
    """Pass when the output contains the configured needles (or ``expected`` if unset)."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["contains"] = "contains"
    values: list[str] = Field(default_factory=list)
    mode: Literal["any", "all"] = "any"
    case_sensitive: bool = False
    threshold: float = Field(default=1.0, ge=0.0, le=1.0)


class RegexConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["regex"] = "regex"
    pattern: str
    threshold: float = Field(default=1.0, ge=0.0, le=1.0)

    @field_validator("pattern")
    @classmethod
    def _compiles(cls, value: str) -> str:
        try:
            re.compile(value)
        except re.error as exc:
            raise ValueError(f"invalid regular expression: {exc}") from exc
        return value


class SemanticSimilarityConfig(BaseModel):
    """Dependency-free lexical similarity: cosine over TF-weighted word uni+bigrams."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["semantic_similarity"] = "semantic_similarity"
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)


class SafetyConfig(BaseModel):
    """Blocklist screen: score 1.0 when none of the blocked terms appear."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["safety"] = "safety"
    blocked_terms: list[str] = Field(min_length=1)
    threshold: float = Field(default=1.0, ge=0.0, le=1.0)


JudgeRubric = Literal["correctness", "faithfulness", "relevance"]


class LLMJudgeConfig(BaseModel):
    """LLM-as-a-judge: a grader model scores the output against a rubric.

    ``rubric`` selects a built-in grading rubric; ``custom_rubric`` overrides it
    with free-form grading instructions. The judge provider defaults to the
    run's provider when not set explicitly.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["llm_judge"] = "llm_judge"
    rubric: JudgeRubric = "correctness"
    custom_rubric: str | None = None
    provider: ProviderConfig | None = None
    threshold: float = Field(default=0.7, ge=0.0, le=1.0)


EvaluatorConfig = Annotated[
    ExactMatchConfig
    | TokenF1Config
    | ContainsConfig
    | RegexConfig
    | SemanticSimilarityConfig
    | SafetyConfig
    | LLMJudgeConfig,
    Field(discriminator="type"),
]

# --------------------------------------------------------------------------- run config


class EvalConfig(BaseModel):
    """Top-level description of one evaluation run."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=200)
    dataset: str
    provider: ProviderConfig
    evaluators: list[EvaluatorConfig] = Field(min_length=1)
    prompt_template: str | None = Field(
        default=None,
        description="Optional template applied to every item; must contain {prompt}.",
    )
    concurrency: int = Field(default=8, ge=1, le=64)
    retries: int = Field(default=2, ge=0, le=10)
    retry_backoff_s: float = Field(default=0.5, ge=0.0)
    cache_responses: bool = Field(
        default=False,
        description="Memoise model outputs by (model, prompt) so metric-only re-runs "
        "skip inference. Only sound for deterministic decoding (temperature 0).",
    )

    @field_validator("prompt_template")
    @classmethod
    def _has_placeholder(cls, value: str | None) -> str | None:
        if value is not None and "{prompt}" not in value:
            raise ValueError("prompt_template must contain the {prompt} placeholder")
        return value

    def render_prompt(self, prompt: str) -> str:
        if self.prompt_template is None:
            return prompt
        return self.prompt_template.replace("{prompt}", prompt)


def load_config(path: str | Path) -> EvalConfig:
    """Load an :class:`EvalConfig` from a YAML (or JSON) file."""
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"config file not found: {path}")
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path}: invalid YAML: {exc}") from exc
    if not isinstance(payload, dict):
        raise ConfigError(f"{path}: expected a mapping at the top level")
    try:
        return EvalConfig.model_validate(payload)
    except ValidationError as exc:
        first = exc.errors()[0]
        loc = ".".join(str(part) for part in first["loc"]) or "config"
        raise ConfigError(f"{path}: invalid config ({loc}: {first['msg']})") from exc
