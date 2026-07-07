"""Model providers.

A provider turns a prompt into text. :func:`build_provider` maps a validated
provider config onto a concrete implementation. API keys are always resolved
from environment variables at build time — never stored in configs or exposed.
"""

from __future__ import annotations

import os

from evalpipe.config import (
    AnthropicProviderConfig,
    GeminiProviderConfig,
    GroqProviderConfig,
    MockProviderConfig,
    OllamaProviderConfig,
    OpenAICompatibleProviderConfig,
    OpenAIProviderConfig,
    OpenRouterProviderConfig,
    ProviderConfig,
)
from evalpipe.exceptions import ConfigError
from evalpipe.providers.anthropic import AnthropicProvider
from evalpipe.providers.base import ModelProvider, ModelResponse
from evalpipe.providers.gemini import GeminiProvider
from evalpipe.providers.mock import MockProvider
from evalpipe.providers.openai_compat import OpenAICompatibleProvider

__all__ = [
    "AnthropicProvider",
    "GeminiProvider",
    "MockProvider",
    "ModelProvider",
    "ModelResponse",
    "OpenAICompatibleProvider",
    "build_provider",
]


def _require_key(env_name: str) -> str:
    key = os.environ.get(env_name)
    if not key:
        raise ConfigError(
            f"environment variable {env_name!r} is not set (the provider needs it for its API key)"
        )
    return key


def build_provider(config: ProviderConfig) -> ModelProvider:
    """Instantiate the provider described by ``config``."""
    if isinstance(config, MockProviderConfig):
        return MockProvider(
            model=config.model,
            quality=config.quality,
            seed=config.seed,
            failure_rate=config.failure_rate,
            latency_ms=config.latency_ms,
            input_cost_per_1k_tokens=config.input_cost_per_1k_tokens,
            output_cost_per_1k_tokens=config.output_cost_per_1k_tokens,
        )
    if isinstance(config, OpenAICompatibleProviderConfig):
        api_key: str | None = None
        if config.api_key_env:
            api_key = _require_key(config.api_key_env)
        return OpenAICompatibleProvider(
            base_url=config.base_url,
            model=config.model,
            api_key=api_key,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            timeout_s=config.timeout_s,
            input_cost_per_1k_tokens=config.input_cost_per_1k_tokens,
            output_cost_per_1k_tokens=config.output_cost_per_1k_tokens,
        )
    if isinstance(
        config,
        OpenAIProviderConfig | GroqProviderConfig | OpenRouterProviderConfig | OllamaProviderConfig,
    ):
        # OpenAI-compatible presets: differ only in default base_url / key env.
        api_key = _require_key(config.api_key_env) if config.api_key_env else None
        return OpenAICompatibleProvider(
            base_url=config.base_url,
            model=config.model,
            api_key=api_key,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            timeout_s=config.timeout_s,
            input_cost_per_1k_tokens=config.input_cost_per_1k_tokens,
            output_cost_per_1k_tokens=config.output_cost_per_1k_tokens,
        )
    if isinstance(config, AnthropicProviderConfig):
        return AnthropicProvider(
            model=config.model,
            api_key=_require_key(config.api_key_env),
            base_url=config.base_url,
            api_version=config.api_version,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            timeout_s=config.timeout_s,
            input_cost_per_1k_tokens=config.input_cost_per_1k_tokens,
            output_cost_per_1k_tokens=config.output_cost_per_1k_tokens,
        )
    if isinstance(config, GeminiProviderConfig):
        return GeminiProvider(
            model=config.model,
            api_key=_require_key(config.api_key_env),
            base_url=config.base_url,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            timeout_s=config.timeout_s,
            input_cost_per_1k_tokens=config.input_cost_per_1k_tokens,
            output_cost_per_1k_tokens=config.output_cost_per_1k_tokens,
        )
    raise ConfigError(f"unknown provider type: {type(config).__name__}")  # pragma: no cover
