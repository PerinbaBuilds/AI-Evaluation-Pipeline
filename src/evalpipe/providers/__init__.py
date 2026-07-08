"""Model providers.

A provider turns a prompt into text. :func:`build_provider` maps a validated
provider config onto a concrete implementation. An API key may be supplied
inline on the config (for interactive/API use, where it lives for one request
and is never persisted) or, by default, read from the environment variable
named by ``api_key_env``.
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


def _resolve_key(config: ProviderConfig) -> str | None:
    """Resolve a provider's API key.

    An inline ``api_key`` on the config wins (interactive/API use). Otherwise the
    key is read from the environment variable named by ``api_key_env``. Providers
    that need no key (local Ollama, keyless proxies) resolve to ``None``.
    """
    inline = getattr(config, "api_key", None)
    if inline:
        return str(inline)
    env_name = getattr(config, "api_key_env", None)
    if env_name:
        key = os.environ.get(env_name)
        if not key:
            raise ConfigError(
                f"No API key for this model: set the {env_name} environment variable "
                "on the server, or provide a key with this request."
            )
        return key
    return None


def _require_key(config: ProviderConfig) -> str:
    """Like :func:`_resolve_key` but for providers that cannot run without a key."""
    key = _resolve_key(config)
    if key is None:  # pragma: no cover - keyed providers always name an env var
        raise ConfigError("this provider requires an API key")
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
        api_key: str | None = _resolve_key(config)
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
        api_key = _resolve_key(config)
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
            api_key=_require_key(config),
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
            api_key=_require_key(config),
            base_url=config.base_url,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            timeout_s=config.timeout_s,
            input_cost_per_1k_tokens=config.input_cost_per_1k_tokens,
            output_cost_per_1k_tokens=config.output_cost_per_1k_tokens,
        )
    raise ConfigError(f"unknown provider type: {type(config).__name__}")  # pragma: no cover
