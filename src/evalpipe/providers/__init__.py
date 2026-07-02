"""Model providers.

A provider turns a prompt into text. :func:`build_provider` maps a validated
provider config onto a concrete implementation.
"""

from __future__ import annotations

import os

from evalpipe.config import MockProviderConfig, OpenAICompatibleProviderConfig, ProviderConfig
from evalpipe.exceptions import ConfigError
from evalpipe.providers.base import ModelProvider, ModelResponse
from evalpipe.providers.mock import MockProvider
from evalpipe.providers.openai_compat import OpenAICompatibleProvider

__all__ = [
    "MockProvider",
    "ModelProvider",
    "ModelResponse",
    "OpenAICompatibleProvider",
    "build_provider",
]


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
            api_key = os.environ.get(config.api_key_env)
            if not api_key:
                raise ConfigError(
                    f"environment variable {config.api_key_env!r} is not set "
                    "(the config references it for the provider API key)"
                )
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
    raise ConfigError(f"unknown provider type: {type(config).__name__}")  # pragma: no cover
