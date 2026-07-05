"""Provider interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelResponse:
    """One completion from a provider.

    ``cached`` marks a response served from the response cache rather than a
    fresh inference call — such a response is billed at zero cost.
    """

    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    cached: bool = False

    def cost_usd(self, input_cost_per_1k: float, output_cost_per_1k: float) -> float:
        if self.cached:
            return 0.0
        return (
            self.input_tokens / 1000.0 * input_cost_per_1k
            + self.output_tokens / 1000.0 * output_cost_per_1k
        )


class ModelProvider(ABC):
    """Turns a prompt into a completion.

    ``reference`` carries the item's expected answer and exists solely for
    *simulation* providers (see :class:`~evalpipe.providers.mock.MockProvider`),
    which use it to synthesise outputs at a controlled accuracy level. Real
    network providers must ignore it — it is never sent over the wire.
    """

    model: str
    input_cost_per_1k_tokens: float = 0.0
    output_cost_per_1k_tokens: float = 0.0

    @abstractmethod
    async def generate(self, prompt: str, *, reference: str | None = None) -> ModelResponse:
        """Produce a completion for ``prompt``; raises ``ProviderError`` on failure."""

    async def aclose(self) -> None:
        """Release any underlying resources (HTTP connections, ...)."""
        return None

    def estimate_cost_usd(self, response: ModelResponse) -> float:
        return response.cost_usd(self.input_cost_per_1k_tokens, self.output_cost_per_1k_tokens)
