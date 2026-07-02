"""Providers: mock determinism and simulation behaviour, HTTP wire handling."""

from __future__ import annotations

import json

import httpx
import pytest

from evalpipe.config import MockProviderConfig, OpenAICompatibleProviderConfig
from evalpipe.exceptions import ConfigError, ProviderError
from evalpipe.providers import build_provider
from evalpipe.providers.mock import MockProvider
from evalpipe.providers.openai_compat import OpenAICompatibleProvider


class TestMockProvider:
    async def test_deterministic_per_seed_model_prompt(self) -> None:
        provider = MockProvider(model="m", quality=0.5, seed=1)
        first = await provider.generate("Q?", reference="The answer.")
        second = await provider.generate("Q?", reference="The answer.")
        assert first.text == second.text

    async def test_different_seeds_can_differ(self) -> None:
        outputs = set()
        for seed in range(12):
            provider = MockProvider(model="m", quality=0.5, seed=seed)
            response = await provider.generate("Q?", reference="A very specific answer here.")
            outputs.add(response.text)
        assert len(outputs) > 1

    async def test_quality_one_always_returns_reference(self) -> None:
        provider = MockProvider(quality=1.0)
        for index in range(20):
            response = await provider.generate(f"Q{index}?", reference=f"Answer {index}.")
            assert response.text == f"Answer {index}."

    async def test_quality_zero_never_returns_reference(self) -> None:
        provider = MockProvider(quality=0.0)
        for index in range(20):
            response = await provider.generate(f"Q{index}?", reference="Exact reference answer.")
            assert response.text != "Exact reference answer."

    async def test_quality_controls_accuracy_rate(self) -> None:
        provider = MockProvider(quality=0.8, seed=3)
        hits = 0
        total = 200
        for index in range(total):
            response = await provider.generate(f"Question {index}?", reference=f"Reply {index}.")
            hits += response.text == f"Reply {index}."
        assert 0.7 < hits / total < 0.9

    async def test_failure_rate_raises_provider_error(self) -> None:
        provider = MockProvider(failure_rate=1.0)
        with pytest.raises(ProviderError, match="simulated"):
            await provider.generate("Q?", reference="A.")

    async def test_no_reference_yields_generic_answer(self) -> None:
        provider = MockProvider(quality=1.0)
        response = await provider.generate("Tell me something.")
        assert response.text
        assert response.output_tokens >= 1

    async def test_judge_prompt_gets_json_verdict(self) -> None:
        provider = MockProvider(quality=0.9, seed=5)
        prompt = 'Grade this.\n\nRespond with only a JSON object: {"score": <integer 0-10>}'
        response = await provider.generate(prompt)
        payload = json.loads(response.text)
        assert 0 <= payload["score"] <= 10

    async def test_token_counts_positive(self) -> None:
        provider = MockProvider(quality=1.0)
        response = await provider.generate("How many?", reference="Twelve in total.")
        assert response.input_tokens >= 1
        assert response.output_tokens >= 1

    def test_invalid_quality_rejected(self) -> None:
        with pytest.raises(ValueError, match="quality"):
            MockProvider(quality=2.0)


def _http_provider(handler: httpx.MockTransport, **kwargs: object) -> OpenAICompatibleProvider:
    return OpenAICompatibleProvider(
        base_url="http://testserver/v1",
        model="test-model",
        transport=handler,
        **kwargs,  # type: ignore[arg-type]
    )


class TestOpenAICompatibleProvider:
    async def test_successful_completion(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/v1/chat/completions"
            body = json.loads(request.content)
            assert body["model"] == "test-model"
            assert body["messages"] == [{"role": "user", "content": "Q?"}]
            return httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": "The answer."}}],
                    "usage": {"prompt_tokens": 7, "completion_tokens": 3},
                },
            )

        provider = _http_provider(httpx.MockTransport(handler))
        response = await provider.generate("Q?")
        assert response.text == "The answer."
        assert response.input_tokens == 7
        assert response.output_tokens == 3
        await provider.aclose()

    async def test_api_key_sent_as_bearer(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.headers["Authorization"] == "Bearer sk-test"
            return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

        provider = _http_provider(httpx.MockTransport(handler), api_key="sk-test")
        await provider.generate("Q?")
        await provider.aclose()

    async def test_http_error_status(self) -> None:
        transport = httpx.MockTransport(lambda request: httpx.Response(429, text="rate limited"))
        provider = _http_provider(transport)
        with pytest.raises(ProviderError, match="HTTP 429"):
            await provider.generate("Q?")
        await provider.aclose()

    async def test_network_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("boom")

        provider = _http_provider(httpx.MockTransport(handler))
        with pytest.raises(ProviderError, match="request failed"):
            await provider.generate("Q?")
        await provider.aclose()

    async def test_malformed_body(self) -> None:
        transport = httpx.MockTransport(lambda request: httpx.Response(200, json={"nope": []}))
        provider = _http_provider(transport)
        with pytest.raises(ProviderError, match="malformed"):
            await provider.generate("Q?")
        await provider.aclose()

    async def test_non_string_content(self) -> None:
        transport = httpx.MockTransport(
            lambda request: httpx.Response(200, json={"choices": [{"message": {"content": 5}}]})
        )
        provider = _http_provider(transport)
        with pytest.raises(ProviderError, match="not text"):
            await provider.generate("Q?")
        await provider.aclose()

    async def test_missing_usage_defaults_to_zero(self) -> None:
        transport = httpx.MockTransport(
            lambda request: httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})
        )
        provider = _http_provider(transport)
        response = await provider.generate("Q?")
        assert response.input_tokens == 0
        assert response.output_tokens == 0
        await provider.aclose()

    async def test_reference_never_sent_over_wire(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            assert "SECRET-REFERENCE" not in request.content.decode()
            return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

        provider = _http_provider(httpx.MockTransport(handler))
        await provider.generate("Q?", reference="SECRET-REFERENCE")
        await provider.aclose()


class TestBuildProvider:
    def test_builds_mock(self) -> None:
        provider = build_provider(MockProviderConfig(model="m", quality=0.4))
        assert isinstance(provider, MockProvider)
        assert provider.quality == 0.4

    def test_builds_http_provider(self) -> None:
        config = OpenAICompatibleProviderConfig(model="m", base_url="http://host/v1")
        provider = build_provider(config)
        assert isinstance(provider, OpenAICompatibleProvider)

    def test_missing_api_key_env_fails_fast(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("MISSING_KEY_VAR", raising=False)
        config = OpenAICompatibleProviderConfig(
            model="m", base_url="http://host/v1", api_key_env="MISSING_KEY_VAR"
        )
        with pytest.raises(ConfigError, match="MISSING_KEY_VAR"):
            build_provider(config)

    def test_api_key_env_resolved(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PRESENT_KEY_VAR", "sk-live")
        config = OpenAICompatibleProviderConfig(
            model="m", base_url="http://host/v1", api_key_env="PRESENT_KEY_VAR"
        )
        provider = build_provider(config)
        assert isinstance(provider, OpenAICompatibleProvider)

    def test_cost_estimate(self) -> None:
        provider = build_provider(
            MockProviderConfig(
                model="m",
                quality=1.0,
                input_cost_per_1k_tokens=1.0,
                output_cost_per_1k_tokens=2.0,
            )
        )
        from evalpipe.providers.base import ModelResponse

        response = ModelResponse(text="x", input_tokens=500, output_tokens=250)
        assert provider.estimate_cost_usd(response) == pytest.approx(0.5 + 0.5)
