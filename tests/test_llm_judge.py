"""LLM-as-a-judge: rubric rendering, verdict parsing robustness, failure handling."""

from __future__ import annotations

import pytest

from evalpipe.config import LLMJudgeConfig
from evalpipe.datasets import DatasetItem
from evalpipe.evaluators.llm_judge import LLMJudgeEvaluator, parse_judge_score
from evalpipe.exceptions import ProviderError
from evalpipe.providers.base import ModelProvider, ModelResponse


class ScriptedJudge(ModelProvider):
    """Returns a fixed reply and records the grading prompt it received."""

    model = "scripted-judge"

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.prompts: list[str] = []

    async def generate(self, prompt: str, *, reference: str | None = None) -> ModelResponse:
        self.prompts.append(prompt)
        return ModelResponse(text=self.reply)


class FailingJudge(ModelProvider):
    model = "failing-judge"

    async def generate(self, prompt: str, *, reference: str | None = None) -> ModelResponse:
        raise ProviderError("judge is down")


def qa_item() -> DatasetItem:
    return DatasetItem(
        id="x",
        prompt="Capital of France?",
        expected="Paris.",
        metadata={"context": "France's capital is Paris."},
    )


class TestParsing:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ('{"score": 8, "reasoning": "good"}', 8.0),
            ('Sure! Here is my grade: {"score": 3}', 3.0),
            ('{"score": 9.5, "reasoning": "excellent"}', 9.5),
            ('{"reasoning": "ok", "score": 6}', 6.0),
            ('score: 7 out of 10... final json {"score": 7}', 7.0),
            ('{"score": 15}', 10.0),  # clamped
            ('{"score": -2}', 0.0),  # clamped
            ("I would rate this 4", 4.0),  # bare number fallback
            ("10", 10.0),
        ],
    )
    def test_parses_common_shapes(self, text: str, expected: float) -> None:
        assert parse_judge_score(text) == expected

    @pytest.mark.parametrize(
        "text",
        ["", "no verdict at all", '{"grade": "A"}', "score: unknown"],
    )
    def test_unparseable_returns_none(self, text: str) -> None:
        assert parse_judge_score(text) is None

    def test_prefers_json_over_bare_numbers(self) -> None:
        assert parse_judge_score('Steps 1 2 3 then {"score": 9}') == 9.0


class TestEvaluator:
    async def test_correctness_rubric_includes_reference(self) -> None:
        judge = ScriptedJudge('{"score": 10}')
        evaluator = LLMJudgeEvaluator(LLMJudgeConfig(rubric="correctness"), judge)
        score = await evaluator.evaluate(qa_item(), "Paris")
        assert score.score == 1.0
        assert score.passed
        assert "Paris." in judge.prompts[0]  # reference answer present
        assert "Answer to grade:\nParis" in judge.prompts[0]

    async def test_faithfulness_rubric_includes_context(self) -> None:
        judge = ScriptedJudge('{"score": 5}')
        evaluator = LLMJudgeEvaluator(LLMJudgeConfig(rubric="faithfulness", threshold=0.6), judge)
        score = await evaluator.evaluate(qa_item(), "Paris")
        assert score.score == 0.5
        assert not score.passed
        assert "France's capital is Paris." in judge.prompts[0]
        assert evaluator.name == "llm_judge_faithfulness"

    async def test_custom_rubric_used_verbatim(self) -> None:
        judge = ScriptedJudge('{"score": 10}')
        config = LLMJudgeConfig(custom_rubric="Grade politeness of: {output}")
        evaluator = LLMJudgeEvaluator(config, judge)
        await evaluator.evaluate(qa_item(), "Thank you kindly!")
        assert judge.prompts[0].startswith("Grade politeness of: Thank you kindly!")
        assert evaluator.name == "llm_judge"

    async def test_unparseable_verdict_scores_zero(self) -> None:
        judge = ScriptedJudge("I refuse to answer in the requested format")
        evaluator = LLMJudgeEvaluator(LLMJudgeConfig(), judge)
        score = await evaluator.evaluate(qa_item(), "Paris")
        assert score.score == 0.0
        assert "unparseable" in score.detail

    async def test_judge_failure_scores_zero_not_raises(self) -> None:
        evaluator = LLMJudgeEvaluator(LLMJudgeConfig(), FailingJudge())
        score = await evaluator.evaluate(qa_item(), "Paris")
        assert score.score == 0.0
        assert "judge call failed" in score.detail

    async def test_missing_reference_and_context_placeholders(self) -> None:
        judge = ScriptedJudge('{"score": 7}')
        evaluator = LLMJudgeEvaluator(LLMJudgeConfig(rubric="correctness"), judge)
        bare = DatasetItem(id="y", prompt="Q?")
        score = await evaluator.evaluate(bare, "output")
        assert score.score == 0.7
        assert "(no reference provided)" in judge.prompts[0]
