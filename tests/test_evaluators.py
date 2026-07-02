"""Evaluator behaviour and edge cases: empty strings, unicode, punctuation, thresholds."""

from __future__ import annotations

import pytest

from evalpipe.config import (
    ContainsConfig,
    ExactMatchConfig,
    RegexConfig,
    SafetyConfig,
    SemanticSimilarityConfig,
    TokenF1Config,
)
from evalpipe.datasets import DatasetItem
from evalpipe.evaluators.base import normalize_text, squad_tokens
from evalpipe.evaluators.safety import SafetyEvaluator
from evalpipe.evaluators.semantic import SemanticSimilarityEvaluator, cosine_similarity
from evalpipe.evaluators.string_metrics import (
    ContainsEvaluator,
    ExactMatchEvaluator,
    RegexEvaluator,
    TokenF1Evaluator,
)


def item(expected: str | None = "Paris is the capital.") -> DatasetItem:
    return DatasetItem(id="x", prompt="Capital of France?", expected=expected)


class TestNormalization:
    def test_casefold_and_whitespace(self) -> None:
        assert normalize_text("  Hello   WORLD  ") == "hello world"

    def test_punctuation_stripping(self) -> None:
        assert normalize_text("Hello, world!", strip_punctuation=True) == "hello world"

    def test_article_dropping(self) -> None:
        assert squad_tokens("The answer is a dog") == ["answer", "is", "dog"]

    def test_empty_string(self) -> None:
        assert normalize_text("") == ""
        assert squad_tokens("...") == []


class TestExactMatch:
    async def test_match_ignores_case_and_spacing(self) -> None:
        evaluator = ExactMatchEvaluator(ExactMatchConfig())
        score = await evaluator.evaluate(item(), "  PARIS is the   capital. ")
        assert score.score == 1.0
        assert score.passed

    async def test_mismatch(self) -> None:
        evaluator = ExactMatchEvaluator(ExactMatchConfig())
        score = await evaluator.evaluate(item(), "London is the capital.")
        assert score.score == 0.0
        assert not score.passed

    async def test_case_sensitive_mode(self) -> None:
        evaluator = ExactMatchEvaluator(ExactMatchConfig(case_sensitive=True))
        score = await evaluator.evaluate(item(), "paris is the capital.")
        assert score.score == 0.0

    async def test_punctuation_insensitive_mode(self) -> None:
        evaluator = ExactMatchEvaluator(ExactMatchConfig(strip_punctuation=True))
        score = await evaluator.evaluate(item(), "Paris is the capital")
        assert score.score == 1.0

    async def test_missing_reference_scores_zero_with_detail(self) -> None:
        evaluator = ExactMatchEvaluator(ExactMatchConfig())
        score = await evaluator.evaluate(item(expected=None), "anything")
        assert score.score == 0.0
        assert "no 'expected'" in score.detail


class TestTokenF1:
    async def test_perfect_overlap(self) -> None:
        evaluator = TokenF1Evaluator(TokenF1Config())
        score = await evaluator.evaluate(item(), "Paris is the capital.")
        assert score.score == pytest.approx(1.0)

    async def test_partial_overlap(self) -> None:
        evaluator = TokenF1Evaluator(TokenF1Config(threshold=0.5))
        score = await evaluator.evaluate(item("the quick brown fox"), "the quick red fox")
        # tokens (articles dropped): {quick, brown, fox} vs {quick, red, fox} -> F1 = 2/3
        assert score.score == pytest.approx(2 / 3, abs=1e-9)
        assert score.passed

    async def test_no_overlap(self) -> None:
        evaluator = TokenF1Evaluator(TokenF1Config())
        score = await evaluator.evaluate(item("alpha beta"), "gamma delta")
        assert score.score == 0.0

    async def test_both_empty_after_normalisation(self) -> None:
        evaluator = TokenF1Evaluator(TokenF1Config())
        score = await evaluator.evaluate(item("the a an"), "the")
        assert score.score == 1.0

    async def test_one_side_empty(self) -> None:
        evaluator = TokenF1Evaluator(TokenF1Config())
        score = await evaluator.evaluate(item("real answer"), "...")
        assert score.score == 0.0

    async def test_repeated_tokens_counted_once_each(self) -> None:
        evaluator = TokenF1Evaluator(TokenF1Config())
        score = await evaluator.evaluate(item("dog dog dog"), "dog")
        # overlap 1, precision 1, recall 1/3 -> F1 = 0.5
        assert score.score == pytest.approx(0.5)


class TestContains:
    async def test_any_mode(self) -> None:
        evaluator = ContainsEvaluator(ContainsConfig(values=["Paris", "London"]))
        score = await evaluator.evaluate(item(), "I think paris fits")
        assert score.score == 1.0

    async def test_all_mode_partial_credit(self) -> None:
        evaluator = ContainsEvaluator(
            ContainsConfig(values=["alpha", "beta", "gamma"], mode="all", threshold=1.0)
        )
        score = await evaluator.evaluate(item(), "alpha and beta only")
        assert score.score == pytest.approx(2 / 3)
        assert not score.passed

    async def test_case_sensitive(self) -> None:
        evaluator = ContainsEvaluator(ContainsConfig(values=["Paris"], case_sensitive=True))
        score = await evaluator.evaluate(item(), "paris")
        assert score.score == 0.0

    async def test_falls_back_to_expected(self) -> None:
        evaluator = ContainsEvaluator(ContainsConfig())
        score = await evaluator.evaluate(item("Paris"), "Paris, obviously.")
        assert score.score == 1.0

    async def test_no_needles_and_no_expected(self) -> None:
        evaluator = ContainsEvaluator(ContainsConfig())
        score = await evaluator.evaluate(item(expected=None), "anything")
        assert score.score == 0.0


class TestRegex:
    async def test_match(self) -> None:
        evaluator = RegexEvaluator(RegexConfig(pattern=r"\b\d{4}\b"))
        score = await evaluator.evaluate(item(), "It happened in 1969.")
        assert score.score == 1.0

    async def test_no_match(self) -> None:
        evaluator = RegexEvaluator(RegexConfig(pattern=r"^\d+$"))
        score = await evaluator.evaluate(item(), "not a number")
        assert score.score == 0.0

    async def test_empty_output(self) -> None:
        evaluator = RegexEvaluator(RegexConfig(pattern="x"))
        score = await evaluator.evaluate(item(), "")
        assert score.score == 0.0


class TestSafety:
    async def test_clean_output_passes(self) -> None:
        evaluator = SafetyEvaluator(SafetyConfig(blocked_terms=["secret"]))
        score = await evaluator.evaluate(item(), "A perfectly safe answer.")
        assert score.score == 1.0
        assert score.passed

    async def test_blocked_term_fails_case_insensitively(self) -> None:
        evaluator = SafetyEvaluator(SafetyConfig(blocked_terms=["SECRET"]))
        score = await evaluator.evaluate(item(), "here is the Secret sauce")
        assert score.score == 0.0
        assert "secret" in score.detail


class TestSemanticSimilarity:
    def test_identical_texts(self) -> None:
        assert cosine_similarity("the cat sat", "the cat sat") == pytest.approx(1.0)

    def test_disjoint_texts(self) -> None:
        assert cosine_similarity("alpha beta", "gamma delta") == 0.0

    def test_partial_similarity_between_zero_and_one(self) -> None:
        value = cosine_similarity(
            "the capital of France is Paris",
            "Paris is the capital city of France",
        )
        assert 0.3 < value < 1.0

    def test_word_order_matters_via_bigrams(self) -> None:
        same_order = cosine_similarity("big red dog", "big red dog")
        shuffled = cosine_similarity("big red dog", "dog red big")
        assert shuffled < same_order

    def test_both_empty(self) -> None:
        assert cosine_similarity("", "") == 1.0

    def test_one_empty(self) -> None:
        assert cosine_similarity("words here", "") == 0.0

    async def test_evaluator_threshold(self) -> None:
        evaluator = SemanticSimilarityEvaluator(SemanticSimilarityConfig(threshold=0.99))
        score = await evaluator.evaluate(item(), "Paris is the capital.")
        assert score.passed

    async def test_missing_reference(self) -> None:
        evaluator = SemanticSimilarityEvaluator(SemanticSimilarityConfig())
        score = await evaluator.evaluate(item(expected=None), "text")
        assert score.score == 0.0
