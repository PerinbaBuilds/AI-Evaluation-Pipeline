"""Evaluator registry."""

from __future__ import annotations

from evalpipe.config import (
    ContainsConfig,
    EvaluatorConfig,
    ExactMatchConfig,
    LLMJudgeConfig,
    RegexConfig,
    SafetyConfig,
    SemanticSimilarityConfig,
    TokenF1Config,
)
from evalpipe.evaluators.base import EvalScore, Evaluator, normalize_text, squad_tokens
from evalpipe.evaluators.llm_judge import LLMJudgeEvaluator, parse_judge_score
from evalpipe.evaluators.safety import SafetyEvaluator
from evalpipe.evaluators.semantic import SemanticSimilarityEvaluator, cosine_similarity
from evalpipe.evaluators.string_metrics import (
    ContainsEvaluator,
    ExactMatchEvaluator,
    RegexEvaluator,
    TokenF1Evaluator,
)
from evalpipe.providers.base import ModelProvider

__all__ = [
    "EvalScore",
    "Evaluator",
    "build_evaluators",
    "cosine_similarity",
    "normalize_text",
    "parse_judge_score",
    "squad_tokens",
]


def build_evaluators(
    configs: list[EvaluatorConfig], default_judge: ModelProvider
) -> list[Evaluator]:
    """Instantiate evaluators; ``default_judge`` backs any judge without its own provider."""
    from evalpipe.providers import build_provider

    evaluators: list[Evaluator] = []
    for config in configs:
        if isinstance(config, ExactMatchConfig):
            evaluators.append(ExactMatchEvaluator(config))
        elif isinstance(config, TokenF1Config):
            evaluators.append(TokenF1Evaluator(config))
        elif isinstance(config, ContainsConfig):
            evaluators.append(ContainsEvaluator(config))
        elif isinstance(config, RegexConfig):
            evaluators.append(RegexEvaluator(config))
        elif isinstance(config, SemanticSimilarityConfig):
            evaluators.append(SemanticSimilarityEvaluator(config))
        elif isinstance(config, SafetyConfig):
            evaluators.append(SafetyEvaluator(config))
        elif isinstance(config, LLMJudgeConfig):
            judge = build_provider(config.provider) if config.provider else default_judge
            evaluators.append(LLMJudgeEvaluator(config, judge))
        else:  # pragma: no cover - unreachable with a validated config
            raise TypeError(f"unknown evaluator config: {type(config).__name__}")
    return evaluators
