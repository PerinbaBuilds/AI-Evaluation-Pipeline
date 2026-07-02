"""Lexical similarity metric.

Cosine similarity over sublinear-TF-weighted word unigrams and bigrams. It is
a dependency-free approximation of semantic similarity: bigrams reward
preserved phrasing and word order, sublinear TF stops long outputs from
dominating through repetition. Scores land in [0, 1].
"""

from __future__ import annotations

import itertools
import math
from collections import Counter

from evalpipe.config import SemanticSimilarityConfig
from evalpipe.datasets import DatasetItem
from evalpipe.evaluators.base import EvalScore, Evaluator, normalize_text


def _features(text: str) -> Counter[str]:
    tokens = normalize_text(text, casefold=True, strip_punctuation=True).split()
    features: Counter[str] = Counter(tokens)
    features.update(f"{first} {second}" for first, second in itertools.pairwise(tokens))
    return features


def _weights(features: Counter[str]) -> dict[str, float]:
    return {feature: 1.0 + math.log(count) for feature, count in features.items()}


def cosine_similarity(text_a: str, text_b: str) -> float:
    """Cosine similarity of the two texts' uni+bigram TF vectors, in [0, 1]."""
    weights_a = _weights(_features(text_a))
    weights_b = _weights(_features(text_b))
    if not weights_a and not weights_b:
        return 1.0
    if not weights_a or not weights_b:
        return 0.0
    dot = sum(weight * weights_b.get(feature, 0.0) for feature, weight in weights_a.items())
    norm_a = math.sqrt(sum(weight * weight for weight in weights_a.values()))
    norm_b = math.sqrt(sum(weight * weight for weight in weights_b.values()))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return min(1.0, max(0.0, dot / (norm_a * norm_b)))


class SemanticSimilarityEvaluator(Evaluator):
    name = "semantic_similarity"

    def __init__(self, config: SemanticSimilarityConfig) -> None:
        self.threshold = config.threshold

    async def evaluate(self, item: DatasetItem, output: str) -> EvalScore:
        if item.expected is None:
            return self._missing_reference()
        return self._score(cosine_similarity(output, item.expected))
