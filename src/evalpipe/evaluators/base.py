"""Evaluator interface and shared text utilities."""

from __future__ import annotations

import re
import string
from abc import ABC, abstractmethod
from dataclasses import dataclass

from evalpipe.datasets import DatasetItem

_ARTICLES = {"a", "an", "the"}
_WHITESPACE_RE = re.compile(r"\s+")
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


@dataclass(frozen=True)
class EvalScore:
    """One evaluator's verdict on one item. ``score`` is always within [0, 1]."""

    name: str
    score: float
    passed: bool
    detail: str = ""


class Evaluator(ABC):
    """Scores a model output for a dataset item.

    Evaluators never raise for ordinary data conditions (missing reference,
    empty output); those produce a zero score with an explanatory detail so a
    single odd item cannot abort a run.
    """

    name: str
    threshold: float

    @abstractmethod
    async def evaluate(self, item: DatasetItem, output: str) -> EvalScore:
        """Return the score for ``output`` against ``item``."""

    def _score(self, score: float, detail: str = "") -> EvalScore:
        score = min(1.0, max(0.0, score))
        return EvalScore(name=self.name, score=score, passed=score >= self.threshold, detail=detail)

    def _missing_reference(self) -> EvalScore:
        return EvalScore(
            name=self.name,
            score=0.0,
            passed=False,
            detail="item has no 'expected' value; reference-based metric scored 0",
        )


def normalize_text(
    text: str,
    *,
    casefold: bool = True,
    strip_punctuation: bool = False,
    drop_articles: bool = False,
) -> str:
    """Whitespace-collapsed canonical form used by the string metrics."""
    if casefold:
        text = text.casefold()
    if strip_punctuation:
        text = text.translate(_PUNCT_TABLE)
    tokens = _WHITESPACE_RE.split(text.strip())
    if drop_articles:
        tokens = [token for token in tokens if token not in _ARTICLES]
    return " ".join(token for token in tokens if token)


def squad_tokens(text: str) -> list[str]:
    """SQuAD-style tokenisation: casefold, strip punctuation, drop articles."""
    normalized = normalize_text(text, casefold=True, strip_punctuation=True, drop_articles=True)
    return normalized.split()
