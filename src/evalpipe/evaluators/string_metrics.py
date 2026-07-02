"""Reference-based string metrics: exact match, token F1, contains, regex."""

from __future__ import annotations

import re
from collections import Counter

from evalpipe.config import ContainsConfig, ExactMatchConfig, RegexConfig, TokenF1Config
from evalpipe.datasets import DatasetItem
from evalpipe.evaluators.base import EvalScore, Evaluator, normalize_text, squad_tokens


class ExactMatchEvaluator(Evaluator):
    name = "exact_match"

    def __init__(self, config: ExactMatchConfig) -> None:
        self.threshold = config.threshold
        self._case_sensitive = config.case_sensitive
        self._strip_punctuation = config.strip_punctuation

    async def evaluate(self, item: DatasetItem, output: str) -> EvalScore:
        if item.expected is None:
            return self._missing_reference()
        left = normalize_text(
            output,
            casefold=not self._case_sensitive,
            strip_punctuation=self._strip_punctuation,
        )
        right = normalize_text(
            item.expected,
            casefold=not self._case_sensitive,
            strip_punctuation=self._strip_punctuation,
        )
        return self._score(1.0 if left == right else 0.0)


class TokenF1Evaluator(Evaluator):
    """SQuAD-style bag-of-tokens F1 — the standard partial-credit answer metric."""

    name = "token_f1"

    def __init__(self, config: TokenF1Config) -> None:
        self.threshold = config.threshold

    async def evaluate(self, item: DatasetItem, output: str) -> EvalScore:
        if item.expected is None:
            return self._missing_reference()
        predicted = squad_tokens(output)
        reference = squad_tokens(item.expected)
        if not predicted and not reference:
            return self._score(1.0, "both sides empty after normalisation")
        if not predicted or not reference:
            return self._score(0.0, "one side empty after normalisation")
        overlap = sum((Counter(predicted) & Counter(reference)).values())
        if overlap == 0:
            return self._score(0.0)
        precision = overlap / len(predicted)
        recall = overlap / len(reference)
        return self._score(2 * precision * recall / (precision + recall))


class ContainsEvaluator(Evaluator):
    name = "contains"

    def __init__(self, config: ContainsConfig) -> None:
        self.threshold = config.threshold
        self._values = config.values
        self._mode = config.mode
        self._case_sensitive = config.case_sensitive

    async def evaluate(self, item: DatasetItem, output: str) -> EvalScore:
        needles = self._values or ([item.expected] if item.expected else [])
        if not needles:
            return self._missing_reference()
        haystack = output if self._case_sensitive else output.casefold()
        matched = [
            needle
            for needle in needles
            if (needle if self._case_sensitive else needle.casefold()) in haystack
        ]
        score = (1.0 if matched else 0.0) if self._mode == "any" else len(matched) / len(needles)
        detail = f"matched {len(matched)}/{len(needles)} needle(s)"
        return self._score(score, detail)


class RegexEvaluator(Evaluator):
    name = "regex"

    def __init__(self, config: RegexConfig) -> None:
        self.threshold = config.threshold
        self._pattern = re.compile(config.pattern)

    async def evaluate(self, item: DatasetItem, output: str) -> EvalScore:
        match = self._pattern.search(output)
        detail = f"pattern {self._pattern.pattern!r} {'matched' if match else 'did not match'}"
        return self._score(1.0 if match else 0.0, detail)
