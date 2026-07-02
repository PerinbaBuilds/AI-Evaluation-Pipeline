"""Blocklist safety screen.

A lightweight guardrail metric: the output must not contain any of the
configured blocked terms (case-insensitive). Useful for brand-safety words,
competitor names, PII markers or phrases the product must never emit.
"""

from __future__ import annotations

from evalpipe.config import SafetyConfig
from evalpipe.datasets import DatasetItem
from evalpipe.evaluators.base import EvalScore, Evaluator


class SafetyEvaluator(Evaluator):
    name = "safety"

    def __init__(self, config: SafetyConfig) -> None:
        self.threshold = config.threshold
        self._blocked = [term.casefold() for term in config.blocked_terms if term.strip()]

    async def evaluate(self, item: DatasetItem, output: str) -> EvalScore:
        haystack = output.casefold()
        found = sorted({term for term in self._blocked if term in haystack})
        if found:
            return self._score(0.0, f"blocked term(s) present: {', '.join(found[:5])}")
        return self._score(1.0, "no blocked terms present")
