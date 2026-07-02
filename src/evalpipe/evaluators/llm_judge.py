"""LLM-as-a-judge evaluator.

A grader model receives the prompt, the model output and (depending on the
rubric) the reference answer or retrieval context, and returns a 0-10 score
as JSON. Parsing is deliberately forgiving — judges drift from the requested
format — but an unparseable verdict scores 0 and says why, rather than
guessing.

Built-in rubrics mirror the common evaluation families:

- ``correctness``  — is the output right, judged against the reference answer?
- ``faithfulness`` — is the output supported by the provided context (RAG)?
- ``relevance``    — does the output actually address the prompt?
"""

from __future__ import annotations

import json
import re

from evalpipe.config import LLMJudgeConfig
from evalpipe.datasets import DatasetItem
from evalpipe.evaluators.base import EvalScore, Evaluator
from evalpipe.exceptions import ProviderError
from evalpipe.providers.base import ModelProvider

_JSON_INSTRUCTION = (
    'Respond with only a JSON object: {"score": <integer 0-10>, "reasoning": "<one sentence>"}'
)

_RUBRICS = {
    "correctness": (
        "You are grading an answer for correctness against a reference answer.\n"
        "Score 0 = completely wrong, 10 = fully correct and complete.\n"
        "Question:\n{prompt}\n\nReference answer:\n{expected}\n\nAnswer to grade:\n{output}"
    ),
    "faithfulness": (
        "You are grading whether an answer is faithful to the provided context.\n"
        "Score 0 = contradicts or invents facts, 10 = every claim is supported.\n"
        "Context:\n{context}\n\nQuestion:\n{prompt}\n\nAnswer to grade:\n{output}"
    ),
    "relevance": (
        "You are grading whether an answer addresses the question asked.\n"
        "Score 0 = off-topic or evasive, 10 = directly and fully on-topic.\n"
        "Question:\n{prompt}\n\nAnswer to grade:\n{output}"
    ),
}

_JSON_OBJECT_RE = re.compile(r"\{.*?\}", re.DOTALL)
_SCORE_FIELD_RE = re.compile(r'"score"\s*:\s*(-?\d+(?:\.\d+)?)')
_BARE_NUMBER_RE = re.compile(r"(?<![\d.])(10|\d(?:\.\d+)?)(?![\d.])")


class LLMJudgeEvaluator(Evaluator):
    name = "llm_judge"

    def __init__(self, config: LLMJudgeConfig, judge: ModelProvider) -> None:
        self.threshold = config.threshold
        self._judge = judge
        self._rubric_name = config.rubric
        self._template = (
            config.custom_rubric if config.custom_rubric is not None else _RUBRICS[config.rubric]
        )
        self.name = f"llm_judge_{config.rubric}" if config.custom_rubric is None else "llm_judge"

    async def evaluate(self, item: DatasetItem, output: str) -> EvalScore:
        grading_prompt = self._render(item, output)
        try:
            response = await self._judge.generate(grading_prompt)
        except ProviderError as exc:
            return self._score(0.0, f"judge call failed: {exc}")
        raw_score = parse_judge_score(response.text)
        if raw_score is None:
            return self._score(0.0, f"judge verdict unparseable: {response.text[:120]!r}")
        return self._score(raw_score / 10.0, f"judge score {raw_score:g}/10")

    def _render(self, item: DatasetItem, output: str) -> str:
        body = self._template.format(
            prompt=item.prompt,
            output=output,
            expected=item.expected or "(no reference provided)",
            context=item.metadata.get("context", "(no context provided)"),
        )
        return f"{body}\n\n{_JSON_INSTRUCTION}"


def parse_judge_score(text: str) -> float | None:
    """Extract a 0-10 score from a judge response; ``None`` when nothing usable exists."""
    for candidate in _JSON_OBJECT_RE.findall(text):
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and isinstance(payload.get("score"), int | float):
            return _clamp(float(payload["score"]))
    field_match = _SCORE_FIELD_RE.search(text)
    if field_match:
        return _clamp(float(field_match.group(1)))
    bare_match = _BARE_NUMBER_RE.search(text)
    if bare_match:
        return _clamp(float(bare_match.group(1)))
    return None


def _clamp(score: float) -> float:
    return min(10.0, max(0.0, score))
