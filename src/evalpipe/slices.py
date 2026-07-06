"""Subgroup (slice) analysis.

Aggregate metrics hide *where* a model is weak. Slice analysis groups a run's
per-item results by a dataset metadata field (topic, difficulty, language, ...)
and reports pass rate — with a Wilson interval — and mean score per group, so a
50% overall pass rate resolves into "95% on easy, 20% on hard".

Only low-cardinality fields make useful slices, so :func:`sliceable_keys`
offers keys with between 2 and :data:`MAX_SLICE_VALUES` distinct values and
skips per-item keys (a free-text ``context`` field is not a subgroup).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from evalpipe.runner import ItemResult
from evalpipe.stats import mean, wilson_interval

UNLABELED = "(unlabeled)"
MAX_SLICE_VALUES = 12


@dataclass(frozen=True)
class SliceStat:
    value: str
    n: int
    passed: int
    pass_rate: float
    pass_ci: tuple[float, float]
    mean_score: float


def sliceable_keys(results: Sequence[ItemResult]) -> list[str]:
    """Metadata keys worth slicing by — categorical, low-cardinality fields."""
    distinct: dict[str, set[str]] = {}
    for result in results:
        for key, value in result.metadata.items():
            if value:
                distinct.setdefault(key, set()).add(value)
    return sorted(key for key, values in distinct.items() if 2 <= len(values) <= MAX_SLICE_VALUES)


def slice_by(results: Sequence[ItemResult], key: str) -> list[SliceStat]:
    """Group results by ``metadata[key]``; weakest slice first (most actionable)."""
    groups: dict[str, list[ItemResult]] = {}
    for result in results:
        value = result.metadata.get(key) or UNLABELED
        groups.setdefault(value, []).append(result)

    stats = [
        SliceStat(
            value=value,
            n=len(items),
            passed=sum(1 for item in items if item.passed),
            pass_rate=sum(1 for item in items if item.passed) / len(items),
            pass_ci=wilson_interval(sum(1 for item in items if item.passed), len(items)),
            mean_score=mean([item.mean_score for item in items]),
        )
        for value, items in groups.items()
    ]
    return sorted(stats, key=lambda s: (s.pass_rate, -s.n, s.value))
