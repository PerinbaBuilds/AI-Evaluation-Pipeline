"""A/B comparison of two evaluation runs.

Items are paired by ``item_id`` and only the intersection is compared, so the
verdict is never skewed by one run covering extra items. Two hypothesis tests
run side by side:

- **pass rate** (primary): two-proportion z-test with Wilson intervals
- **mean score** (secondary): Welch's t-test plus a bootstrap CI on the diff

The verdict is decided by the primary metric; the secondary breaks ties when
the pass-rate test is inconclusive. Small samples don't error — they produce
an explicit warning, because "n=8, not significant" is a finding, not a crash.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

from evalpipe.stats import (
    TTestResult,
    ZTestResult,
    bootstrap_mean_diff_ci,
    two_proportion_z_test,
    welch_t_test,
    wilson_interval,
)

Verdict = Literal["candidate_better", "baseline_better", "inconclusive"]

MIN_RECOMMENDED_SAMPLE = 30


@dataclass(frozen=True)
class ItemOutcome:
    """The per-item facts the comparison needs, decoupled from storage/runner types."""

    item_id: str
    passed: bool
    score: float


@dataclass
class ABReport:
    baseline_name: str
    candidate_name: str
    n_common: int
    n_baseline_only: int
    n_candidate_only: int
    pass_rate_test: ZTestResult
    baseline_pass_ci: tuple[float, float]
    candidate_pass_ci: tuple[float, float]
    score_test: TTestResult
    score_bootstrap_ci: tuple[float, float]
    alpha: float
    verdict: Verdict
    n_regressions: int = 0
    n_improvements: int = 0
    regressed_ids: list[str] = field(default_factory=list)
    improved_ids: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def compare(
    baseline: Sequence[ItemOutcome],
    candidate: Sequence[ItemOutcome],
    *,
    baseline_name: str = "baseline",
    candidate_name: str = "candidate",
    alpha: float = 0.05,
    bootstrap_seed: int = 0,
) -> ABReport:
    """Compare two runs' outcomes; raises ``ValueError`` when comparison is impossible."""
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be strictly between 0 and 1")
    baseline_by_id = {outcome.item_id: outcome for outcome in baseline}
    candidate_by_id = {outcome.item_id: outcome for outcome in candidate}
    common_ids = sorted(baseline_by_id.keys() & candidate_by_id.keys())
    if len(common_ids) < 2:
        raise ValueError(
            "runs share fewer than two items — they must be produced from the same dataset"
        )

    base = [baseline_by_id[item_id] for item_id in common_ids]
    cand = [candidate_by_id[item_id] for item_id in common_ids]
    n = len(common_ids)

    pass_rate_test = two_proportion_z_test(
        successes_a=sum(outcome.passed for outcome in base),
        n_a=n,
        successes_b=sum(outcome.passed for outcome in cand),
        n_b=n,
        confidence=1.0 - alpha,
    )
    base_scores = [outcome.score for outcome in base]
    cand_scores = [outcome.score for outcome in cand]
    score_test = welch_t_test(base_scores, cand_scores, confidence=1.0 - alpha)

    regressed_ids = [
        item_id
        for item_id, b, c in zip(common_ids, base, cand, strict=True)
        if b.passed and not c.passed
    ]
    improved_ids = [
        item_id
        for item_id, b, c in zip(common_ids, base, cand, strict=True)
        if not b.passed and c.passed
    ]

    report = ABReport(
        baseline_name=baseline_name,
        candidate_name=candidate_name,
        n_common=n,
        n_baseline_only=len(baseline_by_id) - n,
        n_candidate_only=len(candidate_by_id) - n,
        pass_rate_test=pass_rate_test,
        baseline_pass_ci=wilson_interval(
            sum(outcome.passed for outcome in base), n, confidence=1.0 - alpha
        ),
        candidate_pass_ci=wilson_interval(
            sum(outcome.passed for outcome in cand), n, confidence=1.0 - alpha
        ),
        score_test=score_test,
        score_bootstrap_ci=bootstrap_mean_diff_ci(base_scores, cand_scores, seed=bootstrap_seed),
        alpha=alpha,
        verdict=_verdict(pass_rate_test, score_test, alpha),
        n_regressions=len(regressed_ids),
        n_improvements=len(improved_ids),
        regressed_ids=regressed_ids,
        improved_ids=improved_ids,
    )
    _collect_warnings(report)
    return report


def _verdict(pass_rate_test: ZTestResult, score_test: TTestResult, alpha: float) -> Verdict:
    if pass_rate_test.p_value < alpha and pass_rate_test.diff != 0.0:
        return "candidate_better" if pass_rate_test.diff > 0.0 else "baseline_better"
    if score_test.p_value < alpha and score_test.diff != 0.0:
        return "candidate_better" if score_test.diff > 0.0 else "baseline_better"
    return "inconclusive"


def _collect_warnings(report: ABReport) -> None:
    if report.n_common < MIN_RECOMMENDED_SAMPLE:
        report.warnings.append(
            f"only {report.n_common} shared items — below the recommended minimum of "
            f"{MIN_RECOMMENDED_SAMPLE}; the tests are underpowered"
        )
    dropped = report.n_baseline_only + report.n_candidate_only
    if dropped:
        report.warnings.append(
            f"{dropped} item(s) present in only one run were excluded from the comparison"
        )
