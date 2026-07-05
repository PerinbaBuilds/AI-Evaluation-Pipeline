"""A/B comparison: pairing, verdicts, warnings."""

from __future__ import annotations

import pytest

from evalpipe.ab import ItemOutcome, compare


def outcomes(
    pass_flags: list[bool], scores: list[float] | None = None, prefix: str = "item"
) -> list[ItemOutcome]:
    scores = scores or [1.0 if flag else 0.0 for flag in pass_flags]
    return [
        ItemOutcome(item_id=f"{prefix}-{index:04d}", passed=flag, score=score)
        for index, (flag, score) in enumerate(zip(pass_flags, scores, strict=True))
    ]


def test_clear_improvement_detected() -> None:
    baseline = outcomes([True] * 50 + [False] * 50)
    candidate = outcomes([True] * 80 + [False] * 20)
    report = compare(baseline, candidate)
    assert report.verdict == "candidate_better"
    assert report.pass_rate_test.p_value < 0.05
    assert report.n_common == 100
    assert not report.warnings


def test_clear_regression_detected() -> None:
    baseline = outcomes([True] * 80 + [False] * 20)
    candidate = outcomes([True] * 50 + [False] * 50)
    report = compare(baseline, candidate)
    assert report.verdict == "baseline_better"


def test_identical_runs_inconclusive() -> None:
    baseline = outcomes([True] * 30 + [False] * 10)
    candidate = outcomes([True] * 30 + [False] * 10)
    report = compare(baseline, candidate)
    assert report.verdict == "inconclusive"
    assert report.pass_rate_test.p_value == 1.0


def test_tiny_difference_is_inconclusive() -> None:
    baseline = outcomes([True] * 20 + [False] * 20)
    candidate = outcomes([True] * 21 + [False] * 19)
    report = compare(baseline, candidate)
    assert report.verdict == "inconclusive"


def test_score_test_breaks_pass_rate_tie() -> None:
    # Same pass flags, but the candidate's scores are consistently higher.
    flags = [True] * 20 + [False] * 20
    baseline = compare_scores = [0.55 if flag else 0.30 for flag in flags]
    candidate_scores = [0.95 if flag else 0.42 for flag in flags]
    report = compare(
        outcomes(flags, compare_scores),
        outcomes(flags, candidate_scores),
    )
    assert report.pass_rate_test.p_value == 1.0
    assert report.score_test.p_value < 0.05
    assert report.verdict == "candidate_better"
    del baseline


def test_small_sample_warning() -> None:
    baseline = outcomes([True, False] * 5)
    candidate = outcomes([True] * 10)
    report = compare(baseline, candidate)
    assert any("underpowered" in warning for warning in report.warnings)


def test_unpaired_items_are_excluded_with_warning() -> None:
    baseline = outcomes([True] * 40)
    candidate = outcomes([True] * 35)  # first 35 ids overlap
    report = compare(baseline, candidate)
    assert report.n_common == 35
    assert report.n_baseline_only == 5
    assert any("excluded" in warning for warning in report.warnings)


def test_disjoint_runs_rejected() -> None:
    baseline = outcomes([True] * 5, prefix="alpha")
    candidate = outcomes([True] * 5, prefix="beta")
    with pytest.raises(ValueError, match="fewer than two items"):
        compare(baseline, candidate)


def test_alpha_validation() -> None:
    baseline = outcomes([True] * 5)
    candidate = outcomes([True] * 5)
    with pytest.raises(ValueError, match="alpha"):
        compare(baseline, candidate, alpha=1.5)


def test_pairing_is_by_item_id_not_position() -> None:
    baseline = [
        ItemOutcome(item_id="a", passed=True, score=1.0),
        ItemOutcome(item_id="b", passed=False, score=0.0),
        ItemOutcome(item_id="c", passed=True, score=1.0),
    ]
    candidate = [  # same items, different order
        ItemOutcome(item_id="c", passed=True, score=1.0),
        ItemOutcome(item_id="a", passed=True, score=1.0),
        ItemOutcome(item_id="b", passed=False, score=0.0),
    ]
    report = compare(baseline, candidate)
    assert report.pass_rate_test.diff == 0.0
    assert report.score_test.diff == 0.0


def test_regression_diff_identifies_flipped_items() -> None:
    baseline = outcomes([True, True, True, True, False, False])
    candidate = outcomes([True, False, False, False, True, True])
    report = compare(baseline, candidate)
    assert report.n_regressions == 3
    assert report.regressed_ids == ["item-0001", "item-0002", "item-0003"]
    assert report.n_improvements == 2
    assert report.improved_ids == ["item-0004", "item-0005"]


def test_no_flips_when_outcomes_identical() -> None:
    flags = [True, False, True, False]
    report = compare(outcomes(flags), outcomes(flags))
    assert report.n_regressions == 0
    assert report.n_improvements == 0
