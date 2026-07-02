"""Statistical primitives verified against published table values."""

from __future__ import annotations

import math

import pytest

from evalpipe.stats import (
    bootstrap_mean_diff_ci,
    cohens_h,
    mean,
    normal_cdf,
    normal_ppf,
    percentile,
    regularized_incomplete_beta,
    required_sample_size_two_proportions,
    sample_variance,
    student_t_critical,
    student_t_two_sided_p,
    two_proportion_z_test,
    welch_t_test,
    wilson_interval,
)


class TestDescriptives:
    def test_mean(self) -> None:
        assert mean([1.0, 2.0, 3.0]) == 2.0

    def test_mean_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            mean([])

    def test_sample_variance(self) -> None:
        assert sample_variance([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]) == pytest.approx(32 / 7)

    def test_variance_needs_two_points(self) -> None:
        with pytest.raises(ValueError):
            sample_variance([1.0])

    def test_percentile_interpolation(self) -> None:
        assert percentile([1.0, 2.0, 3.0, 4.0], 50) == 2.5
        assert percentile([1.0, 2.0, 3.0, 4.0], 0) == 1.0
        assert percentile([1.0, 2.0, 3.0, 4.0], 100) == 4.0

    def test_percentile_bounds(self) -> None:
        with pytest.raises(ValueError):
            percentile([1.0], 101)


class TestDistributions:
    def test_normal_cdf_known_values(self) -> None:
        assert normal_cdf(0.0) == pytest.approx(0.5)
        assert normal_cdf(1.959964) == pytest.approx(0.975, abs=1e-6)
        assert normal_cdf(-1.959964) == pytest.approx(0.025, abs=1e-6)

    def test_normal_ppf_inverts_cdf(self) -> None:
        assert normal_ppf(0.975) == pytest.approx(1.959964, abs=1e-5)
        assert normal_ppf(0.5) == pytest.approx(0.0, abs=1e-9)
        for p in (0.01, 0.2, 0.8, 0.99):
            assert normal_cdf(normal_ppf(p)) == pytest.approx(p, abs=1e-9)

    def test_ppf_domain(self) -> None:
        with pytest.raises(ValueError):
            normal_ppf(0.0)

    def test_incomplete_beta_symmetry_point(self) -> None:
        assert regularized_incomplete_beta(0.5, 0.5, 0.5) == pytest.approx(0.5, abs=1e-9)

    def test_incomplete_beta_uniform_case(self) -> None:
        # I_x(1, 1) = x for the uniform distribution
        for x in (0.1, 0.35, 0.75):
            assert regularized_incomplete_beta(1.0, 1.0, x) == pytest.approx(x, abs=1e-10)

    def test_incomplete_beta_edges(self) -> None:
        assert regularized_incomplete_beta(2.0, 3.0, 0.0) == 0.0
        assert regularized_incomplete_beta(2.0, 3.0, 1.0) == 1.0

    def test_t_distribution_table_values(self) -> None:
        # Classic two-sided 5% critical values from t tables.
        assert student_t_two_sided_p(2.228, 10) == pytest.approx(0.05, abs=5e-4)
        assert student_t_two_sided_p(2.086, 20) == pytest.approx(0.05, abs=5e-4)
        assert student_t_two_sided_p(12.706, 1) == pytest.approx(0.05, abs=5e-4)

    def test_t_cauchy_case(self) -> None:
        # df=1 is the Cauchy distribution: P(|T| > 1) = 0.5 exactly.
        assert student_t_two_sided_p(1.0, 1) == pytest.approx(0.5, abs=1e-9)

    def test_t_critical_inverts_p(self) -> None:
        assert student_t_critical(10, alpha=0.05) == pytest.approx(2.228, abs=2e-3)
        assert student_t_critical(20, alpha=0.05) == pytest.approx(2.086, abs=2e-3)

    def test_t_infinite_statistic(self) -> None:
        assert student_t_two_sided_p(math.inf, 5) == 0.0


class TestWelch:
    def test_symmetric_samples(self) -> None:
        a = [1.0, 2.0, 3.0, 4.0, 5.0]
        b = [2.0, 3.0, 4.0, 5.0, 6.0]
        result = welch_t_test(a, b)
        assert result.t_statistic == pytest.approx(1.0, abs=1e-12)
        assert result.df == pytest.approx(8.0, abs=1e-9)
        assert result.p_value == pytest.approx(0.3466, abs=2e-3)
        assert result.diff == pytest.approx(1.0)
        assert result.ci_low < 1.0 < result.ci_high

    def test_direction_sign(self) -> None:
        result = welch_t_test([5.0, 6.0, 7.0], [1.0, 2.0, 3.0])
        assert result.diff < 0
        assert result.t_statistic < 0

    def test_identical_constant_samples(self) -> None:
        result = welch_t_test([2.0, 2.0, 2.0], [2.0, 2.0, 2.0])
        assert result.p_value == 1.0

    def test_different_constant_samples(self) -> None:
        result = welch_t_test([1.0, 1.0], [2.0, 2.0])
        assert result.p_value == 0.0
        assert math.isinf(result.t_statistic)

    def test_minimum_sample_size(self) -> None:
        with pytest.raises(ValueError):
            welch_t_test([1.0], [1.0, 2.0])

    def test_clearly_different_samples_significant(self) -> None:
        a = [0.1, 0.2, 0.15, 0.12, 0.18, 0.11, 0.16, 0.14]
        b = [0.8, 0.9, 0.85, 0.88, 0.82, 0.91, 0.86, 0.84]
        result = welch_t_test(a, b)
        assert result.p_value < 1e-6
        assert result.cohen_d > 2


class TestTwoProportions:
    def test_known_case(self) -> None:
        # 50% vs 60% at n=200 per arm: z ≈ 2.01, p ≈ 0.044.
        result = two_proportion_z_test(100, 200, 120, 200)
        assert result.z_statistic == pytest.approx(2.0101, abs=2e-3)
        assert result.p_value == pytest.approx(0.0444, abs=2e-3)

    def test_equal_proportions(self) -> None:
        result = two_proportion_z_test(50, 100, 50, 100)
        assert result.p_value == 1.0
        assert result.diff == 0.0

    def test_all_zero_and_all_one(self) -> None:
        assert two_proportion_z_test(0, 10, 0, 10).p_value == 1.0
        assert two_proportion_z_test(0, 10, 10, 10).p_value == pytest.approx(0.0, abs=1e-5)

    def test_sign_symmetry(self) -> None:
        forward = two_proportion_z_test(40, 100, 60, 100)
        backward = two_proportion_z_test(60, 100, 40, 100)
        assert forward.p_value == pytest.approx(backward.p_value)
        assert forward.diff == -backward.diff

    def test_input_validation(self) -> None:
        with pytest.raises(ValueError):
            two_proportion_z_test(5, 0, 1, 10)
        with pytest.raises(ValueError):
            two_proportion_z_test(11, 10, 1, 10)


class TestIntervalsAndPower:
    def test_wilson_contains_point_estimate(self) -> None:
        low, high = wilson_interval(8, 10)
        assert low < 0.8 < high
        assert low >= 0.0 and high <= 1.0

    def test_wilson_extreme_rates_stay_in_bounds(self) -> None:
        low_zero, high_zero = wilson_interval(0, 10)
        assert low_zero == 0.0 and high_zero > 0.0
        low_one, high_one = wilson_interval(10, 10)
        assert low_one < 1.0 and high_one == 1.0

    def test_wilson_narrows_with_n(self) -> None:
        small = wilson_interval(8, 10)
        large = wilson_interval(800, 1000)
        assert (large[1] - large[0]) < (small[1] - small[0])

    def test_bootstrap_ci_brackets_true_diff(self) -> None:
        a = [0.5] * 30
        b = [0.7] * 30
        low, high = bootstrap_mean_diff_ci(a, b, seed=1)
        assert low == pytest.approx(0.2) and high == pytest.approx(0.2)

    def test_bootstrap_deterministic_per_seed(self) -> None:
        a = [0.1, 0.4, 0.35, 0.6, 0.2, 0.55]
        b = [0.5, 0.7, 0.65, 0.9, 0.6, 0.8]
        assert bootstrap_mean_diff_ci(a, b, seed=7) == bootstrap_mean_diff_ci(a, b, seed=7)
        assert bootstrap_mean_diff_ci(a, b, seed=7) != bootstrap_mean_diff_ci(a, b, seed=8)

    def test_bootstrap_validation(self) -> None:
        with pytest.raises(ValueError):
            bootstrap_mean_diff_ci([], [1.0])
        with pytest.raises(ValueError):
            bootstrap_mean_diff_ci([1.0], [1.0], iterations=10)

    def test_cohens_h_values(self) -> None:
        assert cohens_h(0.5, 0.5) == 0.0
        assert cohens_h(0.2, 0.3) == pytest.approx(0.2319, abs=1e-3)
        assert cohens_h(0.3, 0.2) == pytest.approx(-0.2319, abs=1e-3)

    def test_sample_size_standard_case(self) -> None:
        # Detecting a 10-point lift from a 50% baseline at alpha=.05, power=.8.
        assert required_sample_size_two_proportions(0.5, 0.1) == 388

    def test_sample_size_grows_for_smaller_effects(self) -> None:
        big = required_sample_size_two_proportions(0.5, 0.05)
        small = required_sample_size_two_proportions(0.5, 0.2)
        assert big > small

    def test_sample_size_validation(self) -> None:
        with pytest.raises(ValueError):
            required_sample_size_two_proportions(0.95, 0.1)
        with pytest.raises(ValueError):
            required_sample_size_two_proportions(0.0, 0.1)
