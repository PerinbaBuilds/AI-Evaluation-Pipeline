"""Statistical primitives, implemented from first principles.

Everything here is dependency-free on purpose: the normal CDF comes from
``math.erf``, Student's t p-values from the regularized incomplete beta
function (evaluated with Lentz's continued-fraction algorithm), and quantile
functions from bisection on the monotonic CDFs. Each function is verified in
the test suite against published table values.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass

_EPS = 3e-12
_FPMIN = 1e-300
_MAX_ITERATIONS = 300

# ----------------------------------------------------------------- basic descriptives


def mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("mean() of an empty sequence")
    return sum(values) / len(values)


def sample_variance(values: Sequence[float]) -> float:
    if len(values) < 2:
        raise ValueError("sample_variance() needs at least two observations")
    center = mean(values)
    return sum((value - center) ** 2 for value in values) / (len(values) - 1)


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolation percentile, ``q`` in [0, 100]."""
    if not values:
        raise ValueError("percentile() of an empty sequence")
    if not 0.0 <= q <= 100.0:
        raise ValueError("q must be within [0, 100]")
    ordered = sorted(values)
    position = (len(ordered) - 1) * q / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


# ------------------------------------------------------------------------ distributions


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def normal_ppf(p: float) -> float:
    """Inverse standard normal CDF via bisection (the CDF is strictly increasing)."""
    if not 0.0 < p < 1.0:
        raise ValueError("p must be strictly between 0 and 1")
    return _bisect_increasing(normal_cdf, p, -40.0, 40.0)


def _bisect_increasing(fn: Callable[[float], float], target: float, lo: float, hi: float) -> float:
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if fn(mid) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta function (Lentz's method)."""
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < _FPMIN:
        d = _FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, _MAX_ITERATIONS + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < _FPMIN:
            d = _FPMIN
        c = 1.0 + aa / c
        if abs(c) < _FPMIN:
            c = _FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < _FPMIN:
            d = _FPMIN
        c = 1.0 + aa / c
        if abs(c) < _FPMIN:
            c = _FPMIN
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < _EPS:
            return h
    raise ArithmeticError("incomplete beta continued fraction failed to converge")


def regularized_incomplete_beta(a: float, b: float, x: float) -> float:
    if a <= 0.0 or b <= 0.0:
        raise ValueError("a and b must be positive")
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_front = (
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log(1.0 - x)
    )
    front = math.exp(log_front)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def student_t_two_sided_p(t: float, df: float) -> float:
    """Two-sided p-value for a t statistic with ``df`` degrees of freedom."""
    if df <= 0.0:
        raise ValueError("df must be positive")
    if math.isinf(t):
        return 0.0
    return regularized_incomplete_beta(df / 2.0, 0.5, df / (df + t * t))


def student_t_critical(df: float, alpha: float = 0.05) -> float:
    """Two-sided critical value: |t| beyond which p < alpha."""
    if df <= 0.0:
        raise ValueError("df must be positive")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be strictly between 0 and 1")
    return _bisect_increasing(
        lambda t: 1.0 - student_t_two_sided_p(t, df), 1.0 - alpha, 0.0, 1000.0
    )


# ------------------------------------------------------------------- hypothesis tests


@dataclass(frozen=True)
class TTestResult:
    """Welch's unequal-variance t-test on two independent samples."""

    mean_a: float
    mean_b: float
    diff: float  # mean_b - mean_a
    t_statistic: float
    df: float
    p_value: float
    ci_low: float
    ci_high: float
    cohen_d: float

    @property
    def significant(self) -> bool:
        return self.p_value < 0.05


def welch_t_test(a: Sequence[float], b: Sequence[float], confidence: float = 0.95) -> TTestResult:
    if len(a) < 2 or len(b) < 2:
        raise ValueError("welch_t_test() needs at least two observations per sample")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be strictly between 0 and 1")
    mean_a, mean_b = mean(a), mean(b)
    var_a, var_b = sample_variance(a), sample_variance(b)
    diff = mean_b - mean_a
    se_squared = var_a / len(a) + var_b / len(b)

    if se_squared == 0.0:
        # Both samples are constant: identical means are indistinguishable,
        # different means are trivially distinct.
        identical = diff == 0.0
        return TTestResult(
            mean_a=mean_a,
            mean_b=mean_b,
            diff=diff,
            t_statistic=0.0 if identical else math.inf,
            df=float(len(a) + len(b) - 2),
            p_value=1.0 if identical else 0.0,
            ci_low=diff,
            ci_high=diff,
            cohen_d=0.0 if identical else math.inf,
        )

    se = math.sqrt(se_squared)
    t_statistic = diff / se
    df = se_squared**2 / (
        (var_a / len(a)) ** 2 / (len(a) - 1) + (var_b / len(b)) ** 2 / (len(b) - 1)
    )
    p_value = student_t_two_sided_p(t_statistic, df)
    t_crit = student_t_critical(df, alpha=1.0 - confidence)
    pooled_var = ((len(a) - 1) * var_a + (len(b) - 1) * var_b) / (len(a) + len(b) - 2)
    cohen_d = diff / math.sqrt(pooled_var) if pooled_var > 0.0 else math.inf
    return TTestResult(
        mean_a=mean_a,
        mean_b=mean_b,
        diff=diff,
        t_statistic=t_statistic,
        df=df,
        p_value=p_value,
        ci_low=diff - t_crit * se,
        ci_high=diff + t_crit * se,
        cohen_d=cohen_d,
    )


@dataclass(frozen=True)
class ZTestResult:
    """Two-proportion z-test (pooled SE for the test, unpooled for the CI)."""

    proportion_a: float
    proportion_b: float
    diff: float  # proportion_b - proportion_a
    z_statistic: float
    p_value: float
    ci_low: float
    ci_high: float
    cohen_h: float

    @property
    def significant(self) -> bool:
        return self.p_value < 0.05


def two_proportion_z_test(
    successes_a: int,
    n_a: int,
    successes_b: int,
    n_b: int,
    confidence: float = 0.95,
) -> ZTestResult:
    for successes, n, label in ((successes_a, n_a, "a"), (successes_b, n_b, "b")):
        if n <= 0:
            raise ValueError(f"sample {label} is empty")
        if not 0 <= successes <= n:
            raise ValueError(f"sample {label}: successes must be within [0, n]")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be strictly between 0 and 1")

    p_a, p_b = successes_a / n_a, successes_b / n_b
    diff = p_b - p_a
    pooled = (successes_a + successes_b) / (n_a + n_b)
    pooled_se = math.sqrt(pooled * (1.0 - pooled) * (1.0 / n_a + 1.0 / n_b))

    if pooled_se == 0.0:
        z_statistic, p_value = (0.0, 1.0) if diff == 0.0 else (math.inf, 0.0)
    else:
        z_statistic = diff / pooled_se
        p_value = 2.0 * (1.0 - normal_cdf(abs(z_statistic)))

    unpooled_se = math.sqrt(p_a * (1.0 - p_a) / n_a + p_b * (1.0 - p_b) / n_b)
    z_crit = normal_ppf(1.0 - (1.0 - confidence) / 2.0)
    return ZTestResult(
        proportion_a=p_a,
        proportion_b=p_b,
        diff=diff,
        z_statistic=z_statistic,
        p_value=min(1.0, p_value),
        ci_low=diff - z_crit * unpooled_se,
        ci_high=diff + z_crit * unpooled_se,
        cohen_h=cohens_h(p_a, p_b),
    )


def cohens_h(p_a: float, p_b: float) -> float:
    """Effect size for two proportions (arcsine transformation)."""
    for p in (p_a, p_b):
        if not 0.0 <= p <= 1.0:
            raise ValueError("proportions must be within [0, 1]")
    return 2.0 * math.asin(math.sqrt(p_b)) - 2.0 * math.asin(math.sqrt(p_a))


def wilson_interval(successes: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score interval — well behaved even at 0%/100% observed rates."""
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0 <= successes <= n:
        raise ValueError("successes must be within [0, n]")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be strictly between 0 and 1")
    z = normal_ppf(1.0 - (1.0 - confidence) / 2.0)
    p_hat = successes / n
    denominator = 1.0 + z * z / n
    center = (p_hat + z * z / (2.0 * n)) / denominator
    margin = (z / denominator) * math.sqrt(p_hat * (1.0 - p_hat) / n + z * z / (4.0 * n * n))
    low = 0.0 if successes == 0 else max(0.0, center - margin)
    high = 1.0 if successes == n else min(1.0, center + margin)
    return (low, high)


def bootstrap_mean_diff_ci(
    a: Sequence[float],
    b: Sequence[float],
    iterations: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap CI for ``mean(b) - mean(a)``; deterministic per seed."""
    if not a or not b:
        raise ValueError("bootstrap requires non-empty samples")
    if iterations < 100:
        raise ValueError("iterations must be at least 100")
    rng = random.Random(seed)
    diffs = sorted(
        mean(rng.choices(b, k=len(b))) - mean(rng.choices(a, k=len(a))) for _ in range(iterations)
    )
    tail = (1.0 - confidence) / 2.0
    return (percentile(diffs, tail * 100.0), percentile(diffs, (1.0 - tail) * 100.0))


def required_sample_size_two_proportions(
    baseline_rate: float,
    minimum_detectable_effect: float,
    alpha: float = 0.05,
    power: float = 0.8,
) -> int:
    """Per-arm sample size to detect an absolute lift in a pass rate."""
    if not 0.0 < baseline_rate < 1.0:
        raise ValueError("baseline_rate must be strictly between 0 and 1")
    if minimum_detectable_effect <= 0.0:
        raise ValueError("minimum_detectable_effect must be positive")
    target = baseline_rate + minimum_detectable_effect
    if target >= 1.0:
        raise ValueError("baseline_rate + minimum_detectable_effect must be below 1")
    if not 0.0 < alpha < 1.0 or not 0.0 < power < 1.0:
        raise ValueError("alpha and power must be strictly between 0 and 1")

    z_alpha = normal_ppf(1.0 - alpha / 2.0)
    z_power = normal_ppf(power)
    pooled = (baseline_rate + target) / 2.0
    numerator = (
        z_alpha * math.sqrt(2.0 * pooled * (1.0 - pooled))
        + z_power * math.sqrt(baseline_rate * (1.0 - baseline_rate) + target * (1.0 - target))
    ) ** 2
    return math.ceil(numerator / (minimum_detectable_effect**2))
