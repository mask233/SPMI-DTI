"""Small dependency-free paired-statistics helpers."""

from __future__ import annotations

import math
from itertools import product
from typing import Sequence


def mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cannot average an empty sequence")
    return sum(values) / len(values)


def sample_sd(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    center = mean(values)
    return math.sqrt(sum((value - center) ** 2 for value in values) / (len(values) - 1))


def _beta_continued_fraction(a: float, b: float, x: float) -> float:
    max_iterations = 300
    epsilon = 3e-14
    tiny = 1e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    result = d
    for iteration in range(1, max_iterations + 1):
        twice = 2 * iteration
        numerator = iteration * (b - iteration) * x / ((qam + twice) * (a + twice))
        d = 1.0 + numerator * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + numerator / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        result *= d * c
        numerator = -(a + iteration) * (qab + iteration) * x / ((a + twice) * (qap + twice))
        d = 1.0 + numerator * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + numerator / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        result *= delta
        if abs(delta - 1.0) < epsilon:
            return result
    raise ArithmeticError("incomplete beta continued fraction did not converge")


def regularized_incomplete_beta(x: float, a: float, b: float) -> float:
    if not 0.0 <= x <= 1.0:
        raise ValueError("x must be in [0, 1]")
    if x in (0.0, 1.0):
        return x
    factor = math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        + a * math.log(x) + b * math.log1p(-x)
    )
    threshold = (a + 1.0) / (a + b + 2.0)
    if x < threshold:
        return factor * _beta_continued_fraction(a, b, x) / a
    return 1.0 - factor * _beta_continued_fraction(b, a, 1.0 - x) / b


def paired_t_two_sided_p(t_abs: float, degrees_freedom: int) -> float:
    if degrees_freedom < 1:
        raise ValueError("degrees_freedom must be positive")
    if math.isinf(t_abs):
        return 0.0
    x = degrees_freedom / (degrees_freedom + t_abs * t_abs)
    return regularized_incomplete_beta(x, degrees_freedom / 2.0, 0.5)


def t_critical_975(degrees_freedom: int) -> float:
    low, high = 0.0, 100.0
    for _ in range(100):
        middle = (low + high) / 2.0
        if paired_t_two_sided_p(middle, degrees_freedom) > 0.05:
            low = middle
        else:
            high = middle
    return (low + high) / 2.0


def _average_absolute_ranks(values: Sequence[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: abs(item[1]))
    ranks = [0.0] * len(values)
    index = 0
    while index < len(indexed):
        end = index + 1
        while end < len(indexed) and abs(indexed[end][1]) == abs(indexed[index][1]):
            end += 1
        average_rank = ((index + 1) + end) / 2.0
        for original_index, _ in indexed[index:end]:
            ranks[original_index] = average_rank
        index = end
    return ranks


def exact_wilcoxon_signed_rank_p(differences: Sequence[float]) -> float | None:
    nonzero = [value for value in differences if value != 0.0]
    if not nonzero:
        return 1.0
    if len(nonzero) > 20:
        return None
    ranks = _average_absolute_ranks(nonzero)
    observed = abs(sum(rank if value > 0 else -rank for value, rank in zip(nonzero, ranks)))
    extreme = total = 0
    for signs in product((-1.0, 1.0), repeat=len(ranks)):
        statistic = abs(sum(sign * rank for sign, rank in zip(signs, ranks)))
        total += 1
        if statistic >= observed - 1e-12:
            extreme += 1
    return extreme / total


def paired_summary(reference: Sequence[float], comparator: Sequence[float]) -> dict[str, float | int | None]:
    if len(reference) != len(comparator) or len(reference) < 2:
        raise ValueError("paired statistics require at least two aligned pairs")
    differences = [left - right for left, right in zip(reference, comparator)]
    count = len(differences)
    center = mean(differences)
    sd = sample_sd(differences)
    standard_error = sd / math.sqrt(count)
    if sd == 0.0:
        t_statistic = 0.0 if center == 0.0 else math.copysign(math.inf, center)
        p_value = 1.0 if center == 0.0 else 0.0
        margin = 0.0
        cohen_dz = 0.0 if center == 0.0 else math.copysign(math.inf, center)
    else:
        t_statistic = center / standard_error
        p_value = paired_t_two_sided_p(abs(t_statistic), count - 1)
        margin = t_critical_975(count - 1) * standard_error
        cohen_dz = center / sd
    return {
        "n_pairs": count,
        "mean_difference": center,
        "sample_sd_difference": sd,
        "ci95_low": center - margin,
        "ci95_high": center + margin,
        "paired_t": t_statistic,
        "paired_t_p": p_value,
        "cohen_dz": cohen_dz,
        "wilcoxon_exact_p": exact_wilcoxon_signed_rank_p(differences),
    }

