"""Small paired statistical tests with no SciPy dependency."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np


def paired_bootstrap_interval(
    first: Sequence[float],
    second: Sequence[float],
    *,
    statistic: Callable[[np.ndarray], float] = np.mean,
    confidence: float = 0.95,
    resamples: int = 10_000,
    seed: int = 0,
) -> dict[str, float]:
    a = np.asarray(first, dtype=np.float64)
    b = np.asarray(second, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 1 or a.size == 0:
        raise ValueError("paired bootstrap requires nonempty matching vectors")
    if not 0 < confidence < 1:
        raise ValueError("confidence must lie in (0, 1)")
    differences = a - b
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(differences), size=(resamples, len(differences)))
    estimates = np.asarray([statistic(differences[index]) for index in indices])
    alpha = (1.0 - confidence) / 2.0
    return {
        "estimate": float(statistic(differences)),
        "lower": float(np.quantile(estimates, alpha)),
        "upper": float(np.quantile(estimates, 1.0 - alpha)),
    }


def paired_permutation_test(
    first: Sequence[float],
    second: Sequence[float],
    *,
    permutations: int = 100_000,
    seed: int = 0,
) -> dict[str, float | int]:
    a = np.asarray(first, dtype=np.float64)
    b = np.asarray(second, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 1 or a.size == 0:
        raise ValueError("paired permutation test requires nonempty matching vectors")
    differences = a - b
    observed = abs(float(differences.mean()))
    rng = np.random.default_rng(seed)
    exceedances = 0
    for _ in range(permutations):
        signs = rng.choice((-1.0, 1.0), size=differences.shape[0])
        exceedances += abs(float(np.mean(signs * differences))) >= observed
    return {
        "mean_difference": float(differences.mean()),
        "p_value": (exceedances + 1) / (permutations + 1),
        "permutations": permutations,
    }


def adjust_pvalues(
    pvalues: Sequence[float],
    *,
    method: str,
) -> np.ndarray:
    """Adjust a family of p values with Holm or Benjamini-Hochberg."""

    values = np.asarray(pvalues, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("p values must be one-dimensional")
    if np.any((values < 0) | (values > 1) | ~np.isfinite(values)):
        raise ValueError("p values must be finite and lie in [0, 1]")
    count = len(values)
    if count == 0:
        return values.copy()
    order = np.argsort(values)
    ordered = values[order]
    if method == "holm":
        adjusted_ordered = np.maximum.accumulate(
            (count - np.arange(count)) * ordered
        )
    elif method in {"benjamini-hochberg", "bh"}:
        adjusted_ordered = np.minimum.accumulate(
            (count / np.arange(count, 0, -1)) * ordered[::-1]
        )[::-1]
    else:
        raise ValueError("method must be 'holm' or 'benjamini-hochberg'")
    adjusted = np.empty(count, dtype=np.float64)
    adjusted[order] = np.clip(adjusted_ordered, 0.0, 1.0)
    return adjusted


def paired_standardized_effect(
    first: Sequence[float],
    second: Sequence[float],
) -> float:
    """Cohen's dz for paired observations."""

    a = np.asarray(first, dtype=np.float64)
    b = np.asarray(second, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 1 or a.size == 0:
        raise ValueError("paired effect size requires nonempty matching vectors")
    differences = a - b
    standard_deviation = differences.std(ddof=1) if differences.size > 1 else 0.0
    if standard_deviation == 0:
        return (
            float(np.sign(differences.mean()) * np.inf)
            if differences.mean() != 0
            else 0.0
        )
    return float(differences.mean() / standard_deviation)
