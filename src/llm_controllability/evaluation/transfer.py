"""Tests linking reachable-subspace overlap to intervention transfer."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.shape[0], dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def _correlation(first: np.ndarray, second: np.ndarray) -> float:
    if first.size < 2 or np.std(first) == 0 or np.std(second) == 0:
        return float("nan")
    return float(np.corrcoef(first, second)[0, 1])


def overlap_transfer_association(
    overlaps: Sequence[float],
    transfer_scores: Sequence[float],
) -> dict[str, float | int]:
    """Correlation and one-variable predictive fit for the central hypothesis."""

    x = np.asarray(overlaps, dtype=np.float64)
    y = np.asarray(transfer_scores, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError("overlaps and transfer scores must be matching vectors")
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if x.size < 2:
        return {
            "n": int(x.size),
            "pearson_r": float("nan"),
            "spearman_r": float("nan"),
            "slope": float("nan"),
            "intercept": float("nan"),
            "r_squared": float("nan"),
        }
    slope, intercept = np.polyfit(x, y, 1)
    prediction = slope * x + intercept
    residual = np.square(y - prediction).sum()
    total = np.square(y - y.mean()).sum()
    return {
        "n": int(x.size),
        "pearson_r": _correlation(x, y),
        "spearman_r": _correlation(_rankdata(x), _rankdata(y)),
        "slope": float(slope),
        "intercept": float(intercept),
        "r_squared": float(1.0 - residual / total) if total > 0 else 1.0,
    }


def controlled_overlap_association(
    overlaps: Sequence[float],
    transfer_scores: Sequence[float],
    covariates: Sequence[Sequence[float]],
) -> dict[str, float | int]:
    """OLS effect of overlap after standardizing nonconstant covariates."""

    overlap = np.asarray(overlaps, dtype=np.float64)
    target = np.asarray(transfer_scores, dtype=np.float64)
    controls = np.asarray(covariates, dtype=np.float64)
    if overlap.size == 0 and controls.size == 0:
        controls = controls.reshape(0, 0)
    if overlap.ndim != 1 or target.shape != overlap.shape:
        raise ValueError("overlaps and transfer scores must be matching vectors")
    if controls.ndim != 2 or controls.shape[0] != overlap.shape[0]:
        raise ValueError("covariates must be shaped [observations, covariates]")
    finite = (
        np.isfinite(overlap)
        & np.isfinite(target)
        & np.isfinite(controls).all(axis=1)
    )
    overlap = overlap[finite]
    target = target[finite]
    controls = controls[finite]
    if overlap.size < 3 or np.std(overlap) == 0:
        return {
            "n_controlled": int(overlap.size),
            "overlap_standardized_beta": float("nan"),
            "controlled_r_squared": float("nan"),
            "overlap_partial_r_squared": float("nan"),
            "design_rank": 0,
        }

    raw = np.column_stack([overlap, controls])
    keep = np.std(raw, axis=0) > 0
    keep[0] = True
    standardized = (raw[:, keep] - raw[:, keep].mean(axis=0)) / raw[
        :, keep
    ].std(axis=0)
    design = np.column_stack([np.ones(overlap.size), standardized])
    beta, _, rank, _ = np.linalg.lstsq(design, target, rcond=None)
    prediction = design @ beta
    residual = float(np.square(target - prediction).sum())
    total = float(np.square(target - target.mean()).sum())

    reduced = np.column_stack([np.ones(overlap.size), standardized[:, 1:]])
    reduced_beta, _, _, _ = np.linalg.lstsq(reduced, target, rcond=None)
    reduced_residual = float(np.square(target - reduced @ reduced_beta).sum())
    return {
        "n_controlled": int(overlap.size),
        "overlap_standardized_beta": float(beta[1]),
        "controlled_r_squared": (
            float(1.0 - residual / total) if total > 0 else 1.0
        ),
        "overlap_partial_r_squared": (
            float((reduced_residual - residual) / reduced_residual)
            if reduced_residual > 0
            else 0.0
        ),
        "design_rank": int(rank),
    }
