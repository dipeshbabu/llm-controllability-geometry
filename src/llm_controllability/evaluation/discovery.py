"""Conservative qualification of cross-model controllability trends."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

_SCALING_ENDPOINTS = {
    ("geometry", "effective_rank"): "log_log",
    ("geometry", "participation_ratio"): "log_log",
    ("controllability_atlas", "effective_rank"): "log_log",
    ("controllability_atlas", "maximum_state_displacement"): "log_log",
    ("controllability_boundaries", "mean_maximum_state_displacement"): "log_log",
    ("detection_control_gap", "control_to_detection_ratio"): "log_log",
    ("representation_control_gap", "standardized_detection_margin"): "log_log",
    ("representation_control_gap", "standardized_control_margin"): "log_log",
    ("representation_control_gap", "representation_control_gap"): "linear_log_size",
    ("cmap_summary", "effective_rank"): "log_log",
    ("cmap_summary", "maximum_state_displacement"): "log_log",
    ("jacobians", "rank_fraction"): "linear_log_size",
    ("jacobians", "squared_gain"): "log_log",
}

_GROUP_FIELDS = (
    "channel",
    "direction",
    "family",
    "side",
    "control_dimension",
    "role",
)


def _fit_trend(
    sizes: np.ndarray,
    values: np.ndarray,
    *,
    transform: str,
) -> tuple[float, float, float]:
    x = np.log(sizes)
    y = np.log(values) if transform == "log_log" else values
    design = np.column_stack([np.ones_like(x), x])
    intercept, slope = np.linalg.lstsq(design, y, rcond=None)[0]
    fitted = design @ np.asarray([intercept, slope])
    residual = float(np.square(y - fitted).sum())
    total = float(np.square(y - y.mean()).sum())
    r_squared = 1.0 - residual / total if total > 0 else float("nan")
    return float(intercept), float(slope), r_squared


def _rank_correlation(first: np.ndarray, second: np.ndarray) -> float:
    if first.size < 2:
        return float("nan")
    first_rank = _average_ranks(first)
    second_rank = _average_ranks(second)
    if np.all(first_rank == first_rank[0]) or np.all(second_rank == second_rank[0]):
        return float("nan")
    return float(np.corrcoef(first_rank, second_rank)[0, 1])


def _average_ranks(values: np.ndarray) -> np.ndarray:
    ranks = np.empty(values.size, dtype=np.float64)
    order = np.argsort(values, kind="stable")
    ordered = values[order]
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and ordered[end] == ordered[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def scaling_diagnostic_rows(
    summaries: Sequence[Mapping[str, Any]],
    models: Sequence[Mapping[str, Any]],
    *,
    minimum_scale_points: int = 3,
    confirmatory_scale_points: int = 4,
    minimum_r_squared: float = 0.80,
) -> list[dict[str, Any]]:
    """Fit declared trends and state when the model matrix cannot support a law."""

    if minimum_scale_points < 2:
        raise ValueError("minimum_scale_points must be at least two")
    if confirmatory_scale_points < minimum_scale_points:
        raise ValueError(
            "confirmatory_scale_points must be at least minimum_scale_points"
        )
    metadata = {str(model["slug"]): model for model in models}
    grouped: dict[
        tuple[str, str, str, str, str, str, str, str, str, str],
        list[tuple[float, float, str]],
    ] = defaultdict(list)
    for row in summaries:
        endpoint = (str(row["table"]), str(row["metric"]))
        transform = _SCALING_ENDPOINTS.get(endpoint)
        if transform is None:
            continue
        model = metadata[str(row["slug"])]
        size = float(model["parameter_count_billions"])
        value = float(row["mean"])
        if not np.isfinite(size) or size <= 0 or not np.isfinite(value):
            continue
        if transform == "log_log" and value <= 0:
            continue
        grouped[
            (
                str(model["family"]),
                str(model["training_regime"]),
                endpoint[0],
                endpoint[1],
                *(str(row.get(field, "")) for field in _GROUP_FIELDS),
            )
        ].append((size, value, str(row["slug"])))

    rows = []
    for key, observations in sorted(grouped.items()):
        (
            model_family,
            training_regime,
            table,
            metric,
            channel,
            direction,
            intervention_family,
            side,
            control_dimension,
            role,
        ) = key
        by_size: dict[float, list[tuple[float, str]]] = defaultdict(list)
        for size, value, slug in observations:
            by_size[size].append((value, slug))
        sizes = np.asarray(sorted(by_size), dtype=np.float64)
        values = np.asarray(
            [np.mean([value for value, _ in by_size[size]]) for size in sizes],
            dtype=np.float64,
        )
        transform = _SCALING_ENDPOINTS[(table, metric)]
        if sizes.size >= 2:
            intercept, slope, r_squared = _fit_trend(
                sizes,
                values,
                transform=transform,
            )
            rank_correlation = _rank_correlation(sizes, values)
        else:
            intercept = slope = r_squared = rank_correlation = float("nan")

        leave_one_out_slopes = []
        if sizes.size >= 4:
            for held_out in range(sizes.size):
                keep = np.arange(sizes.size) != held_out
                _, held_out_slope, _ = _fit_trend(
                    sizes[keep],
                    values[keep],
                    transform=transform,
                )
                leave_one_out_slopes.append(held_out_slope)
        sign_stable = bool(
            leave_one_out_slopes
            and np.all(np.sign(leave_one_out_slopes) == np.sign(slope))
        )
        if sizes.size < minimum_scale_points:
            qualification = "insufficient_scale_points"
        elif sizes.size < confirmatory_scale_points:
            qualification = "exploratory_three_point_trend"
        elif (
            np.isfinite(r_squared)
            and r_squared >= minimum_r_squared
            and sign_stable
        ):
            qualification = "within_family_scaling_candidate"
        else:
            qualification = "no_stable_scaling_trend"
        rows.append(
            {
                "model_family": model_family,
                "training_regime": training_regime,
                "table": table,
                "metric": metric,
                "channel": channel,
                "direction": direction,
                "family": intervention_family,
                "side": side,
                "control_dimension": control_dimension,
                "role": role,
                "transform": transform,
                "n_checkpoints": len(observations),
                "n_unique_scales": sizes.size,
                "minimum_size_billions": float(sizes.min()),
                "maximum_size_billions": float(sizes.max()),
                "intercept": intercept,
                "scaling_exponent_or_slope": slope,
                "r_squared": r_squared,
                "rank_correlation": rank_correlation,
                "leave_one_out_slope_min": (
                    float(np.min(leave_one_out_slopes))
                    if leave_one_out_slopes
                    else float("nan")
                ),
                "leave_one_out_slope_max": (
                    float(np.max(leave_one_out_slopes))
                    if leave_one_out_slopes
                    else float("nan")
                ),
                "leave_one_out_sign_stable": int(sign_stable),
                "qualification": qualification,
                "claim_is_confirmatory": 0,
            }
        )
    return rows


def scaling_replication_rows(
    diagnostics: Sequence[Mapping[str, Any]],
    *,
    minimum_families: int = 2,
) -> list[dict[str, Any]]:
    """Check whether a within-family trend has an independent family replication."""

    grouped: dict[tuple[str, ...], list[Mapping[str, Any]]] = defaultdict(list)
    fields = (
        "training_regime",
        "table",
        "metric",
        "channel",
        "direction",
        "family",
        "side",
        "control_dimension",
        "role",
        "transform",
    )
    for row in diagnostics:
        grouped[tuple(str(row[field]) for field in fields)].append(row)

    rows = []
    for key, group in sorted(grouped.items()):
        candidates = [
            row
            for row in group
            if row["qualification"] == "within_family_scaling_candidate"
        ]
        positive = sum(float(row["scaling_exponent_or_slope"]) > 0 for row in candidates)
        negative = sum(float(row["scaling_exponent_or_slope"]) < 0 for row in candidates)
        replicated = len(candidates) >= minimum_families and max(positive, negative) == len(
            candidates
        )
        payload = dict(zip(fields, key))
        rows.append(
            {
                **payload,
                "n_families_evaluated": len(group),
                "n_qualified_families": len(candidates),
                "n_positive_slopes": positive,
                "n_negative_slopes": negative,
                "replicated_direction": (
                    "increase" if replicated and positive else "decrease" if replicated else ""
                ),
                "qualification": (
                    "replicated_scaling_candidate"
                    if replicated
                    else "not_replicated"
                ),
                "claim_is_confirmatory": int(replicated),
            }
        )
    return rows
