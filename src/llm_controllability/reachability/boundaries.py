"""Finite-sample controllability boundaries and cross-channel access gaps."""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from llm_controllability.controllability.types import ControlChannel, StateSample

_NUMERIC_PARAMETERS = (
    ("strength", "_a"),
    ("fraction", "_f"),
    ("setpoint", "_s"),
)


def _bootstrap_mean_interval(
    values: Sequence[float],
    *,
    resamples: int,
    seed: int,
    confidence: float = 0.95,
) -> tuple[float, float, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan"), float("nan"), float("nan")
    estimate = float(finite.mean())
    if finite.size == 1 or resamples <= 0:
        return estimate, estimate, estimate
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, finite.size, size=(resamples, finite.size))
    estimates = finite[indices].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return (
        estimate,
        float(np.quantile(estimates, alpha)),
        float(np.quantile(estimates, 1.0 - alpha)),
    )


def _baseline_index(
    samples: Sequence[StateSample],
) -> dict[tuple[str, str, int], StateSample]:
    return {
        (sample.model_name, sample.example_id, sample.layer): sample
        for sample in samples
        if sample.intervention.channel is ControlChannel.BASELINE
    }


def _control_family(sample: StateSample) -> tuple[str, str | None, float | None]:
    parameters = sample.intervention.parameters
    for parameter, marker in _NUMERIC_PARAMETERS:
        if parameter in parameters:
            return (
                sample.intervention.name.rsplit(marker, 1)[0],
                parameter,
                float(parameters[parameter]),
            )
    family = re.sub(r"_\d{4}$", "", sample.intervention.name)
    return family, None, None


def _side(value: float) -> str:
    if value > 0:
        return "increase"
    if value < 0:
        return "decrease"
    return "zero"


def controllability_boundary_rows(
    samples: Sequence[StateSample],
    *,
    target_metric: str = "target_projection",
) -> list[dict[str, Any]]:
    """Bracket behavior-preserving limits along declared control sweeps.

    Numeric strength, fraction, and setpoint sweeps are ordered and can bracket a
    boundary. Discrete prompt candidates are reported as sample-limited envelopes
    instead of being presented as an ordered boundary.
    """

    baselines = _baseline_index(samples)
    grouped: dict[
        tuple[str, int, str, str, str, str, str],
        list[tuple[StateSample, float, float, float]],
    ] = defaultdict(list)
    for sample in samples:
        if sample.intervention.channel is ControlChannel.BASELINE:
            continue
        key = (sample.model_name, sample.example_id, sample.layer)
        baseline = baselines.get(key)
        if baseline is None:
            raise ValueError(
                f"missing baseline for example {sample.example_id!r}, layer {sample.layer}"
            )
        if target_metric not in sample.metrics or target_metric not in baseline.metrics:
            continue
        target_displacement = float(
            sample.metrics[target_metric] - baseline.metrics[target_metric]
        )
        state_displacement = float(np.linalg.norm(sample.state - baseline.state))
        family, parameter, parameter_value = _control_family(sample)
        if parameter_value is None:
            coordinate = sample.intervention.control_cost
            side = _side(target_displacement)
            parameter_name = "control_cost"
        else:
            baseline_target = float(baseline.metrics[target_metric])
            signed_coordinate = (
                parameter_value - baseline_target
                if parameter == "setpoint"
                else parameter_value
            )
            coordinate = abs(signed_coordinate)
            side = _side(signed_coordinate)
            parameter_name = parameter
        if side == "zero":
            continue
        grouped[
            (
                sample.model_name,
                sample.layer,
                sample.intervention.channel.value,
                sample.example_id,
                family,
                parameter_name,
                side,
            )
        ].append((sample, coordinate, target_displacement, state_displacement))

    rows: list[dict[str, Any]] = []
    for (
        model_name,
        layer,
        channel,
        example_id,
        family,
        parameter_name,
        side,
    ), values in sorted(grouped.items()):
        ordered_control = parameter_name != "control_cost"
        values.sort(key=lambda item: item[1])
        preserved = [item for item in values if item[0].behavior_preserved]
        failed = [item for item in values if not item[0].behavior_preserved]
        lower = max((item[1] for item in preserved), default=0.0)
        failures_beyond = [item for item in failed if item[1] > lower]
        upper_item = min(failures_beyond, key=lambda item: item[1], default=None)

        if not ordered_control:
            status = "sample_limited"
        elif not preserved:
            status = "no_feasible_control"
        elif upper_item is None:
            status = "right_censored"
        else:
            status = "bracketed"

        violations = sum(
            failed_item[1] < passed_item[1]
            for failed_item in failed
            for passed_item in preserved
        )
        binding = ""
        if upper_item is not None:
            binding = ",".join(
                sorted(
                    name
                    for name, passed in upper_item[0].constraint_results.items()
                    if not passed
                )
            )
        rows.append(
            {
                "model_name": model_name,
                "layer": layer,
                "channel": channel,
                "example_id": example_id,
                "family": family,
                "parameter": parameter_name,
                "side": side,
                "ordered_control": int(ordered_control),
                "boundary_status": status,
                "n_attempted": len(values),
                "n_preserved": len(preserved),
                "preservation_rate": len(preserved) / len(values),
                "lower_control_bound": lower if preserved else float("nan"),
                "upper_control_bound": (
                    upper_item[1] if upper_item is not None else float("nan")
                ),
                "boundary_width": (
                    upper_item[1] - lower
                    if upper_item is not None and preserved
                    else float("nan")
                ),
                "maximum_target_displacement": max(
                    (abs(item[2]) for item in preserved),
                    default=float("nan"),
                ),
                "maximum_state_displacement": max(
                    (item[3] for item in preserved),
                    default=float("nan"),
                ),
                "monotonicity_violations": violations,
                "binding_constraints": binding,
            }
        )
    return rows


def summarize_controllability_boundaries(
    rows: Sequence[Mapping[str, Any]],
    *,
    bootstrap_resamples: int = 2_000,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Aggregate boundary estimates across examples with bootstrap intervals."""

    grouped: dict[tuple[str, int, str, str, str, str], list[Mapping[str, Any]]] = (
        defaultdict(list)
    )
    for row in rows:
        grouped[
            (
                str(row["model_name"]),
                int(row["layer"]),
                str(row["channel"]),
                str(row["family"]),
                str(row["parameter"]),
                str(row["side"]),
            )
        ].append(row)

    summaries = []
    for group_index, (key, group) in enumerate(sorted(grouped.items())):
        model_name, layer, channel, family, parameter, side = key
        lower = _bootstrap_mean_interval(
            [float(row["lower_control_bound"]) for row in group],
            resamples=bootstrap_resamples,
            seed=seed + 2 * group_index,
        )
        displacement = _bootstrap_mean_interval(
            [float(row["maximum_state_displacement"]) for row in group],
            resamples=bootstrap_resamples,
            seed=seed + 2 * group_index + 1,
        )
        ordered = bool(int(group[0]["ordered_control"]))
        summaries.append(
            {
                "model_name": model_name,
                "layer": layer,
                "channel": channel,
                "family": family,
                "parameter": parameter,
                "side": side,
                "ordered_control": int(ordered),
                "n_examples": len(group),
                "feasible_example_rate": float(
                    np.mean([int(row["n_preserved"]) > 0 for row in group])
                ),
                "bracketed_rate": float(
                    np.mean([row["boundary_status"] == "bracketed" for row in group])
                ),
                "right_censored_rate": float(
                    np.mean(
                        [row["boundary_status"] == "right_censored" for row in group]
                    )
                ),
                "mean_lower_control_bound": lower[0],
                "lower_control_bound_ci_low": lower[1],
                "lower_control_bound_ci_high": lower[2],
                "mean_maximum_state_displacement": displacement[0],
                "maximum_state_displacement_ci_low": displacement[1],
                "maximum_state_displacement_ci_high": displacement[2],
                "mean_monotonicity_violations": float(
                    np.mean([int(row["monotonicity_violations"]) for row in group])
                ),
            }
        )
    return summaries


def _stable_seed(*parts: object, seed: int) -> int:
    payload = "\x1f".join(str(part) for part in (*parts, seed)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def _pairwise_distances(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    squared = (
        np.square(first).sum(axis=1, keepdims=True)
        + np.square(second).sum(axis=1)[None, :]
        - 2.0 * (first @ second.T)
    )
    return np.sqrt(np.maximum(squared, 0.0))


def _within_set_resolution(distances: np.ndarray) -> float:
    if distances.shape[0] < 2:
        return float("nan")
    values = distances.copy()
    np.fill_diagonal(values, np.inf)
    values[values <= 1e-10] = np.inf
    nearest = values.min(axis=1)
    finite = nearest[np.isfinite(nearest)]
    return float(np.median(finite)) if finite.size else 0.0


def _directed_gap(
    cross_distances: np.ndarray,
    target_distances: np.ndarray,
    source_norms: np.ndarray,
    target_norms: np.ndarray,
    *,
    radius_multiplier: float,
) -> dict[str, float]:
    nearest = cross_distances.min(axis=1)
    scale_values = np.concatenate([source_norms, target_norms])
    positive_scale = scale_values[scale_values > 1e-12]
    scale = float(np.median(positive_scale)) if positive_scale.size else 1.0
    resolution = _within_set_resolution(target_distances)
    radius = radius_multiplier * resolution if np.isfinite(resolution) else float("nan")
    return {
        "mean_nearest_distance": float(nearest.mean()),
        "median_nearest_distance": float(np.median(nearest)),
        "directed_hausdorff_distance": float(nearest.max()),
        "mean_normalized_gap": float(nearest.mean() / scale),
        "directed_hausdorff_normalized": float(nearest.max() / scale),
        "target_resolution": resolution,
        "target_resolution_normalized": float(resolution / scale),
        "coverage_radius": radius,
        "coverage_rate": (
            float(np.mean(nearest <= radius)) if np.isfinite(radius) else float("nan")
        ),
    }


def directed_accessibility_rows(
    samples: Sequence[StateSample],
    *,
    channels: Sequence[ControlChannel] = (
        ControlChannel.PROMPT,
        ControlChannel.ACTIVATION,
    ),
    radius_multiplier: float = 1.5,
    subsample_repeats: int = 32,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Estimate directed cross-channel state coverage for each example.

    Both sets are repeatedly reduced to the same number of samples before nearest
    set distances are measured. This prevents a denser control sweep from appearing
    more expressive solely because it contains more sampled states.
    """

    if radius_multiplier <= 0:
        raise ValueError("radius_multiplier must be positive")
    if subsample_repeats <= 0:
        raise ValueError("subsample_repeats must be positive")
    baselines = _baseline_index(samples)
    grouped: dict[
        tuple[str, str, int, ControlChannel],
        list[np.ndarray],
    ] = defaultdict(list)
    attempted: set[tuple[str, str, int, ControlChannel]] = set()
    allowed = set(channels)
    for sample in samples:
        if sample.intervention.channel not in allowed:
            continue
        key = (sample.model_name, sample.example_id, sample.layer)
        attempted.add((*key, sample.intervention.channel))
        if not sample.behavior_preserved:
            continue
        baseline = baselines.get(key)
        if baseline is None:
            raise ValueError(
                f"missing baseline for example {sample.example_id!r}, layer {sample.layer}"
            )
        grouped[(*key, sample.intervention.channel)].append(
            np.asarray(sample.state - baseline.state, dtype=np.float64)
        )

    rows = []
    base_keys = sorted({key[:3] for key in attempted})
    for model_name, example_id, layer in base_keys:
        for source_channel in channels:
            for target_channel in channels:
                if source_channel is target_channel:
                    continue
                source_values = grouped.get(
                    (model_name, example_id, layer, source_channel), []
                )
                target_values = grouped.get(
                    (model_name, example_id, layer, target_channel), []
                )
                if not source_values or not target_values:
                    status = (
                        "both_empty"
                        if not source_values and not target_values
                        else "source_empty"
                        if not source_values
                        else "target_empty"
                    )
                    rows.append(
                        {
                            "model_name": model_name,
                            "layer": layer,
                            "example_id": example_id,
                            "source_channel": source_channel.value,
                            "target_channel": target_channel.value,
                            "direction": (
                                f"{source_channel.value}_to_{target_channel.value}"
                            ),
                            "status": status,
                            "n_source_states": len(source_values),
                            "n_target_states": len(target_values),
                            "balanced_sample_size": 0,
                            "subsample_repeats": 0,
                            "radius_multiplier": radius_multiplier,
                            "mean_nearest_distance": float("nan"),
                            "median_nearest_distance": float("nan"),
                            "directed_hausdorff_distance": float("nan"),
                            "mean_normalized_gap": float("nan"),
                            "directed_hausdorff_normalized": float("nan"),
                            "target_resolution": float("nan"),
                            "target_resolution_normalized": float("nan"),
                            "coverage_radius": float("nan"),
                            "coverage_rate": (
                                0.0
                                if source_values and not target_values
                                else float("nan")
                            ),
                        }
                    )
                    continue
                source = np.stack(source_values)
                target = np.stack(target_values)
                cross_distances = _pairwise_distances(source, target)
                target_distances = _pairwise_distances(target, target)
                source_norms = np.linalg.norm(source, axis=1)
                target_norms = np.linalg.norm(target, axis=1)
                count = min(source.shape[0], target.shape[0])
                repeats = (
                    1
                    if source.shape[0] == target.shape[0] == count
                    else subsample_repeats
                )
                rng = np.random.default_rng(
                    _stable_seed(
                        model_name,
                        example_id,
                        layer,
                        source_channel.value,
                        target_channel.value,
                        seed=seed,
                    )
                )
                estimates = []
                for _ in range(repeats):
                    source_index = rng.choice(source.shape[0], count, replace=False)
                    target_index = rng.choice(target.shape[0], count, replace=False)
                    estimates.append(
                        _directed_gap(
                            cross_distances[np.ix_(source_index, target_index)],
                            target_distances[np.ix_(target_index, target_index)],
                            source_norms[source_index],
                            target_norms[target_index],
                            radius_multiplier=radius_multiplier,
                        )
                    )
                metrics = {
                    name: float(np.nanmean([estimate[name] for estimate in estimates]))
                    if np.any(np.isfinite([estimate[name] for estimate in estimates]))
                    else float("nan")
                    for name in estimates[0]
                }
                rows.append(
                    {
                        "model_name": model_name,
                        "layer": layer,
                        "example_id": example_id,
                        "source_channel": source_channel.value,
                        "target_channel": target_channel.value,
                        "direction": f"{source_channel.value}_to_{target_channel.value}",
                        "status": "estimable",
                        "n_source_states": source.shape[0],
                        "n_target_states": target.shape[0],
                        "balanced_sample_size": count,
                        "subsample_repeats": repeats,
                        "radius_multiplier": radius_multiplier,
                        **metrics,
                    }
                )
    return rows


def summarize_directed_accessibility(
    rows: Sequence[Mapping[str, Any]],
    *,
    bootstrap_resamples: int = 2_000,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Aggregate directed accessibility gaps over examples."""

    grouped: dict[tuple[str, int, str, str], list[Mapping[str, Any]]] = defaultdict(
        list
    )
    for row in rows:
        grouped[
            (
                str(row["model_name"]),
                int(row["layer"]),
                str(row["source_channel"]),
                str(row["target_channel"]),
            )
        ].append(row)

    summaries = []
    for group_index, (key, group) in enumerate(sorted(grouped.items())):
        model_name, layer, source_channel, target_channel = key
        estimable = [row for row in group if row.get("status") == "estimable"]
        gap = _bootstrap_mean_interval(
            [float(row["mean_normalized_gap"]) for row in group],
            resamples=bootstrap_resamples,
            seed=seed + 2 * group_index,
        )
        coverage = _bootstrap_mean_interval(
            [float(row["coverage_rate"]) for row in group],
            resamples=bootstrap_resamples,
            seed=seed + 2 * group_index + 1,
        )
        summaries.append(
            {
                "model_name": model_name,
                "layer": layer,
                "source_channel": source_channel,
                "target_channel": target_channel,
                "direction": f"{source_channel}_to_{target_channel}",
                "n_examples": len(group),
                "n_estimable_examples": len(estimable),
                "estimable_example_rate": len(estimable) / len(group),
                "target_empty_rate": float(
                    np.mean([row.get("status") == "target_empty" for row in group])
                ),
                "source_empty_rate": float(
                    np.mean([row.get("status") == "source_empty" for row in group])
                ),
                "mean_normalized_gap": gap[0],
                "mean_normalized_gap_ci_low": gap[1],
                "mean_normalized_gap_ci_high": gap[2],
                "mean_coverage_rate": coverage[0],
                "coverage_rate_ci_low": coverage[1],
                "coverage_rate_ci_high": coverage[2],
                "mean_directed_hausdorff_normalized": _bootstrap_mean_interval(
                    [float(row["directed_hausdorff_normalized"]) for row in group],
                    resamples=0,
                    seed=seed,
                )[0],
                "mean_balanced_sample_size": (
                    float(
                        np.mean([int(row["balanced_sample_size"]) for row in estimable])
                    )
                    if estimable
                    else 0.0
                ),
            }
        )
    return summaries


def _binary_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    positive = scores[labels == 1]
    negative = scores[labels == 0]
    if positive.size == 0 or negative.size == 0:
        return float("nan")
    comparisons = positive[:, None] - negative[None, :]
    return float(np.mean(comparisons > 0) + 0.5 * np.mean(comparisons == 0))


def detection_control_gap_rows(
    samples: Sequence[StateSample],
    *,
    label_metric: str = "monitor_label",
    target_metric: str = "target_projection",
    low_control_fraction: float = 0.25,
) -> list[dict[str, Any]]:
    """Estimate the representation-control gap in common projection units."""

    if low_control_fraction < 0:
        raise ValueError("low_control_fraction must be nonnegative")
    grouped: dict[tuple[str, int], list[StateSample]] = defaultdict(list)
    for sample in samples:
        grouped[(sample.model_name, sample.layer)].append(sample)

    rows = []
    for (model_name, layer), group in sorted(grouped.items()):
        baselines = [
            sample
            for sample in group
            if sample.intervention.channel is ControlChannel.BASELINE
            and label_metric in sample.metrics
            and target_metric in sample.metrics
            and int(sample.metrics[label_metric]) in (0, 1)
        ]
        labels = np.asarray(
            [int(sample.metrics[label_metric]) for sample in baselines],
            dtype=np.int64,
        )
        scores = np.asarray(
            [float(sample.metrics[target_metric]) for sample in baselines],
            dtype=np.float64,
        )
        if len(np.unique(labels)) != 2:
            continue
        signed_gap = float(scores[labels == 1].mean() - scores[labels == 0].mean())
        natural_gap = abs(signed_gap)
        centered_positive = scores[labels == 1] - scores[labels == 1].mean()
        centered_negative = scores[labels == 0] - scores[labels == 0].mean()
        within_class_scale = float(
            np.sqrt(
                np.mean(
                    np.concatenate(
                        [centered_positive**2, centered_negative**2]
                    )
                    ** 2
                )
            )
        )
        pooled_scale = max(within_class_scale, float(scores.std()), 1e-12)
        auc = _binary_auc(scores, labels)
        oriented_auc = max(auc, 1.0 - auc)
        baseline_scores = {
            sample.example_id: float(sample.metrics[target_metric])
            for sample in baselines
        }
        for channel in (
            ControlChannel.PROMPT,
            ControlChannel.ACTIVATION,
            ControlChannel.HYBRID,
        ):
            reached: dict[str, list[float]] = defaultdict(list)
            for sample in group:
                if (
                    sample.intervention.channel is channel
                    and sample.behavior_preserved
                    and target_metric in sample.metrics
                    and sample.example_id in baseline_scores
                ):
                    reached[sample.example_id].append(
                        float(sample.metrics[target_metric])
                    )
            if not any(sample.intervention.channel is channel for sample in group):
                continue
            spans = []
            maximum_movements = []
            for example_id, baseline in baseline_scores.items():
                values = reached.get(example_id, [])
                with_baseline = np.asarray([baseline, *values], dtype=np.float64)
                spans.append(float(with_baseline.max() - with_baseline.min()))
                maximum_movements.append(
                    float(np.max(np.abs(with_baseline - baseline)))
                )
            spans_array = np.asarray(spans, dtype=np.float64)
            movement_array = np.asarray(maximum_movements, dtype=np.float64)
            threshold = low_control_fraction * natural_gap
            detection_margin = natural_gap / pooled_scale
            control_margin = float(movement_array.mean() / pooled_scale)
            rows.append(
                {
                    "model_name": model_name,
                    "layer": layer,
                    "channel": channel.value,
                    "n_labeled_baselines": len(baselines),
                    "n_controlled_examples": len(reached),
                    "controlled_example_rate": len(reached) / len(baselines),
                    "natural_projection_gap": natural_gap,
                    "natural_signed_projection_gap": signed_gap,
                    "natural_projection_auc": auc,
                    "natural_oriented_projection_auc": oriented_auc,
                    "mean_reachable_span": float(spans_array.mean()),
                    "median_reachable_span": float(np.median(spans_array)),
                    "mean_maximum_control_movement": float(movement_array.mean()),
                    "pooled_natural_projection_scale": pooled_scale,
                    "standardized_detection_margin": detection_margin,
                    "standardized_control_margin": control_margin,
                    "representation_control_gap": detection_margin - control_margin,
                    "control_to_detection_ratio": (
                        float(spans_array.mean() / natural_gap)
                        if natural_gap > 0
                        else float("nan")
                    ),
                    "low_control_threshold": threshold,
                    "low_control_fraction": float(np.mean(movement_array < threshold)),
                }
            )
    return rows


def representation_control_gap_rows(
    samples: Sequence[StateSample],
    *,
    label_metric: str = "monitor_label",
    target_metric: str = "target_projection",
    low_control_fraction: float = 0.25,
) -> list[dict[str, Any]]:
    """Named entry point for the representation-control gap estimator."""

    return detection_control_gap_rows(
        samples,
        label_metric=label_metric,
        target_metric=target_metric,
        low_control_fraction=low_control_fraction,
    )
