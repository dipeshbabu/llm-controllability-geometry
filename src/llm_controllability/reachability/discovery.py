"""Predeclared diagnostics for controllability atlases and boundary sharpness."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from llm_controllability.controllability.types import ControlChannel, StateSample
from llm_controllability.reachability.geometry import spectral_metrics

_NUMERIC_PARAMETERS = (
    ("strength", "_a"),
    ("fraction", "_f"),
    ("setpoint", "_s"),
)


def _stable_seed(*parts: object, seed: int) -> int:
    payload = "\x1f".join(str(part) for part in (*parts, seed)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def _baseline_index(
    samples: Sequence[StateSample],
) -> dict[tuple[str, str, int], StateSample]:
    return {
        (sample.model_name, sample.example_id, sample.layer): sample
        for sample in samples
        if sample.intervention.channel is ControlChannel.BASELINE
    }


def controllability_atlas_rows(
    samples: Sequence[StateSample],
    *,
    group_tags: Sequence[str] = ("concept", "category", "source"),
    target_metric: str = "target_projection",
) -> list[dict[str, Any]]:
    """Map accepted control authority by model, depth, channel, and data facet."""

    baselines = _baseline_index(samples)
    maximum_layer = defaultdict(int)
    for sample in samples:
        maximum_layer[sample.model_name] = max(
            maximum_layer[sample.model_name], sample.layer
        )

    grouped: dict[
        tuple[str, int, str, str, str],
        list[StateSample],
    ] = defaultdict(list)
    for sample in samples:
        if sample.intervention.channel is ControlChannel.BASELINE:
            continue
        for tag in group_tags:
            value = sample.tags.get(tag)
            if value:
                grouped[
                    (
                        sample.model_name,
                        sample.layer,
                        sample.intervention.channel.value,
                        tag,
                        value,
                    )
                ].append(sample)

    rows = []
    for (model_name, layer, channel, facet, value), group in sorted(grouped.items()):
        accepted = [sample for sample in group if sample.behavior_preserved]
        attempted_examples = {sample.example_id for sample in group}
        controlled_examples = {sample.example_id for sample in accepted}
        displacements = []
        target_displacements = []
        for sample in accepted:
            key = (model_name, sample.example_id, layer)
            baseline = baselines.get(key)
            if baseline is None:
                raise ValueError(
                    f"missing baseline for example {sample.example_id!r}, layer {layer}"
                )
            displacements.append(
                np.asarray(sample.state - baseline.state, dtype=np.float64)
            )
            if target_metric in sample.metrics and target_metric in baseline.metrics:
                target_displacements.append(
                    abs(
                        float(sample.metrics[target_metric])
                        - float(baseline.metrics[target_metric])
                    )
                )
        matrix = (
            np.stack(displacements)
            if displacements
            else np.empty((0, group[0].state.shape[0]), dtype=np.float64)
        )
        norms = np.linalg.norm(matrix, axis=1) if matrix.size else np.empty(0)
        spectrum = spectral_metrics(matrix, center=False)
        depth_denominator = max(maximum_layer[model_name], 1)
        rows.append(
            {
                "model_name": model_name,
                "layer": layer,
                "relative_depth": layer / depth_denominator,
                "channel": channel,
                "facet": facet,
                "facet_value": value,
                "n_attempted": len(group),
                "n_preserved": len(accepted),
                "n_examples": len(attempted_examples),
                "n_controlled_examples": len(controlled_examples),
                "preservation_rate": len(accepted) / len(group),
                "controlled_example_rate": (
                    len(controlled_examples) / len(attempted_examples)
                ),
                "effective_rank": spectrum["effective_rank"],
                "participation_ratio": spectrum["participation_ratio"],
                "mean_state_displacement": (
                    float(norms.mean()) if norms.size else float("nan")
                ),
                "maximum_state_displacement": (
                    float(norms.max()) if norms.size else float("nan")
                ),
                "mean_target_displacement": (
                    float(np.mean(target_displacements))
                    if target_displacements
                    else float("nan")
                ),
                "maximum_target_displacement": (
                    float(np.max(target_displacements))
                    if target_displacements
                    else float("nan")
                ),
            }
        )
    return rows


def _numeric_control(
    sample: StateSample,
    baseline: StateSample,
    *,
    target_metric: str,
) -> tuple[str, str, float] | None:
    parameters = sample.intervention.parameters
    for parameter, marker in _NUMERIC_PARAMETERS:
        if parameter not in parameters:
            continue
        value = float(parameters[parameter])
        if parameter == "setpoint":
            if target_metric not in baseline.metrics:
                return None
            value -= float(baseline.metrics[target_metric])
        if abs(value) <= 1e-12:
            return None
        family = sample.intervention.name.rsplit(marker, 1)[0]
        return family, parameter, value
    return None


def boundary_survival_rows(
    samples: Sequence[StateSample],
    *,
    target_metric: str = "target_projection",
    dose_grid: Sequence[float] = (0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0),
    bootstrap_resamples: int = 2_000,
    minimum_sharp_drop: float = 0.25,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Estimate preservation curves over each sampled numeric dose schedule.

    Doses are normalized within each example and control side. The resulting
    coordinate measures position in the declared sweep, not a universal physical
    intervention unit.
    """

    grid = np.asarray(dose_grid, dtype=np.float64)
    if grid.ndim != 1 or grid.size < 2:
        raise ValueError("dose_grid must contain at least two points")
    if np.any(~np.isfinite(grid)) or np.any(grid <= 0) or np.any(grid > 1):
        raise ValueError("dose_grid values must be finite and in (0, 1]")
    if np.any(np.diff(grid) <= 0):
        raise ValueError("dose_grid must be strictly increasing")
    if bootstrap_resamples < 0:
        raise ValueError("bootstrap_resamples must be nonnegative")
    if not 0 <= minimum_sharp_drop <= 1:
        raise ValueError("minimum_sharp_drop must be in [0, 1]")

    baselines = _baseline_index(samples)
    trajectories: dict[
        tuple[str, int, str, str, str, str],
        dict[str, list[tuple[float, bool]]],
    ] = defaultdict(lambda: defaultdict(list))
    for sample in samples:
        if sample.intervention.channel is ControlChannel.BASELINE:
            continue
        key = (sample.model_name, sample.example_id, sample.layer)
        baseline = baselines.get(key)
        if baseline is None:
            raise ValueError(
                f"missing baseline for example {sample.example_id!r}, layer {sample.layer}"
            )
        control = _numeric_control(sample, baseline, target_metric=target_metric)
        if control is None:
            continue
        family, parameter, signed_coordinate = control
        side = "increase" if signed_coordinate > 0 else "decrease"
        trajectories[
            (
                sample.model_name,
                sample.layer,
                sample.intervention.channel.value,
                family,
                parameter,
                side,
            )
        ][sample.example_id].append(
            (abs(signed_coordinate), sample.behavior_preserved)
        )

    rows = []
    for group_index, (group_key, by_example) in enumerate(sorted(trajectories.items())):
        model_name, layer, channel, family, parameter, side = group_key
        outcomes = []
        coordinates = []
        nonmonotonic = []
        for example_id in sorted(by_example):
            ordered = sorted(by_example[example_id])
            deduplicated: dict[float, bool] = {}
            for coordinate, preserved in ordered:
                deduplicated[coordinate] = preserved
            ordered = sorted(deduplicated.items())
            maximum = ordered[-1][0]
            if maximum <= 0:
                continue
            normalized = np.asarray(
                [coordinate / maximum for coordinate, _ in ordered],
                dtype=np.float64,
            )
            preserved = np.asarray([state for _, state in ordered], dtype=bool)
            selected = np.asarray(
                [int(np.argmin(np.abs(normalized - dose))) for dose in grid],
                dtype=np.int64,
            )
            outcomes.append(preserved[selected])
            coordinates.append(
                np.asarray([ordered[index][0] for index in selected], dtype=np.float64)
            )
            nonmonotonic.append(
                int(np.count_nonzero((~preserved[:-1]) & preserved[1:]))
            )
        if not outcomes:
            continue
        outcome_matrix = np.stack(outcomes).astype(np.float64)
        coordinate_matrix = np.stack(coordinates)
        previous = np.concatenate(
            [np.ones((outcome_matrix.shape[0], 1)), outcome_matrix[:, :-1]], axis=1
        )
        drops = previous - outcome_matrix
        rates = outcome_matrix.mean(axis=0)
        observed_drops = drops.mean(axis=0)
        rng = np.random.default_rng(
            _stable_seed(*group_key, group_index, seed=seed)
        )
        if outcome_matrix.shape[0] > 1 and bootstrap_resamples > 0:
            indices = rng.integers(
                0,
                outcome_matrix.shape[0],
                size=(bootstrap_resamples, outcome_matrix.shape[0]),
            )
            bootstrap_rates = outcome_matrix[indices].mean(axis=1)
            bootstrap_drops = drops[indices].mean(axis=1)
        else:
            bootstrap_rates = rates[None, :]
            bootstrap_drops = observed_drops[None, :]
        for index, dose in enumerate(grid):
            rows.append(
                {
                    "model_name": model_name,
                    "layer": layer,
                    "channel": channel,
                    "family": family,
                    "parameter": parameter,
                    "side": side,
                    "dose_fraction": float(dose),
                    "n_examples": outcome_matrix.shape[0],
                    "median_control_coordinate": float(
                        np.median(coordinate_matrix[:, index])
                    ),
                    "preservation_rate": float(rates[index]),
                    "preservation_rate_ci_low": float(
                        np.quantile(bootstrap_rates[:, index], 0.025)
                    ),
                    "preservation_rate_ci_high": float(
                        np.quantile(bootstrap_rates[:, index], 0.975)
                    ),
                    "preservation_drop": float(observed_drops[index]),
                    "preservation_drop_ci_low": float(
                        np.quantile(bootstrap_drops[:, index], 0.025)
                    ),
                    "preservation_drop_ci_high": float(
                        np.quantile(bootstrap_drops[:, index], 0.975)
                    ),
                    "probability_sharp_drop": float(
                        np.mean(bootstrap_drops[:, index] >= minimum_sharp_drop)
                    ),
                    "mean_nonmonotonic_transitions": float(np.mean(nonmonotonic)),
                }
            )
    return rows


def phase_transition_candidate_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    minimum_examples: int = 8,
    minimum_sharp_drop: float = 0.25,
    minimum_probability: float = 0.95,
    maximum_mean_nonmonotonic_transitions: float = 0.10,
) -> list[dict[str, Any]]:
    """Qualify abrupt preservation losses without claiming a phase transition."""

    if minimum_examples <= 0:
        raise ValueError("minimum_examples must be positive")
    if not 0 <= minimum_sharp_drop <= 1 or not 0 <= minimum_probability <= 1:
        raise ValueError("drop and probability thresholds must be in [0, 1]")
    if maximum_mean_nonmonotonic_transitions < 0:
        raise ValueError("maximum_mean_nonmonotonic_transitions must be nonnegative")
    grouped: dict[
        tuple[str, int, str, str, str, str],
        list[Mapping[str, Any]],
    ] = defaultdict(list)
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
    for key, group in sorted(grouped.items()):
        ordered = sorted(group, key=lambda row: float(row["dose_fraction"]))
        critical = max(ordered, key=lambda row: float(row["preservation_drop"]))
        n_examples = int(critical["n_examples"])
        drop = float(critical["preservation_drop"])
        probability = float(critical["probability_sharp_drop"])
        final_rate = float(ordered[-1]["preservation_rate"])
        nonmonotonicity = float(
            np.mean(
                [float(row["mean_nonmonotonic_transitions"]) for row in ordered]
            )
        )
        if n_examples < minimum_examples:
            qualification = "insufficient_examples"
        elif nonmonotonicity > maximum_mean_nonmonotonic_transitions:
            qualification = "nonmonotonic_sweep"
        elif drop >= minimum_sharp_drop and probability >= minimum_probability:
            qualification = "sharp_boundary_candidate"
        elif final_rate >= 0.5:
            qualification = "right_censored"
        else:
            qualification = "no_sharp_boundary"
        model_name, layer, channel, family, parameter, side = key
        summaries.append(
            {
                "model_name": model_name,
                "layer": layer,
                "channel": channel,
                "family": family,
                "parameter": parameter,
                "side": side,
                "n_examples": n_examples,
                "critical_dose_fraction": float(critical["dose_fraction"]),
                "critical_control_coordinate": float(
                    critical["median_control_coordinate"]
                ),
                "largest_preservation_drop": drop,
                "preservation_drop_ci_low": float(
                    critical["preservation_drop_ci_low"]
                ),
                "preservation_drop_ci_high": float(
                    critical["preservation_drop_ci_high"]
                ),
                "probability_sharp_drop": probability,
                "final_preservation_rate": final_rate,
                "mean_nonmonotonic_transitions": nonmonotonicity,
                "sharp_boundary_candidate": int(
                    qualification == "sharp_boundary_candidate"
                ),
                "qualification": qualification,
            }
        )
    return summaries
