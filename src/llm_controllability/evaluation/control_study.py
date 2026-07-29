"""Channel-specific setpoint reachability and minimum control cost."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from llm_controllability.controllability.types import ControlChannel, StateSample
from llm_controllability.evaluation.control import dose_response
from llm_controllability.reachability.io import load_state_samples


def setpoints_from_spec(spec: Mapping[str, Any]) -> list[float]:
    values = []
    for intervention in spec.get("interventions", []):
        if intervention.get("type") == "adaptive":
            if "setpoints" in intervention:
                values.extend(float(value) for value in intervention["setpoints"])
            elif "setpoint" in intervention:
                values.append(float(intervention["setpoint"]))
    if not values:
        raise ValueError("study spec does not define adaptive setpoints")
    return sorted(set(values))


def control_cost_rows(
    samples: Sequence[StateSample],
    *,
    target_metric: str,
    setpoints: Sequence[float],
    tolerance: float,
    exclude_prefixes: Sequence[str] = ("orthogonal_random",),
) -> list[dict[str, Any]]:
    if tolerance < 0:
        raise ValueError("tolerance must be nonnegative")
    grouped: dict[tuple[str, int, str, str], list[StateSample]] = defaultdict(list)
    for sample in samples:
        if (
            sample.intervention.channel
            not in (ControlChannel.BASELINE, ControlChannel.RANDOM)
            and target_metric in sample.metrics
            and not sample.intervention.name.startswith(tuple(exclude_prefixes))
        ):
            grouped[
                (
                    sample.model_name,
                    sample.layer,
                    sample.intervention.channel.value,
                    sample.example_id,
                )
            ].append(sample)

    rows = []
    for (model_name, layer, channel, example_id), group in sorted(grouped.items()):
        for setpoint in setpoints:
            eligible = [
                sample
                for sample in group
                if sample.behavior_preserved
                and abs(float(sample.metrics[target_metric]) - setpoint) <= tolerance
            ]
            best = (
                min(eligible, key=lambda sample: sample.intervention.control_cost)
                if eligible
                else None
            )
            rows.append(
                {
                    "model_name": model_name,
                    "layer": layer,
                    "channel": channel,
                    "example_id": example_id,
                    "setpoint": float(setpoint),
                    "tolerance": tolerance,
                    "reached": int(best is not None),
                    "minimum_cost": (
                        best.intervention.control_cost
                        if best is not None
                        else float("nan")
                    ),
                    "control_error": (
                        abs(float(best.metrics[target_metric]) - setpoint)
                        if best is not None
                        else float("nan")
                    ),
                    "intervention": best.intervention.name if best is not None else "",
                }
            )
    return rows


def summarize_control_costs(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, str, float], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                str(row["model_name"]),
                int(row["layer"]),
                str(row["channel"]),
                float(row["setpoint"]),
            )
        ].append(row)

    summaries = []
    for (model_name, layer, channel, setpoint), group in sorted(grouped.items()):
        reached = [row for row in group if int(row["reached"])]
        summaries.append(
            {
                "model_name": model_name,
                "layer": layer,
                "channel": channel,
                "setpoint": setpoint,
                "n_examples": len(group),
                "n_reached": len(reached),
                "reach_rate": len(reached) / len(group),
                "mean_minimum_cost": (
                    float(np.mean([float(row["minimum_cost"]) for row in reached]))
                    if reached
                    else float("nan")
                ),
                "mean_control_error": (
                    float(np.mean([float(row["control_error"]) for row in reached]))
                    if reached
                    else float("nan")
                ),
            }
        )
    return summaries


def dose_response_rows(
    samples: Sequence[StateSample],
    *,
    target_metric: str,
) -> list[dict[str, Any]]:
    """Summarize monotonic target response for each numeric intervention sweep."""

    grouped: dict[
        tuple[str, int, str, str, str],
        list[tuple[float, float]],
    ] = defaultdict(list)
    for sample in samples:
        if not sample.behavior_preserved or target_metric not in sample.metrics:
            continue
        parameter = None
        parameter_name = ""
        for name in ("strength", "fraction", "setpoint"):
            if name in sample.intervention.parameters:
                parameter = float(sample.intervention.parameters[name])
                parameter_name = name
                break
        if parameter is None:
            continue
        family = sample.intervention.name.rsplit("_", 1)[0]
        grouped[
            (
                sample.model_name,
                sample.layer,
                sample.intervention.channel.value,
                sample.example_id,
                family,
            )
        ].append((parameter, float(sample.metrics[target_metric])))

    rows = []
    for (model_name, layer, channel, example_id, family), values in sorted(
        grouped.items()
    ):
        unique = dict(values)
        if len(unique) < 2:
            continue
        ordered = sorted(unique.items())
        summary = dose_response(
            [value for value, _ in ordered],
            [target for _, target in ordered],
        )
        rows.append(
            {
                "model_name": model_name,
                "layer": layer,
                "channel": channel,
                "example_id": example_id,
                "family": family,
                "parameter": parameter_name,
                "n_points": len(ordered),
                **summary,
            }
        )
    return rows


def tracking_stability_rows(
    samples: Sequence[StateSample],
) -> list[dict[str, Any]]:
    rows = []
    for sample in samples:
        if "generation_tracking_mae" not in sample.metrics:
            continue
        rows.append(
            {
                "model_name": sample.model_name,
                "layer": sample.layer,
                "example_id": sample.example_id,
                "intervention": sample.intervention.name,
                "channel": sample.intervention.channel.value,
                "behavior_preserved": int(sample.behavior_preserved),
                "setpoint": sample.intervention.parameters.get("setpoint"),
                "control_cost": sample.intervention.control_cost,
                "tracking_steps": sample.metrics["generation_tracking_steps"],
                "tracking_mae": sample.metrics["generation_tracking_mae"],
                "tracking_max_error": sample.metrics[
                    "generation_tracking_max_error"
                ],
                "tracking_final_error": sample.metrics[
                    "generation_tracking_final_error"
                ],
                "mean_abs_update": sample.metrics[
                    "generation_mean_abs_update"
                ],
            }
        )
    return rows


def _write_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_control_study(
    states_dir: str | Path,
    spec_path: str | Path,
    out_dir: str | Path,
    *,
    target_metric: str = "target_projection",
    tolerance_fraction: float = 0.1,
) -> dict[str, Any]:
    if tolerance_fraction < 0:
        raise ValueError("tolerance_fraction must be nonnegative")
    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    setpoints = setpoints_from_spec(spec)
    span = max(setpoints) - min(setpoints)
    tolerance = max(span * tolerance_fraction, 1e-8)
    samples = load_state_samples(states_dir)
    if any(sample.tags.get("split") == "test" for sample in samples):
        samples = [
            sample for sample in samples if sample.tags.get("split") == "test"
        ]
    rows = control_cost_rows(
        samples,
        target_metric=target_metric,
        setpoints=setpoints,
        tolerance=tolerance,
    )
    summaries = summarize_control_costs(rows)
    responses = dose_response_rows(samples, target_metric=target_metric)
    stability = tracking_stability_rows(samples)
    out_dir = Path(out_dir)
    _write_csv(rows, out_dir / "control_costs.csv")
    _write_csv(summaries, out_dir / "control_summary.csv")
    _write_csv(responses, out_dir / "dose_response.csv")
    _write_csv(stability, out_dir / "tracking_stability.csv")
    manifest = {
        "setpoints": setpoints,
        "tolerance": tolerance,
        "target_metric": target_metric,
        "n_rows": len(rows),
        "n_summary_rows": len(summaries),
        "n_dose_response_rows": len(responses),
        "n_tracking_rows": len(stability),
        "artifacts": [
            "control_costs.csv",
            "control_summary.csv",
            "dose_response.csv",
            "tracking_stability.csv",
        ],
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest
