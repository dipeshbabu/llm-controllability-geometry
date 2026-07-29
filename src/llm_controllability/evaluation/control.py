"""Control error and minimum-energy operating points."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from llm_controllability.controllability.types import StateSample


def minimum_control_cost(
    samples: Sequence[StateSample],
    *,
    target_metric: str,
    setpoints: Sequence[float],
    tolerance: float,
) -> list[dict[str, float | int | str]]:
    """Find the cheapest behavior-preserving state near each desired setpoint."""

    if tolerance < 0:
        raise ValueError("tolerance must be nonnegative")
    rows: list[dict[str, float | int | str]] = []
    for setpoint in setpoints:
        eligible = [
            sample
            for sample in samples
            if sample.behavior_preserved
            and target_metric in sample.metrics
            and abs(sample.metrics[target_metric] - setpoint) <= tolerance
        ]
        if not eligible:
            rows.append(
                {
                    "setpoint": float(setpoint),
                    "reached": 0,
                    "minimum_cost": float("nan"),
                    "control_error": float("nan"),
                    "intervention": "",
                    "channel": "",
                }
            )
            continue
        best = min(eligible, key=lambda sample: sample.intervention.control_cost)
        rows.append(
            {
                "setpoint": float(setpoint),
                "reached": 1,
                "minimum_cost": best.intervention.control_cost,
                "control_error": abs(best.metrics[target_metric] - setpoint),
                "intervention": best.intervention.name,
                "channel": best.intervention.channel.value,
            }
        )
    return rows


def dose_response(
    strengths: Sequence[float],
    target_values: Sequence[float],
) -> dict[str, float]:
    """Summarize monotonicity and linear fit of an intervention sweep."""

    x = np.asarray(strengths, dtype=np.float64)
    y = np.asarray(target_values, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 1 or x.size < 2:
        raise ValueError("dose response requires matching one-dimensional arrays")
    slope, intercept = np.polyfit(x, y, 1)
    prediction = slope * x + intercept
    residual = np.square(y - prediction).sum()
    total = np.square(y - y.mean()).sum()
    r_squared = 1.0 - residual / total if total > 0 else 1.0
    ordered = y[np.argsort(x)]
    differences = np.diff(ordered)
    increasing = float(np.mean(differences >= 0))
    decreasing = float(np.mean(differences <= 0))
    return {
        "slope": float(slope),
        "r_squared": float(r_squared),
        "monotonic_fraction": max(increasing, decreasing),
    }
