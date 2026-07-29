"""Worst-case monitor consistency over behavior-preserving states."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence

import numpy as np

from llm_controllability.controllability.types import ControlChannel, StateSample


def monitor_invariance(
    samples: Sequence[StateSample],
    scores: Sequence[float],
    *,
    threshold: float = 0.5,
    labels: Mapping[str, bool] | None = None,
) -> list[dict[str, float | int | str]]:
    """Report per-channel agreement and worst-case score drift from baseline."""

    if len(samples) != len(scores):
        raise ValueError("sample and score counts do not match")
    baseline: dict[tuple[str, str, int], float] = {}
    grouped: dict[tuple[str, int, str], list[tuple[StateSample, float]]] = defaultdict(list)
    for sample, score in zip(samples, scores):
        key = (sample.example_id, sample.model_name, sample.layer)
        if sample.intervention.channel is ControlChannel.BASELINE:
            baseline[key] = float(score)
        elif sample.behavior_preserved:
            grouped[(sample.model_name, sample.layer, sample.intervention.channel.value)].append(
                (sample, float(score))
            )

    rows = []
    for (model_name, layer, channel), values in sorted(grouped.items()):
        agreements = []
        drifts = []
        accuracies = []
        by_example: dict[str, list[float]] = defaultdict(list)
        for sample, score in values:
            key = (sample.example_id, sample.model_name, sample.layer)
            if key not in baseline:
                raise ValueError(f"missing monitor baseline for {key}")
            base_score = baseline[key]
            agreement = (score >= threshold) == (base_score >= threshold)
            agreements.append(float(agreement))
            drift = abs(score - base_score)
            drifts.append(drift)
            by_example[sample.example_id].append(float(agreement))
            if labels is not None and sample.example_id in labels:
                accuracies.append(float((score >= threshold) == labels[sample.example_id]))
        worst_case_consistency = np.mean(
            [min(example_agreement) for example_agreement in by_example.values()]
        )
        rows.append(
            {
                "model_name": model_name,
                "layer": layer,
                "channel": channel,
                "n_states": len(values),
                "agreement": float(np.mean(agreements)),
                "worst_case_consistency": float(worst_case_consistency),
                "mean_score_drift": float(np.mean(drifts)),
                "maximum_score_drift": float(np.max(drifts)),
                "accuracy": float(np.mean(accuracies)) if accuracies else float("nan"),
            }
        )
    return rows
