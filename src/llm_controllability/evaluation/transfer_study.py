"""Group transfer analysis from a behavior-preserving state archive."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from collections.abc import Sequence
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

from llm_controllability.controllability.types import ControlChannel, StateSample
from llm_controllability.evaluation.transfer import (
    controlled_overlap_association,
    overlap_transfer_association,
)
from llm_controllability.reachability.geometry import (
    baseline_displacements,
    subspace_overlap,
)
from llm_controllability.reachability.io import load_state_samples


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _intervention_effects(
    samples: Sequence[StateSample],
    *,
    target_metric: str,
) -> dict[str, float]:
    baselines = {
        (sample.example_id, sample.layer): sample.metrics[target_metric]
        for sample in samples
        if sample.intervention.channel is ControlChannel.BASELINE
        and target_metric in sample.metrics
    }
    effects: dict[str, list[float]] = defaultdict(list)
    for sample in samples:
        key = (sample.example_id, sample.layer)
        if (
            sample.behavior_preserved
            and sample.intervention.channel is not ControlChannel.BASELINE
            and target_metric in sample.metrics
            and key in baselines
        ):
            effects[sample.intervention.name].append(
                sample.metrics[target_metric] - baselines[key]
            )
    return {
        intervention: float(np.mean(values))
        for intervention, values in effects.items()
    }


def transfer_rows(
    samples: Sequence[StateSample],
    *,
    group_tag: str = "source",
    target_metric: str = "target_projection",
) -> list[dict[str, float | int | str]]:
    grouped: dict[tuple[str, int, str, str], list[StateSample]] = defaultdict(list)
    baselines = {
        (sample.example_id, sample.model_name, sample.layer): sample
        for sample in samples
        if sample.intervention.channel is ControlChannel.BASELINE
    }
    for sample in samples:
        group = sample.tags.get(group_tag)
        channel = sample.intervention.channel
        if group is None or channel is ControlChannel.BASELINE:
            continue
        grouped[(sample.model_name, sample.layer, channel.value, group)].append(sample)
    for key, group in grouped.items():
        example_ids = {sample.example_id for sample in group}
        group.extend(
            baselines[(example_id, key[0], key[1])]
            for example_id in sorted(example_ids)
            if (example_id, key[0], key[1]) in baselines
        )

    rows = []
    maximum_layer = {
        model_name: max(sample.layer for sample in samples if sample.model_name == model_name)
        for model_name in {sample.model_name for sample in samples}
    }
    prefixes = sorted({key[:3] for key in grouped})
    for model_name, layer, channel_name in prefixes:
        groups = sorted(
            key[3]
            for key in grouped
            if key[:3] == (model_name, layer, channel_name)
        )
        for source, target in combinations(groups, 2):
            first = grouped[(model_name, layer, channel_name, source)]
            second = grouped[(model_name, layer, channel_name, target)]
            channel = ControlChannel(channel_name)
            first_displacements = baseline_displacements(first, channel=channel)
            second_displacements = baseline_displacements(second, channel=channel)
            first_effects = _intervention_effects(first, target_metric=target_metric)
            second_effects = _intervention_effects(second, target_metric=target_metric)
            shared = sorted(set(first_effects) & set(second_effects))
            if len(shared) < 2:
                continue
            source_values = np.asarray([first_effects[name] for name in shared])
            target_values = np.asarray([second_effects[name] for name in shared])
            if np.std(source_values) == 0 or np.std(target_values) == 0:
                transfer_score = float("nan")
            else:
                transfer_score = float(np.corrcoef(source_values, target_values)[0, 1])
            first_norms = np.linalg.norm(first_displacements, axis=1)
            second_norms = np.linalg.norm(second_displacements, axis=1)
            attempted = [
                sample
                for sample in first + second
                if sample.intervention.channel is channel
            ]
            preserved = sum(sample.behavior_preserved for sample in attempted)
            rows.append(
                {
                    "model_name": model_name,
                    "layer": layer,
                    "channel": channel_name,
                    "source_group": source,
                    "target_group": target,
                    "n_shared_interventions": len(shared),
                    "mean_displacement": float(
                        np.concatenate([first_norms, second_norms]).mean()
                    ),
                    "preservation_rate": preserved / len(attempted),
                    "layer_fraction": (
                        layer / maximum_layer[model_name]
                        if maximum_layer[model_name] > 0
                        else 0.0
                    ),
                    "subspace_overlap": subspace_overlap(
                        first_displacements,
                        second_displacements,
                    ),
                    "transfer_score": transfer_score,
                }
            )
    return rows


def run_transfer_study(
    states_dir: str | Path,
    out_path: str | Path,
    *,
    group_tag: str = "source",
    target_metric: str = "target_projection",
) -> dict[str, Any]:
    samples = load_state_samples(states_dir)
    if any(sample.tags.get("split") == "test" for sample in samples):
        samples = [
            sample for sample in samples if sample.tags.get("split") == "test"
        ]
    rows = transfer_rows(
        samples,
        group_tag=group_tag,
        target_metric=target_metric,
    )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(rows, out_path)
    def summarize(group: Sequence[dict[str, float | int | str]]) -> dict[str, Any]:
        result = overlap_transfer_association(
            [float(row["subspace_overlap"]) for row in group],
            [float(row["transfer_score"]) for row in group],
        )
        result.update(
            controlled_overlap_association(
                [float(row["subspace_overlap"]) for row in group],
                [float(row["transfer_score"]) for row in group],
                [
                    [
                        float(row["mean_displacement"]),
                        float(row["preservation_rate"]),
                        float(row["layer_fraction"]),
                        float(row["n_shared_interventions"]),
                    ]
                    for row in group
                ],
            )
        )
        return result

    summary = {
        "pooled": summarize(rows),
        "by_channel": {
            channel: summarize([row for row in rows if row["channel"] == channel])
            for channel in sorted({str(row["channel"]) for row in rows})
        },
    }
    out_path.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary
