"""Gemma Scope 2 feature analysis over behavior-preserving state archives."""

from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from llm_controllability.controllability.types import ControlChannel, StateSample
from llm_controllability.evaluation.invariance import monitor_invariance
from llm_controllability.evaluation.monitor_study import _select_threshold
from llm_controllability.models.runtime import resolve_device
from llm_controllability.monitors import LinearMonitor, augment_with_reachable_states
from llm_controllability.reachability.geometry import (
    effective_rank,
    numerical_rank,
    participation_ratio,
    subspace_overlap,
)
from llm_controllability.reachability.io import load_state_samples

_GEMMA3_PATTERN = re.compile(
    r"^google/gemma-3-(?P<size>270m|1b|4b|12b|27b)-(?P<variant>pt|it)$",
    re.IGNORECASE,
)
_SITE_RELEASE_SUFFIX = {
    "resid_post_all": "res-all",
}


def gemma_scope_release(
    model_name: str,
    *,
    site: str = "resid_post_all",
) -> str:
    """Return the official SAELens release key for a Gemma 3 checkpoint."""

    match = _GEMMA3_PATTERN.fullmatch(model_name)
    if match is None:
        raise ValueError(
            "Gemma Scope 2 requires an official google/gemma-3-{size}-{pt|it} checkpoint"
        )
    try:
        release_suffix = _SITE_RELEASE_SUFFIX[site]
    except KeyError as error:
        raise ValueError("supported Gemma Scope site: resid_post_all") from error
    return (
        f"gemma-scope-2-{match.group('size').lower()}-"
        f"{match.group('variant').lower()}-{release_suffix}"
    )


def gemma_scope_sae_id(
    layer: int,
    *,
    width: str = "16k",
    l0: str = "small",
) -> str:
    if layer < 0:
        raise ValueError("layer must be nonnegative")
    if width not in {"16k", "262k"}:
        raise ValueError("all-layer Gemma Scope SAEs use width '16k' or '262k'")
    if l0 not in {"small", "big"}:
        raise ValueError("all-layer Gemma Scope SAEs use l0 'small' or 'big'")
    return f"layer_{layer}_width_{width}_l0_{l0}"


def _best_layer(direction_sweep: str | Path) -> int:
    with Path(direction_sweep).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("direction sweep is empty")
    metric = (
        "eval_projection_gap"
        if all(row.get("eval_projection_gap") not in (None, "") for row in rows)
        else "projection_gap"
    )
    best = max(
        rows,
        key=lambda row: (
            float(row[metric])
            if metric == "eval_projection_gap"
            else abs(float(row[metric]))
        ),
    )
    if metric == "eval_projection_gap" and float(best[metric]) <= 0:
        raise ValueError("no fitted direction has a positive held-out projection gap")
    return int(best["layer"])


def _load_sae(release: str, sae_id: str, device: str):
    try:
        from sae_lens import SAE
    except ImportError as error:
        raise RuntimeError(
            "Gemma Scope analysis requires `uv sync --extra scope`"
        ) from error

    loaded = SAE.from_pretrained(
        release=release,
        sae_id=sae_id,
        device=device,
    )
    sae = loaded[0] if isinstance(loaded, tuple) else loaded
    return sae.to(device)


def _sae_device_dtype(sae) -> tuple[torch.device, torch.dtype]:
    try:
        parameter = next(sae.parameters())
        return parameter.device, parameter.dtype
    except StopIteration:
        return torch.device("cpu"), torch.float32


def _write_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _channel_summary(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["channel"]), str(row["intervention"]))].append(row)
    summaries = []
    for (channel, intervention), group in sorted(grouped.items()):
        summaries.append(
            {
                "channel": channel,
                "intervention": intervention,
                "n_samples": len(group),
                "mean_active_features": float(
                    np.mean([float(row["active_features"]) for row in group])
                ),
                "mean_feature_l1": float(
                    np.mean([float(row["feature_l1"]) for row in group])
                ),
                "mean_feature_l2": float(
                    np.mean([float(row["feature_l2"]) for row in group])
                ),
                "mean_relative_reconstruction_error": float(
                    np.mean(
                        [
                            float(row["relative_reconstruction_error"])
                            for row in group
                        ]
                    )
                ),
            }
        )
    return summaries


def analyze_gemma_scope_features(
    samples: Sequence[StateSample],
    sae,
    *,
    layer: int,
    model_name: str,
    top_k: int = 128,
    analysis_features: int = 2048,
    batch_size: int = 32,
    include_unpreserved: bool = False,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
    np.ndarray,
]:
    """Encode states and compare prompt and activation feature displacements."""

    selected = [
        sample
        for sample in samples
        if sample.layer == layer
        and sample.model_name == model_name
        and (include_unpreserved or sample.behavior_preserved)
    ]
    if not selected:
        raise ValueError("no matching state samples were found for Gemma Scope analysis")
    if top_k <= 0 or analysis_features <= 0 or batch_size <= 0:
        raise ValueError("top_k, analysis_features, and batch_size must be positive")

    device, dtype = _sae_device_dtype(sae)
    sample_rows: list[dict[str, Any]] = []
    sparse_features: list[tuple[np.ndarray, np.ndarray]] = []
    importance: np.ndarray | None = None

    for start in range(0, len(selected), batch_size):
        group = selected[start:start + batch_size]
        states = torch.as_tensor(
            np.stack([sample.state for sample in group]),
            device=device,
            dtype=dtype,
        )
        with torch.no_grad():
            features = sae.encode(states)
            if isinstance(features, tuple):
                features = features[0]
            reconstructed = sae.decode(features)
        if features.ndim != 2 or features.shape[0] != len(group):
            raise ValueError("SAE encode must return [samples, features]")
        if reconstructed.shape != states.shape:
            raise ValueError("SAE decode must reconstruct the residual-state shape")

        errors = torch.linalg.vector_norm(reconstructed - states, dim=1)
        denominators = torch.linalg.vector_norm(states, dim=1).clamp_min(1e-12)
        k = min(top_k, features.shape[1])
        values, indices = torch.topk(features.abs(), k=k, dim=1)
        signed_values = features.gather(1, indices)

        if importance is None:
            importance = np.zeros(features.shape[1], dtype=np.float64)
        importance += features.abs().sum(dim=0).detach().float().cpu().numpy()

        active = (features != 0).sum(dim=1)
        feature_l1 = features.abs().sum(dim=1)
        feature_l2 = torch.linalg.vector_norm(features, dim=1)
        for offset, sample in enumerate(group):
            index_values = indices[offset].detach().cpu().numpy().astype(np.int64)
            signed = signed_values[offset].detach().float().cpu().numpy()
            sparse_features.append((index_values, signed))
            sample_rows.append(
                {
                    "example_id": sample.example_id,
                    "model_name": sample.model_name,
                    "layer": layer,
                    "intervention": sample.intervention.name,
                    "channel": sample.intervention.channel.value,
                    "behavior_preserved": sample.behavior_preserved,
                    "monitor_label": sample.metrics.get("monitor_label"),
                    "split": sample.tags.get("split"),
                    "pair_id": sample.tags.get("pair_id"),
                    "active_features": int(active[offset].detach().cpu()),
                    "feature_l1": float(feature_l1[offset].detach().cpu()),
                    "feature_l2": float(feature_l2[offset].detach().cpu()),
                    "relative_reconstruction_error": float(
                        (errors[offset] / denominators[offset]).detach().cpu()
                    ),
                    "top_feature_indices": index_values.tolist(),
                    "top_feature_values": signed.tolist(),
                }
            )

    assert importance is not None
    retained = np.argsort(importance)[-min(analysis_features, importance.size):]
    retained = np.sort(retained)
    feature_column = {int(feature): index for index, feature in enumerate(retained)}
    matrix = np.zeros((len(selected), len(retained)), dtype=np.float32)
    for row, (indices, values) in enumerate(sparse_features):
        for feature, value in zip(indices, values):
            column = feature_column.get(int(feature))
            if column is not None:
                matrix[row, column] = float(value)

    baselines = {
        sample.example_id: matrix[index]
        for index, sample in enumerate(selected)
        if sample.intervention.channel is ControlChannel.BASELINE
    }
    channel_displacements: dict[str, list[np.ndarray]] = defaultdict(list)
    for index, sample in enumerate(selected):
        baseline = baselines.get(sample.example_id)
        if baseline is None or sample.intervention.channel is ControlChannel.BASELINE:
            continue
        channel_displacements[sample.intervention.channel.value].append(
            matrix[index] - baseline
        )

    geometry: dict[str, Any] = {
        "n_samples": len(selected),
        "n_sae_features": int(importance.size),
        "n_analysis_features": len(retained),
        "truncation_top_k_per_sample": min(top_k, int(importance.size)),
        "channels": {},
    }
    for channel, displacements in sorted(channel_displacements.items()):
        values = np.asarray(displacements, dtype=np.float64)
        geometry["channels"][channel] = {
            "n_displacements": int(values.shape[0]),
            "effective_rank": effective_rank(values, center=False),
            "participation_ratio": participation_ratio(values, center=False),
            "numerical_rank": numerical_rank(values, center=False),
            "mean_displacement": float(np.linalg.norm(values, axis=1).mean()),
        }
    prompt = channel_displacements.get(ControlChannel.PROMPT.value, [])
    activation = channel_displacements.get(ControlChannel.ACTIVATION.value, [])
    overlap = (
        subspace_overlap(
            np.asarray(prompt, dtype=np.float64),
            np.asarray(activation, dtype=np.float64),
        )
        if prompt and activation
        else float("nan")
    )
    geometry["prompt_activation_overlap"] = (
        float(overlap) if np.isfinite(overlap) else None
    )
    geometry["retained_feature_indices"] = retained.tolist()
    return sample_rows, _channel_summary(sample_rows), geometry, matrix


def _limit_samples_by_split(
    samples: Sequence[StateSample],
    maximum: int | None,
) -> list[StateSample]:
    if maximum is None or len(samples) <= maximum:
        return list(samples)
    grouped: dict[str, list[StateSample]] = defaultdict(list)
    for sample in samples:
        grouped[sample.tags.get("split", "all")].append(sample)
    selected = []
    splits = sorted(grouped)
    base = maximum // len(splits)
    remainder = maximum % len(splits)
    for index, split in enumerate(splits):
        selected.extend(grouped[split][: base + (index < remainder)])
    return selected


def _sae_monitor_study(
    samples: Sequence[StateSample],
    matrix: np.ndarray,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    split_indices: dict[str, list[int]] = defaultdict(list)
    for index, sample in enumerate(samples):
        split_indices[sample.tags.get("split", "")].append(index)
    if not all(split_indices[name] for name in ("train", "validation", "test")):
        return [], []

    train_baseline = [
        index
        for index in split_indices["train"]
        if samples[index].intervention.channel is ControlChannel.BASELINE
        and "monitor_label" in samples[index].metrics
    ]
    train_reached = [
        index
        for index in split_indices["train"]
        if samples[index].intervention.channel is not ControlChannel.BASELINE
        and samples[index].behavior_preserved
        and "monitor_label" in samples[index].metrics
    ]
    validation = [
        index
        for index in split_indices["validation"]
        if samples[index].intervention.channel is ControlChannel.BASELINE
        and "monitor_label" in samples[index].metrics
    ]
    test = [
        index
        for index in split_indices["test"]
        if (
            samples[index].intervention.channel is ControlChannel.BASELINE
            or samples[index].behavior_preserved
        )
        and "monitor_label" in samples[index].metrics
    ]
    if not all((train_baseline, validation, test)):
        return [], []

    natural_y = np.asarray(
        [int(samples[index].metrics["monitor_label"]) for index in train_baseline]
    )
    reached_y = np.asarray(
        [int(samples[index].metrics["monitor_label"]) for index in train_reached]
    )
    validation_y = np.asarray(
        [int(samples[index].metrics["monitor_label"]) for index in validation]
    )
    test_y = np.asarray(
        [int(samples[index].metrics["monitor_label"]) for index in test]
    )
    if any(
        len(np.unique(labels)) < 2
        for labels in (natural_y, validation_y, test_y)
    ):
        return [], []

    score_rows = []
    invariance_rows = []
    test_samples = [samples[index] for index in test]
    for training in ("natural", "reachable_augmented"):
        if training == "reachable_augmented" and train_reached:
            fit_x, fit_y, weights = augment_with_reachable_states(
                matrix[train_baseline],
                natural_y,
                matrix[train_reached],
                reached_y,
            )
            monitor = LinearMonitor(seed=0).fit(
                fit_x,
                fit_y,
                sample_weight=weights,
            )
        else:
            monitor = LinearMonitor(seed=0).fit(
                matrix[train_baseline],
                natural_y,
            )
        validation_scores = monitor.predict_proba(matrix[validation])
        threshold = _select_threshold(validation_scores, validation_y)
        scores = monitor.predict_proba(matrix[test])
        label_map = {
            sample.example_id: bool(label)
            for sample, label in zip(test_samples, test_y)
        }
        summaries = monitor_invariance(
            test_samples,
            scores,
            threshold=threshold,
            labels=label_map,
        )
        for row in summaries:
            row.update(
                {
                    "monitor": "sae_feature_linear",
                    "training": training,
                    "threshold": threshold,
                }
            )
            invariance_rows.append(row)
        for sample, score, label in zip(test_samples, scores, test_y):
            score_rows.append(
                {
                    "model_name": sample.model_name,
                    "layer": sample.layer,
                    "example_id": sample.example_id,
                    "intervention": sample.intervention.name,
                    "channel": sample.intervention.channel.value,
                    "monitor": "sae_feature_linear",
                    "training": training,
                    "label": int(label),
                    "score": float(score),
                    "threshold": threshold,
                }
            )
    return score_rows, invariance_rows


def run_gemma_scope_study(
    states_dir: str | Path,
    out_dir: str | Path,
    *,
    model_name: str,
    layer: int | None = None,
    direction_sweep: str | Path | None = None,
    release: str | None = None,
    sae_id: str | None = None,
    site: str = "resid_post_all",
    width: str = "16k",
    l0: str = "small",
    device: str = "auto",
    top_k: int = 128,
    analysis_features: int = 2048,
    batch_size: int = 32,
    max_samples: int | None = 4096,
    include_unpreserved: bool = False,
    sae=None,
) -> dict[str, Any]:
    if layer is None:
        if direction_sweep is None:
            raise ValueError("provide layer or direction_sweep")
        layer = _best_layer(direction_sweep)
    release = release or gemma_scope_release(model_name, site=site)
    sae_id = sae_id or gemma_scope_sae_id(layer, width=width, l0=l0)
    resolved_device = str(resolve_device(device))
    if sae is None:
        sae = _load_sae(release, sae_id, resolved_device)

    samples = [
        sample
        for sample in load_state_samples(states_dir)
        if sample.layer == layer
        and sample.model_name == model_name
        and (include_unpreserved or sample.behavior_preserved)
    ]
    samples = _limit_samples_by_split(samples, max_samples)
    sample_rows, summaries, geometry, feature_matrix = analyze_gemma_scope_features(
        samples,
        sae,
        layer=layer,
        model_name=model_name,
        top_k=top_k,
        analysis_features=analysis_features,
        batch_size=batch_size,
        include_unpreserved=True,
    )
    monitor_scores, monitor_invariance_rows = _sae_monitor_study(
        samples,
        feature_matrix,
    )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "feature_samples.jsonl").open("w", encoding="utf-8") as handle:
        for row in sample_rows:
            handle.write(json.dumps(row) + "\n")
    _write_csv(summaries, out_dir / "feature_summary.csv")
    _write_csv(monitor_scores, out_dir / "sae_monitor_scores.csv")
    _write_csv(
        monitor_invariance_rows,
        out_dir / "sae_monitor_invariance.csv",
    )
    (out_dir / "feature_geometry.json").write_text(
        json.dumps(geometry, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "model_name": model_name,
        "layer": layer,
        "release": release,
        "sae_id": sae_id,
        "requested_device": device,
        "device": resolved_device,
        "n_samples": len(sample_rows),
        "n_monitor_score_rows": len(monitor_scores),
        "include_unpreserved": include_unpreserved,
        "artifacts": [
            "feature_samples.jsonl",
            "feature_summary.csv",
            "feature_geometry.json",
            "sae_monitor_scores.csv",
            "sae_monitor_invariance.csv",
        ],
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest
