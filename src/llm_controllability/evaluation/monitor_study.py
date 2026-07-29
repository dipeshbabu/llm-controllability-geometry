"""Train monitors and evaluate them over held-out reachable states."""

from __future__ import annotations

import csv
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from llm_controllability.controllability.types import ControlChannel, StateSample
from llm_controllability.evaluation.invariance import monitor_invariance
from llm_controllability.evaluation.statistics import (
    adjust_pvalues,
    paired_bootstrap_interval,
    paired_permutation_test,
    paired_standardized_effect,
)
from llm_controllability.monitors import (
    AttentionPooledMonitor,
    LinearMonitor,
    MahalanobisOODMonitor,
    MultiLayerMonitor,
    NonlinearMonitor,
    augment_with_reachable_states,
    pool_hidden_states,
)
from llm_controllability.reachability.io import load_state_samples


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({field for row in rows for field in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def paired_monitor_comparisons(
    score_rows: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    bootstrap_resamples: int = 2_000,
    permutations: int = 10_000,
) -> list[dict[str, Any]]:
    """Pair natural and augmented monitor outcomes by question."""

    baselines: dict[tuple[str, str, str, str, str], tuple[bool, bool]] = {}
    for row in score_rows:
        if row["channel"] == ControlChannel.BASELINE.value:
            key = (
                str(row["model_name"]),
                str(row["layer"]),
                str(row["monitor"]),
                str(row["training"]),
                str(row["example_id"]),
            )
            baselines[key] = (
                float(row["score"]) >= float(row.get("threshold", 0.5)),
                bool(int(row["label"])),
            )

    values: dict[
        tuple[str, str, str, str, str, str],
        list[tuple[str, float]],
    ] = defaultdict(list)
    for key, (prediction, label) in baselines.items():
        model_name, layer, monitor, training, example_id = key
        values[
            (
                model_name,
                layer,
                monitor,
                ControlChannel.BASELINE.value,
                "clean_accuracy",
                training,
            )
        ].append((example_id, float(prediction == label)))

    grouped: dict[
        tuple[str, str, str, str, str, str],
        list[Mapping[str, Any]],
    ] = defaultdict(list)
    for row in score_rows:
        if row["channel"] != ControlChannel.BASELINE.value:
            grouped[
                (
                    str(row["model_name"]),
                    str(row["layer"]),
                    str(row["monitor"]),
                    str(row["channel"]),
                    str(row["training"]),
                    str(row["example_id"]),
                )
            ].append(row)
    for (model_name, layer, monitor, channel, training, example_id), group in grouped.items():
        baseline_key = (model_name, layer, monitor, training, example_id)
        if baseline_key not in baselines:
            continue
        baseline_prediction, _ = baselines[baseline_key]
        predictions = [
            float(row["score"]) >= float(row.get("threshold", 0.5))
            for row in group
        ]
        labels = [bool(int(row["label"])) for row in group]
        values[
            (
                model_name,
                layer,
                monitor,
                channel,
                "state_accuracy",
                training,
            )
        ].append(
            (example_id, float(np.mean([a == b for a, b in zip(predictions, labels)])))
        )
        values[
            (
                model_name,
                layer,
                monitor,
                channel,
                "worst_case_consistency",
                training,
            )
        ].append(
            (
                example_id,
                float(min(prediction == baseline_prediction for prediction in predictions)),
            )
        )

    indexed = {
        key: {example_id: value for example_id, value in group}
        for key, group in values.items()
    }
    comparisons = []
    prefixes = sorted({key[:-1] for key in indexed})
    for prefix in prefixes:
        natural = indexed.get((*prefix, "natural"), {})
        augmented = indexed.get((*prefix, "reachable_augmented"), {})
        shared = sorted(set(natural) & set(augmented))
        if not shared:
            continue
        first = [augmented[example_id] for example_id in shared]
        second = [natural[example_id] for example_id in shared]
        interval = paired_bootstrap_interval(
            first,
            second,
            resamples=bootstrap_resamples,
            seed=seed,
        )
        permutation = paired_permutation_test(
            first,
            second,
            permutations=permutations,
            seed=seed,
        )
        model_name, layer, monitor, channel, metric = prefix
        comparisons.append(
            {
                "model_name": model_name,
                "layer": layer,
                "monitor": monitor,
                "channel": channel,
                "metric": metric,
                "n_examples": len(shared),
                "augmented_minus_natural": interval["estimate"],
                "ci_lower": interval["lower"],
                "ci_upper": interval["upper"],
                "permutation_p": permutation["p_value"],
                "paired_effect_dz": paired_standardized_effect(first, second),
                "bootstrap_resamples": bootstrap_resamples,
                "permutations": permutations,
            }
        )
    if comparisons:
        pvalues = [float(row["permutation_p"]) for row in comparisons]
        holm = adjust_pvalues(pvalues, method="holm")
        fdr = adjust_pvalues(pvalues, method="benjamini-hochberg")
        for row, holm_value, fdr_value in zip(comparisons, holm, fdr):
            row["holm_p"] = float(holm_value)
            row["fdr_bh_q"] = float(fdr_value)
    return comparisons


def _partition_ids(
    samples: Sequence[StateSample],
    label_key: str,
    train_fraction: float,
    validation_fraction: float,
    seed: int,
) -> tuple[set[str], set[str], set[str]]:
    """Use declared splits when present, otherwise create grouped three-way splits."""

    if train_fraction <= 0 or validation_fraction <= 0:
        raise ValueError("train and validation fractions must be positive")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train and validation fractions must sum to less than one")
    tagged: dict[str, str] = {}
    labeled_ids = {
        sample.example_id
        for sample in samples
        if label_key in sample.metrics
    }
    for sample in samples:
        if sample.example_id not in labeled_ids:
            continue
        split = sample.tags.get("split")
        if split in {"train", "validation", "test"}:
            previous = tagged.setdefault(sample.example_id, split)
            if previous != split:
                raise ValueError(
                    f"inconsistent declared split for {sample.example_id}"
                )
    if tagged and set(tagged) == labeled_ids:
        partitions = {
            split: {
                example_id
                for example_id, value in tagged.items()
                if value == split
            }
            for split in ("train", "validation", "test")
        }
        if all(partitions.values()):
            return (
                partitions["train"],
                partitions["validation"],
                partitions["test"],
            )
        raise ValueError("declared monitor splits must all be nonempty")

    train, remainder = _split_ids(
        samples,
        label_key,
        train_fraction,
        seed,
    )
    remainder_samples = [
        sample for sample in samples if sample.example_id in remainder
    ]
    relative_validation = validation_fraction / (1.0 - train_fraction)
    validation, test = _split_ids(
        remainder_samples,
        label_key,
        relative_validation,
        seed + 1,
    )
    return train, validation, test


def _balanced_accuracy(
    predictions: np.ndarray,
    labels: np.ndarray,
) -> float:
    values = []
    for label in (0, 1):
        selected = labels == label
        if np.any(selected):
            values.append(float(np.mean(predictions[selected] == labels[selected])))
    return float(np.mean(values)) if values else float("nan")


def _select_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    """Select a deterministic balanced-accuracy threshold on validation data."""

    values = np.asarray(scores, dtype=np.float64)
    targets = np.asarray(labels, dtype=int)
    if values.shape != targets.shape or values.ndim != 1 or values.size == 0:
        raise ValueError("threshold selection requires matching validation vectors")
    unique = np.unique(values)
    candidates = np.concatenate(
        [
            [0.0],
            0.5 * (unique[:-1] + unique[1:]),
            [1.0],
        ]
    )
    return float(
        max(
            candidates,
            key=lambda threshold: (
                _balanced_accuracy(values >= threshold, targets),
                -abs(float(threshold) - 0.5),
            ),
        )
    )


def _split_ids(
    samples: Sequence[StateSample],
    label_key: str,
    train_fraction: float,
    seed: int,
) -> tuple[set[str], set[str]]:
    labels: dict[str, int] = {}
    group_for_id: dict[str, str] = {}
    for sample in samples:
        if label_key in sample.metrics:
            label = int(sample.metrics[label_key])
            previous = labels.setdefault(sample.example_id, label)
            if previous != label:
                raise ValueError(f"inconsistent label for example {sample.example_id}")
            group = sample.tags.get("pair_id", sample.example_id)
            previous_group = group_for_id.setdefault(sample.example_id, group)
            if previous_group != group:
                raise ValueError(f"inconsistent split group for example {sample.example_id}")
    if not labels:
        raise ValueError(f"no samples contain monitor label {label_key!r}")

    ids_by_group: dict[str, set[str]] = {}
    labels_by_group: dict[str, set[int]] = {}
    for example_id, label in labels.items():
        group = group_for_id[example_id]
        ids_by_group.setdefault(group, set()).add(example_id)
        labels_by_group.setdefault(group, set()).add(label)

    rng = np.random.default_rng(seed)
    train: set[str] = set()
    test: set[str] = set()
    if any(len(values) > 1 for values in labels_by_group.values()):
        groups = np.asarray(sorted(ids_by_group))
        rng.shuffle(groups)
        split = round(len(groups) * train_fraction)
        split = min(max(split, 1), max(len(groups) - 1, 1))
        train_groups = set(groups[:split].tolist())
        for group, example_ids in ids_by_group.items():
            (train if group in train_groups else test).update(example_ids)
    else:
        for label in sorted(set(labels.values())):
            groups = np.asarray(
                sorted(
                    group
                    for group, values in labels_by_group.items()
                    if label in values
                )
            )
            rng.shuffle(groups)
            split = round(len(groups) * train_fraction)
            split = min(max(split, 1), max(len(groups) - 1, 1))
            for group in groups[:split]:
                train.update(ids_by_group[str(group)])
            for group in groups[split:]:
                test.update(ids_by_group[str(group)])
    if not test:
        raise ValueError("monitor split has no held-out examples")
    expected_labels = set(labels.values())
    if {labels[key] for key in train} != expected_labels:
        raise ValueError("monitor training split does not contain every label")
    if {labels[key] for key in test} != expected_labels:
        raise ValueError("monitor test split does not contain every label")
    return train, test


def _features(
    samples: Sequence[StateSample],
    ids: set[str],
    *,
    layer: int,
    label_key: str,
    baseline_only: bool,
) -> tuple[np.ndarray, np.ndarray, list[StateSample]]:
    selected = [
        sample
        for sample in samples
        if sample.example_id in ids
        and sample.layer == layer
        and label_key in sample.metrics
        and (sample.intervention.channel is ControlChannel.BASELINE or sample.behavior_preserved)
        and (not baseline_only or sample.intervention.channel is ControlChannel.BASELINE)
    ]
    if not selected:
        return np.empty((0, 0), dtype=np.float32), np.empty(0, dtype=int), []
    return (
        np.stack([sample.state for sample in selected]),
        np.asarray([int(sample.metrics[label_key]) for sample in selected]),
        selected,
    )


def _pooled_matrix(samples: Sequence[StateSample], pooling: str) -> np.ndarray:
    if pooling == "saved":
        return np.stack([sample.state for sample in samples])
    if any(sample.token_states is None for sample in samples):
        raise ValueError(
            f"{pooling} pooling requires a state archive collected with store_token_states=true"
        )
    return np.stack(
        [
            pool_hidden_states(sample.token_states[None, ...], pooling=pooling)[0]
            for sample in samples
        ]
    )


def _token_batch(samples: Sequence[StateSample]) -> tuple[np.ndarray, np.ndarray]:
    if any(sample.token_states is None for sample in samples):
        raise ValueError(
            "attention pooling requires a state archive collected with store_token_states=true"
        )
    max_tokens = max(sample.token_states.shape[0] for sample in samples)
    width = samples[0].state.shape[0]
    values = np.zeros((len(samples), max_tokens, width), dtype=np.float32)
    mask = np.zeros((len(samples), max_tokens), dtype=bool)
    for index, sample in enumerate(samples):
        assert sample.token_states is not None
        length = sample.token_states.shape[0]
        values[index, :length] = sample.token_states
        mask[index, :length] = True
    return values, mask


def _fit_monitor(kind: str, states: np.ndarray, labels: np.ndarray, seed: int):
    if kind in {"linear", "last_linear", "mean_linear", "max_linear"}:
        return LinearMonitor(seed=seed).fit(states, labels)
    if kind == "random_linear":
        shuffled = np.asarray(labels).copy()
        np.random.default_rng(seed).shuffle(shuffled)
        return LinearMonitor(seed=seed).fit(states, shuffled)
    if kind == "nonlinear":
        return NonlinearMonitor(seed=seed).fit(states, labels)
    raise ValueError(f"unknown monitor kind: {kind}")


def _aligned_multilayer(
    samples: Sequence[StateSample],
    ids: set[str],
    layers: Sequence[int],
    *,
    label_key: str,
    baseline_only: bool,
) -> tuple[dict[int, np.ndarray], np.ndarray, list[StateSample]]:
    grouped: dict[tuple[str, str], dict[int, StateSample]] = {}
    for sample in samples:
        if sample.example_id not in ids or label_key not in sample.metrics:
            continue
        if sample.intervention.channel is not ControlChannel.BASELINE and (
            baseline_only or not sample.behavior_preserved
        ):
            continue
        key = (sample.example_id, sample.intervention.name)
        grouped.setdefault(key, {})[sample.layer] = sample
    complete = [
        group
        for _, group in sorted(grouped.items())
        if all(layer in group for layer in layers)
    ]
    if not complete:
        return {}, np.empty(0, dtype=int), []
    representatives = [group[layers[0]] for group in complete]
    return (
        {
            layer: np.stack([group[layer].state for group in complete])
            for layer in layers
        },
        np.asarray(
            [int(sample.metrics[label_key]) for sample in representatives],
            dtype=int,
        ),
        representatives,
    )


def run_monitor_study(
    states_dir: str | Path,
    out_dir: str | Path,
    *,
    label_key: str = "monitor_label",
    monitor_kinds: Sequence[str] = ("linear", "nonlinear"),
    train_fraction: float = 0.6,
    validation_fraction: float = 0.2,
    reachable_weight: float = 1.0,
    monitor_device: str = "auto",
    seed: int = 0,
) -> dict[str, int]:
    samples = load_state_samples(states_dir)
    train_ids, validation_ids, test_ids = _partition_ids(
        samples,
        label_key,
        train_fraction,
        validation_fraction,
        seed,
    )
    layers = sorted({sample.layer for sample in samples})
    score_rows: list[dict[str, Any]] = []
    invariance_rows: list[dict[str, Any]] = []
    ood_rows: list[dict[str, Any]] = []

    for layer in layers:
        natural_x, natural_y, _ = _features(
            samples,
            train_ids,
            layer=layer,
            label_key=label_key,
            baseline_only=True,
        )
        _, _, reached_samples = _features(
            samples,
            train_ids,
            layer=layer,
            label_key=label_key,
            baseline_only=False,
        )
        validation_x, validation_y, validation_samples = _features(
            samples,
            validation_ids,
            layer=layer,
            label_key=label_key,
            baseline_only=True,
        )
        test_x, test_y, test_samples = _features(
            samples,
            test_ids,
            layer=layer,
            label_key=label_key,
            baseline_only=False,
        )
        if natural_x.shape[0] == 0 or len(np.unique(natural_y)) < 2:
            raise ValueError(f"layer {layer} natural monitor training split lacks both labels")

        augmented_samples = [
            sample
            for sample in reached_samples
            if sample.intervention.channel is not ControlChannel.BASELINE
        ]
        augmented_y = np.asarray(
            [int(sample.metrics[label_key]) for sample in augmented_samples],
            dtype=int,
        )

        for kind in monitor_kinds:
            if kind == "multilayer_linear":
                continue
            pooling = {
                "linear": "saved",
                "random_linear": "saved",
                "nonlinear": "saved",
                "last_linear": "last",
                "mean_linear": "mean",
                "max_linear": "max",
                "attention": "attention",
            }.get(kind)
            if pooling is None:
                raise ValueError(f"unknown monitor kind: {kind}")
            natural_kind_x = (
                natural_x if pooling == "saved" else _pooled_matrix(
                    [
                        sample
                        for sample in samples
                        if sample.example_id in train_ids
                        and sample.layer == layer
                        and sample.intervention.channel is ControlChannel.BASELINE
                    ],
                    pooling,
                )
                if pooling != "attention"
                else None
            )
            augmented_kind_x = (
                _pooled_matrix(augmented_samples, pooling)
                if augmented_samples and pooling not in {"attention"}
                else None
            )
            validation_kind_x = (
                validation_x
                if pooling == "saved"
                else _pooled_matrix(validation_samples, pooling)
                if pooling != "attention"
                else None
            )
            test_kind_x = (
                test_x if pooling == "saved" else _pooled_matrix(test_samples, pooling)
                if pooling != "attention"
                else None
            )
            for training in ("natural", "reachable_augmented"):
                use_augmentation = training == "reachable_augmented" and bool(augmented_samples)
                if kind == "attention":
                    natural_tokens, natural_mask = _token_batch(
                        [
                            sample
                            for sample in samples
                            if sample.example_id in train_ids
                            and sample.layer == layer
                            and sample.intervention.channel is ControlChannel.BASELINE
                        ]
                    )
                    if use_augmentation:
                        augmented_tokens, augmented_mask = _token_batch(augmented_samples)
                        max_tokens = max(natural_tokens.shape[1], augmented_tokens.shape[1])

                        def pad(values, mask, max_tokens=max_tokens):
                            padding = max_tokens - values.shape[1]
                            return (
                                np.pad(values, ((0, 0), (0, padding), (0, 0))),
                                np.pad(mask, ((0, 0), (0, padding))),
                            )

                        natural_tokens, natural_mask = pad(natural_tokens, natural_mask)
                        augmented_tokens, augmented_mask = pad(augmented_tokens, augmented_mask)
                        fit_tokens = np.concatenate([natural_tokens, augmented_tokens])
                        fit_mask = np.concatenate([natural_mask, augmented_mask])
                        fit_y = np.concatenate([natural_y, augmented_y])
                        fit_weights = np.concatenate(
                            [
                                np.ones(len(natural_y), dtype=np.float32),
                                np.full(
                                    len(augmented_y),
                                    reachable_weight,
                                    dtype=np.float32,
                                ),
                            ]
                        )
                    else:
                        fit_tokens, fit_mask, fit_y = natural_tokens, natural_mask, natural_y
                        fit_weights = None
                    monitor = AttentionPooledMonitor(
                        seed=seed,
                        device=monitor_device,
                    ).fit(
                        fit_tokens,
                        fit_y,
                        mask=fit_mask,
                        sample_weight=fit_weights,
                    )
                    validation_tokens, validation_mask = _token_batch(
                        validation_samples
                    )
                    validation_scores = monitor.predict_proba(
                        validation_tokens,
                        mask=validation_mask,
                    )
                    test_tokens, test_mask = _token_batch(test_samples)
                    scores = monitor.predict_proba(test_tokens, mask=test_mask)
                else:
                    assert (
                        natural_kind_x is not None
                        and validation_kind_x is not None
                        and test_kind_x is not None
                    )
                    if use_augmentation:
                        assert augmented_kind_x is not None
                        fit_x, fit_y, weights = augment_with_reachable_states(
                            natural_kind_x,
                            natural_y,
                            augmented_kind_x,
                            augmented_y,
                            reachable_weight=reachable_weight,
                        )
                    else:
                        fit_x, fit_y, weights = natural_kind_x, natural_y, None
                    monitor = _fit_monitor(kind, fit_x, fit_y, seed)
                    if weights is not None and isinstance(monitor, LinearMonitor):
                        weighted_labels = fit_y
                        if kind == "random_linear":
                            weighted_labels = fit_y.copy()
                            np.random.default_rng(seed).shuffle(weighted_labels)
                        monitor.fit(
                            fit_x,
                            weighted_labels,
                            sample_weight=weights,
                        )
                    validation_scores = monitor.predict_proba(
                        validation_kind_x
                    )
                    scores = monitor.predict_proba(test_kind_x)
                threshold = _select_threshold(
                    validation_scores,
                    validation_y,
                )
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
                            "monitor": kind,
                            "training": training,
                            "threshold": threshold,
                        }
                    )
                    invariance_rows.append(row)
                for sample, score, label in zip(test_samples, scores, test_y):
                    score_rows.append(
                        {
                            "model_name": sample.model_name,
                            "layer": layer,
                            "example_id": sample.example_id,
                            "intervention": sample.intervention.name,
                            "channel": sample.intervention.channel.value,
                            "behavior_preserved": sample.behavior_preserved,
                            "monitor": kind,
                            "training": training,
                            "label": int(label),
                            "score": float(score),
                            "threshold": threshold,
                        }
                    )

        ood = MahalanobisOODMonitor().fit(natural_x)
        ood_scores = ood.score(test_x)
        baseline_scores = {
            sample.example_id: score
            for sample, score in zip(test_samples, ood_scores)
            if sample.intervention.channel is ControlChannel.BASELINE
        }
        for sample, score in zip(test_samples, ood_scores):
            if sample.example_id not in baseline_scores:
                continue
            ood_rows.append(
                {
                    "model_name": sample.model_name,
                    "layer": layer,
                    "example_id": sample.example_id,
                    "intervention": sample.intervention.name,
                    "channel": sample.intervention.channel.value,
                    "behavior_preserved": sample.behavior_preserved,
                    "ood_score": float(score),
                    "ood_drift": float(score - baseline_scores[sample.example_id]),
                }
            )

    if "multilayer_linear" in monitor_kinds:
        natural_states, natural_labels, _ = _aligned_multilayer(
            samples,
            train_ids,
            layers,
            label_key=label_key,
            baseline_only=True,
        )
        reached_states, reached_labels, reached_samples = _aligned_multilayer(
            samples,
            train_ids,
            layers,
            label_key=label_key,
            baseline_only=False,
        )
        validation_states, validation_labels, _ = _aligned_multilayer(
            samples,
            validation_ids,
            layers,
            label_key=label_key,
            baseline_only=True,
        )
        test_states, test_labels, test_samples = _aligned_multilayer(
            samples,
            test_ids,
            layers,
            label_key=label_key,
            baseline_only=False,
        )
        reached_indices = [
            index
            for index, sample in enumerate(reached_samples)
            if sample.intervention.channel is not ControlChannel.BASELINE
        ]
        for training in ("natural", "reachable_augmented"):
            monitor = MultiLayerMonitor(tuple(layers), seed=seed)
            if training == "reachable_augmented" and reached_indices:
                fit_states = {
                    layer: np.concatenate(
                        [natural_states[layer], reached_states[layer][reached_indices]]
                    )
                    for layer in layers
                }
                fit_labels = np.concatenate(
                    [natural_labels, reached_labels[reached_indices]]
                )
                weights = np.concatenate(
                    [
                        np.ones(len(natural_labels), dtype=np.float32),
                        np.full(
                            len(reached_indices),
                            reachable_weight,
                            dtype=np.float32,
                        ),
                    ]
                )
                monitor.fit(fit_states, fit_labels, sample_weight=weights)
            else:
                monitor.fit(natural_states, natural_labels)
            validation_scores = monitor.predict_proba(validation_states)
            threshold = _select_threshold(
                validation_scores,
                validation_labels,
            )
            scores = monitor.predict_proba(test_states)
            label_map = {
                sample.example_id: bool(label)
                for sample, label in zip(test_samples, test_labels)
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
                        "layer": "multi:" + ",".join(map(str, layers)),
                        "monitor": "multilayer_linear",
                        "training": training,
                        "threshold": threshold,
                    }
                )
                invariance_rows.append(row)
            for sample, score, label in zip(test_samples, scores, test_labels):
                score_rows.append(
                    {
                        "model_name": sample.model_name,
                        "layer": "multi:" + ",".join(map(str, layers)),
                        "example_id": sample.example_id,
                        "intervention": sample.intervention.name,
                        "channel": sample.intervention.channel.value,
                        "behavior_preserved": sample.behavior_preserved,
                        "monitor": "multilayer_linear",
                        "training": training,
                        "label": int(label),
                        "score": float(score),
                        "threshold": threshold,
                    }
                )

    out_dir = Path(out_dir)
    comparison_rows = paired_monitor_comparisons(score_rows, seed=seed)
    _write_csv(score_rows, out_dir / "monitor_scores.csv")
    _write_csv(invariance_rows, out_dir / "monitor_invariance.csv")
    _write_csv(ood_rows, out_dir / "ood_scores.csv")
    _write_csv(comparison_rows, out_dir / "monitor_comparisons.csv")
    return {
        "n_train_examples": len(train_ids),
        "n_validation_examples": len(validation_ids),
        "n_test_examples": len(test_ids),
        "n_score_rows": len(score_rows),
        "n_invariance_rows": len(invariance_rows),
        "n_ood_rows": len(ood_rows),
        "n_comparison_rows": len(comparison_rows),
    }
