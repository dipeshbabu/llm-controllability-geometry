"""Geometry of behavior-preserving state displacements."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from itertools import pairwise

import numpy as np

from llm_controllability.controllability.types import ControlChannel, StateSample


def _matrix(states: np.ndarray | Sequence[np.ndarray]) -> np.ndarray:
    values = np.asarray(states, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("states must be a matrix shaped [samples, width]")
    if not np.isfinite(values).all():
        raise ValueError("states contain a nonfinite value")
    return values


def singular_values(states: np.ndarray, *, center: bool = True) -> np.ndarray:
    values = _matrix(states)
    if values.shape[0] == 0:
        return np.empty(0, dtype=np.float64)
    if center:
        values = values - values.mean(axis=0, keepdims=True)
    return np.linalg.svd(values, full_matrices=False, compute_uv=False)


def _spectral_metrics_from_values(
    values: np.ndarray,
    *,
    relative_tolerance: float,
) -> dict[str, float | int]:
    if values.size == 0 or values[0] == 0:
        return {
            "numerical_rank": 0,
            "effective_rank": 0.0,
            "participation_ratio": 0.0,
        }
    spectrum = values**2
    total = spectrum.sum()
    probabilities = spectrum[spectrum > 0] / total
    squared_spectrum_sum = np.square(spectrum).sum()
    return {
        "numerical_rank": int(
            np.count_nonzero(values > values[0] * relative_tolerance)
        ),
        "effective_rank": float(
            np.exp(-(probabilities * np.log(probabilities)).sum())
        ),
        "participation_ratio": float(total**2 / squared_spectrum_sum),
    }


def spectral_metrics(
    states: np.ndarray,
    *,
    center: bool = True,
    relative_tolerance: float = 1e-6,
) -> dict[str, float | int]:
    """Compute rank diagnostics from one singular-value decomposition."""

    return _spectral_metrics_from_values(
        singular_values(states, center=center),
        relative_tolerance=relative_tolerance,
    )


def effective_rank(states: np.ndarray, *, center: bool = True) -> float:
    """Entropy effective rank based on the covariance spectrum."""

    values = singular_values(states, center=center)
    spectrum = values**2
    total = spectrum.sum()
    if total <= 0:
        return 0.0
    probabilities = spectrum[spectrum > 0] / total
    return float(np.exp(-(probabilities * np.log(probabilities)).sum()))


def participation_ratio(states: np.ndarray, *, center: bool = True) -> float:
    values = singular_values(states, center=center) ** 2
    denominator = np.square(values).sum()
    if denominator <= 0:
        return 0.0
    return float(values.sum() ** 2 / denominator)


def numerical_rank(
    states: np.ndarray,
    *,
    center: bool = True,
    relative_tolerance: float = 1e-6,
) -> int:
    values = singular_values(states, center=center)
    if values.size == 0 or values[0] == 0:
        return 0
    return int(np.count_nonzero(values > values[0] * relative_tolerance))


def principal_basis(
    displacements: np.ndarray,
    *,
    rank: int | None = None,
    variance_fraction: float = 0.95,
) -> np.ndarray:
    """Return an orthonormal row basis for displacement vectors from an origin."""

    values = _matrix(displacements)
    if values.shape[0] == 0:
        return np.empty((0, values.shape[1]), dtype=np.float64)
    _, singular, vh = np.linalg.svd(values, full_matrices=False)
    if rank is None:
        energy = singular**2
        if energy.sum() <= 0:
            return np.empty((0, values.shape[1]), dtype=np.float64)
        cumulative = np.cumsum(energy) / energy.sum()
        rank = int(np.searchsorted(cumulative, variance_fraction) + 1)
    rank = max(0, min(rank, vh.shape[0]))
    return vh[:rank]


def principal_angles(
    first: np.ndarray,
    second: np.ndarray,
    *,
    rank: int | None = None,
    variance_fraction: float = 0.95,
) -> np.ndarray:
    first_basis = principal_basis(first, rank=rank, variance_fraction=variance_fraction)
    second_basis = principal_basis(
        second, rank=rank, variance_fraction=variance_fraction
    )
    if first_basis.shape[0] == 0 or second_basis.shape[0] == 0:
        return np.empty(0, dtype=np.float64)
    cosines = np.linalg.svd(first_basis @ second_basis.T, compute_uv=False)
    return np.arccos(np.clip(cosines, -1.0, 1.0))


def subspace_overlap(
    first: np.ndarray,
    second: np.ndarray,
    *,
    rank: int | None = None,
    variance_fraction: float = 0.95,
) -> float:
    """Mean squared cosine of principal angles between two displacement spaces."""

    angles = principal_angles(
        first,
        second,
        rank=rank,
        variance_fraction=variance_fraction,
    )
    if angles.size == 0:
        return float("nan")
    return float(np.square(np.cos(angles)).mean())


def trajectory_curvature(points: np.ndarray) -> np.ndarray:
    """Turning angle per unit path length for an ordered control trajectory."""

    values = _matrix(points)
    if values.shape[0] < 3:
        return np.empty(0, dtype=np.float64)
    first = np.diff(values, axis=0)
    first_norm = np.linalg.norm(first, axis=1)
    left = first[:-1]
    right = first[1:]
    denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    cosines = np.divide(
        np.einsum("ij,ij->i", left, right),
        denominator,
        out=np.ones_like(denominator),
        where=denominator > 0,
    )
    turning = np.arccos(np.clip(cosines, -1.0, 1.0))
    local_length = 0.5 * (first_norm[:-1] + first_norm[1:])
    return np.divide(
        turning,
        local_length,
        out=np.zeros_like(turning),
        where=local_length > 0,
    )


def connectivity(states: np.ndarray, radius: float) -> dict[str, float | int]:
    """Connected components of the radius graph over reached states."""

    values = _matrix(states)
    if radius < 0:
        raise ValueError("radius must be nonnegative")
    n_samples = values.shape[0]
    if n_samples == 0:
        return {"n_components": 0, "largest_component_fraction": 0.0}
    distances = _pairwise_distances(values)
    return _connectivity_from_distances(distances, radius)


def _pairwise_distances(values: np.ndarray) -> np.ndarray:
    squared_norms = np.einsum("ij,ij->i", values, values)
    squared = squared_norms[:, None] + squared_norms[None, :] - 2.0 * (values @ values.T)
    return np.sqrt(np.maximum(squared, 0.0))


def _connectivity_from_distances(
    distances: np.ndarray,
    radius: float,
) -> dict[str, float | int]:
    n_samples = distances.shape[0]
    if n_samples == 0:
        return {"n_components": 0, "largest_component_fraction": 0.0}
    visited = np.zeros(n_samples, dtype=bool)
    sizes: list[int] = []
    for start in range(n_samples):
        if visited[start]:
            continue
        stack = [start]
        visited[start] = True
        size = 0
        while stack:
            node = stack.pop()
            size += 1
            neighbors = np.flatnonzero((distances[node] <= radius) & ~visited)
            visited[neighbors] = True
            stack.extend(neighbors.tolist())
        sizes.append(size)
    return {
        "n_components": len(sizes),
        "largest_component_fraction": max(sizes) / n_samples,
    }


def baseline_displacements(
    samples: Sequence[StateSample],
    *,
    channel: ControlChannel | None = None,
    preserved_only: bool = True,
) -> np.ndarray:
    """Subtract the matching example baseline before pooling across examples."""

    baselines: dict[tuple[str, str, int], np.ndarray] = {}
    for sample in samples:
        if sample.intervention.channel is ControlChannel.BASELINE:
            baselines[(sample.example_id, sample.model_name, sample.layer)] = (
                sample.state
            )

    displacements: list[np.ndarray] = []
    for sample in samples:
        if sample.intervention.channel is ControlChannel.BASELINE:
            continue
        if channel is not None and sample.intervention.channel is not channel:
            continue
        if preserved_only and not sample.behavior_preserved:
            continue
        key = (sample.example_id, sample.model_name, sample.layer)
        if key not in baselines:
            raise ValueError(
                f"missing baseline for example {sample.example_id!r}, layer {sample.layer}"
            )
        displacements.append(sample.state - baselines[key])

    if not displacements:
        width = samples[0].state.shape[0] if samples else 0
        return np.empty((0, width), dtype=np.float64)
    return np.stack(displacements).astype(np.float64, copy=False)


def _geometry_row(
    model_name: str,
    layer: int,
    channel: str,
    attempted: int,
    preserved: int,
    displacements: np.ndarray,
) -> dict[str, float | int | str]:
    norms = np.linalg.norm(displacements, axis=1) if displacements.size else np.empty(0)
    if displacements.shape[0] > 1:
        distances = _pairwise_distances(displacements)
        np.fill_diagonal(distances, np.inf)
        radius = float(np.median(distances.min(axis=1)))
        np.fill_diagonal(distances, 0.0)
        graph = _connectivity_from_distances(distances, radius)
    else:
        radius = 0.0
        graph = connectivity(displacements, radius)
    spectrum = spectral_metrics(displacements, center=False)
    return {
        "model_name": model_name,
        "layer": layer,
        "channel": channel,
        "n_attempted": attempted,
        "n_preserved": preserved,
        "preservation_rate": preserved / attempted if attempted else float("nan"),
        **spectrum,
        "mean_displacement": float(norms.mean()) if norms.size else float("nan"),
        "max_displacement": float(norms.max()) if norms.size else float("nan"),
        "connectivity_radius": radius,
        "n_components": graph["n_components"],
        "largest_component_fraction": graph["largest_component_fraction"],
    }


def _trajectory_parameter(sample: StateSample) -> tuple[str, float] | None:
    parameters = sample.intervention.parameters
    for key, marker in (
        ("strength", "_a"),
        ("fraction", "_f"),
        ("setpoint", "_s"),
    ):
        if key in parameters:
            family = sample.intervention.name.rsplit(marker, 1)[0]
            return family, float(parameters[key])
    return None


def summarize_trajectories(
    samples: Sequence[StateSample],
) -> list[dict[str, float | int | str]]:
    """Curvature of accepted states along ordered numeric control sweeps."""

    grouped: dict[
        tuple[str, int, str, str, str],
        list[tuple[float, np.ndarray]],
    ] = defaultdict(list)
    for sample in samples:
        if not sample.behavior_preserved:
            continue
        parameter = _trajectory_parameter(sample)
        if parameter is None:
            continue
        family, value = parameter
        grouped[
            (
                sample.model_name,
                sample.layer,
                sample.intervention.channel.value,
                sample.example_id,
                family,
            )
        ].append((value, sample.state))

    rows = []
    for (model_name, layer, channel, example_id, family), values in sorted(
        grouped.items()
    ):
        unique = {}
        for value, state in values:
            unique[value] = state
        ordered = sorted(unique.items())
        points = np.stack([state for _, state in ordered])
        curvatures = trajectory_curvature(points)
        rows.append(
            {
                "model_name": model_name,
                "layer": layer,
                "channel": channel,
                "example_id": example_id,
                "family": family,
                "n_points": len(ordered),
                "mean_curvature": (
                    float(curvatures.mean()) if curvatures.size else float("nan")
                ),
                "max_curvature": (
                    float(curvatures.max()) if curvatures.size else float("nan")
                ),
                "path_length": float(
                    np.linalg.norm(np.diff(points, axis=0), axis=1).sum()
                )
                if len(points) > 1
                else 0.0,
            }
        )
    return rows


def summarize_reachability(
    samples: Sequence[StateSample],
) -> list[dict[str, float | int | str]]:
    """Summarize each model, layer, and channel plus prompt/activation overlap."""

    grouped: dict[tuple[str, int], list[StateSample]] = defaultdict(list)
    for sample in samples:
        grouped[(sample.model_name, sample.layer)].append(sample)

    rows: list[dict[str, float | int | str]] = []
    for (model_name, layer), group in sorted(grouped.items()):
        channel_displacements: dict[ControlChannel, np.ndarray] = {}
        for channel in (
            ControlChannel.PROMPT,
            ControlChannel.ACTIVATION,
            ControlChannel.HYBRID,
            ControlChannel.RANDOM,
        ):
            attempted = sum(sample.intervention.channel is channel for sample in group)
            preserved = sum(
                sample.intervention.channel is channel and sample.behavior_preserved
                for sample in group
            )
            displacements = baseline_displacements(group, channel=channel)
            channel_displacements[channel] = displacements
            if attempted:
                rows.append(
                    _geometry_row(
                        model_name,
                        layer,
                        channel.value,
                        attempted,
                        preserved,
                        displacements,
                    )
                )

        prompt = channel_displacements[ControlChannel.PROMPT]
        activation = channel_displacements[ControlChannel.ACTIVATION]
        if prompt.shape[0] and activation.shape[0]:
            rows.append(
                {
                    "model_name": model_name,
                    "layer": layer,
                    "channel": "prompt_activation_overlap",
                    "n_attempted": prompt.shape[0] + activation.shape[0],
                    "n_preserved": prompt.shape[0] + activation.shape[0],
                    "preservation_rate": 1.0,
                    "numerical_rank": min(
                        numerical_rank(prompt, center=False),
                        numerical_rank(activation, center=False),
                    ),
                    "effective_rank": float("nan"),
                    "participation_ratio": float("nan"),
                    "mean_displacement": subspace_overlap(prompt, activation),
                    "max_displacement": float("nan"),
                    "connectivity_radius": float("nan"),
                    "n_components": 0,
                    "largest_component_fraction": float("nan"),
                }
            )
    return rows


def target_orthogonal_decomposition(
    samples: Sequence[StateSample],
    *,
    target_metric: str = "target_projection",
) -> list[dict[str, float | int | str]]:
    """Separate reached displacement into target-aligned and orthogonal parts."""

    baselines = {
        (sample.example_id, sample.model_name, sample.layer): sample
        for sample in samples
        if sample.intervention.channel is ControlChannel.BASELINE
        and target_metric in sample.metrics
    }
    rows = []
    for sample in samples:
        key = (sample.example_id, sample.model_name, sample.layer)
        if (
            sample.intervention.channel is ControlChannel.BASELINE
            or not sample.behavior_preserved
            or target_metric not in sample.metrics
            or key not in baselines
        ):
            continue
        baseline = baselines[key]
        displacement = sample.state - baseline.state
        norm = float(np.linalg.norm(displacement))
        signed_target = float(
            sample.metrics[target_metric] - baseline.metrics[target_metric]
        )
        target_magnitude = abs(signed_target)
        orthogonal = float(
            np.sqrt(max(norm * norm - target_magnitude * target_magnitude, 0.0))
        )
        rows.append(
            {
                "model_name": sample.model_name,
                "layer": sample.layer,
                "example_id": sample.example_id,
                "intervention": sample.intervention.name,
                "channel": sample.intervention.channel.value,
                "control_cost": sample.intervention.control_cost,
                "signed_target_displacement": signed_target,
                "target_displacement": target_magnitude,
                "orthogonal_displacement": orthogonal,
                "total_displacement": norm,
                "target_fraction": target_magnitude / norm if norm > 0 else 0.0,
            }
        )
    return rows


def budget_growth(
    samples: Sequence[StateSample],
    *,
    maximum_points: int = 16,
) -> list[dict[str, float | int | str]]:
    """Measure reachable-set growth as the accepted control budget increases."""

    grouped: dict[tuple[str, int, ControlChannel], list[StateSample]] = defaultdict(
        list
    )
    for sample in samples:
        if (
            sample.intervention.channel is not ControlChannel.BASELINE
            and sample.behavior_preserved
        ):
            grouped[
                (sample.model_name, sample.layer, sample.intervention.channel)
            ].append(sample)
    rows = []
    for (model_name, layer, channel), group in sorted(
        grouped.items(),
        key=lambda item: (item[0][0], item[0][1], item[0][2].value),
    ):
        costs = np.asarray(
            sorted({sample.intervention.control_cost for sample in group}),
            dtype=np.float64,
        )
        if costs.size > maximum_points:
            indices = np.unique(
                np.rint(np.linspace(0, costs.size - 1, maximum_points)).astype(int)
            )
            costs = costs[indices]
        baselines = {
            (sample.example_id, sample.model_name, sample.layer): sample.state
            for sample in samples
            if sample.intervention.channel is ControlChannel.BASELINE
        }
        for budget in costs:
            accepted = [
                sample for sample in group if sample.intervention.control_cost <= budget
            ]
            displacements = np.stack(
                [
                    sample.state
                    - baselines[(sample.example_id, sample.model_name, sample.layer)]
                    for sample in accepted
                ]
            )
            norms = np.linalg.norm(displacements, axis=1)
            spectrum = spectral_metrics(displacements, center=False)
            rows.append(
                {
                    "model_name": model_name,
                    "layer": layer,
                    "channel": channel.value,
                    "budget": float(budget),
                    "n_states": len(accepted),
                    "n_examples": len({sample.example_id for sample in accepted}),
                    "effective_rank": spectrum["effective_rank"],
                    "participation_ratio": spectrum["participation_ratio"],
                    "maximum_radius": float(norms.max()),
                    "mean_radius": float(norms.mean()),
                }
            )
    return rows


def layerwise_propagation(
    samples: Sequence[StateSample],
) -> list[dict[str, float | int | str]]:
    """Track expansion and contraction of one reached displacement through depth."""

    baselines = {
        (sample.example_id, sample.model_name, sample.layer): sample.state
        for sample in samples
        if sample.intervention.channel is ControlChannel.BASELINE
    }
    grouped: dict[
        tuple[str, str, str, str],
        list[tuple[int, float]],
    ] = defaultdict(list)
    for sample in samples:
        key = (sample.example_id, sample.model_name, sample.layer)
        if (
            sample.intervention.channel is ControlChannel.BASELINE
            or not sample.behavior_preserved
            or key not in baselines
        ):
            continue
        grouped[
            (
                sample.model_name,
                sample.example_id,
                sample.intervention.name,
                sample.intervention.channel.value,
            )
        ].append((sample.layer, float(np.linalg.norm(sample.state - baselines[key]))))

    rows = []
    for (model_name, example_id, intervention, channel), values in sorted(
        grouped.items()
    ):
        ordered = sorted(values)
        for (previous_layer, previous_norm), (layer, norm) in pairwise(ordered):
            rows.append(
                {
                    "model_name": model_name,
                    "example_id": example_id,
                    "intervention": intervention,
                    "channel": channel,
                    "source_layer": previous_layer,
                    "target_layer": layer,
                    "source_norm": previous_norm,
                    "target_norm": norm,
                    "norm_change": norm - previous_norm,
                    "expansion_ratio": (
                        norm / previous_norm if previous_norm > 0 else float("nan")
                    ),
                }
            )
    return rows


def principal_angle_rows(
    samples: Sequence[StateSample],
) -> list[dict[str, float | int | str]]:
    """Return the full prompt versus activation principal-angle spectrum."""

    grouped: dict[tuple[str, int], list[StateSample]] = defaultdict(list)
    for sample in samples:
        grouped[(sample.model_name, sample.layer)].append(sample)
    rows = []
    for (model_name, layer), group in sorted(grouped.items()):
        prompt = baseline_displacements(group, channel=ControlChannel.PROMPT)
        activation = baseline_displacements(group, channel=ControlChannel.ACTIVATION)
        if not prompt.shape[0] or not activation.shape[0]:
            continue
        angles = principal_angles(prompt, activation)
        for index, angle in enumerate(angles):
            rows.append(
                {
                    "model_name": model_name,
                    "layer": layer,
                    "component": index,
                    "angle_radians": float(angle),
                    "angle_degrees": float(np.degrees(angle)),
                    "cosine_squared": float(np.cos(angle) ** 2),
                }
            )
    return rows


def split_half_stability_rows(
    samples: Sequence[StateSample],
    *,
    channels: Sequence[ControlChannel] = (
        ControlChannel.PROMPT,
        ControlChannel.ACTIVATION,
    ),
    repeats: int = 10,
    maximum_states_per_half: int = 32,
    seed: int = 0,
) -> list[dict[str, float | int | str]]:
    """Estimate within-channel subspace stability across example splits."""

    if repeats <= 0:
        raise ValueError("repeats must be positive")
    if maximum_states_per_half <= 0:
        raise ValueError("maximum_states_per_half must be positive")
    baselines = {
        (sample.model_name, sample.example_id, sample.layer): sample.state
        for sample in samples
        if sample.intervention.channel is ControlChannel.BASELINE
    }
    grouped: dict[
        tuple[str, int, ControlChannel],
        dict[str, list[np.ndarray]],
    ] = defaultdict(lambda: defaultdict(list))
    allowed_channels = set(channels)
    for sample in samples:
        key = (sample.model_name, sample.example_id, sample.layer)
        if (
            sample.intervention.channel is ControlChannel.BASELINE
            or sample.intervention.channel not in allowed_channels
            or not sample.behavior_preserved
            or key not in baselines
        ):
            continue
        grouped[(sample.model_name, sample.layer, sample.intervention.channel)][
            sample.example_id
        ].append(sample.state - baselines[key])

    rows = []
    rng = np.random.default_rng(seed)
    for (model_name, layer, channel), by_example in sorted(
        grouped.items(),
        key=lambda item: (item[0][0], item[0][1], item[0][2].value),
    ):
        example_ids = sorted(by_example)
        if len(example_ids) < 4:
            continue
        half_size = len(example_ids) // 2
        for repeat in range(repeats):
            permutation = rng.permutation(example_ids)
            first_ids = permutation[:half_size]
            second_ids = permutation[half_size : 2 * half_size]
            first = np.stack(
                [state for example_id in first_ids for state in by_example[example_id]]
            )
            second = np.stack(
                [state for example_id in second_ids for state in by_example[example_id]]
            )
            sample_size = min(
                first.shape[0],
                second.shape[0],
                maximum_states_per_half,
            )
            if first.shape[0] > sample_size:
                first = first[rng.choice(first.shape[0], sample_size, replace=False)]
            if second.shape[0] > sample_size:
                second = second[rng.choice(second.shape[0], sample_size, replace=False)]
            rows.append(
                {
                    "model_name": model_name,
                    "layer": layer,
                    "channel": channel.value,
                    "repeat": repeat,
                    "n_examples_per_half": half_size,
                    "n_states_per_half": sample_size,
                    "subspace_overlap": subspace_overlap(first, second),
                    "first_effective_rank": effective_rank(first, center=False),
                    "second_effective_rank": effective_rank(second, center=False),
                }
            )
    return rows


def summarize_split_half_stability(
    rows: Sequence[dict[str, float | int | str]],
) -> list[dict[str, float | int | str]]:
    """Summarize repeated split-half estimates without treating repeats as data."""

    grouped: dict[tuple[str, int, str], list[dict[str, float | int | str]]] = (
        defaultdict(list)
    )
    for row in rows:
        grouped[
            (str(row["model_name"]), int(row["layer"]), str(row["channel"]))
        ].append(row)
    summaries = []
    for (model_name, layer, channel), group in sorted(grouped.items()):
        overlaps = np.asarray(
            [float(row["subspace_overlap"]) for row in group], dtype=np.float64
        )
        ranks = np.asarray(
            [
                0.5
                * (
                    float(row["first_effective_rank"])
                    + float(row["second_effective_rank"])
                )
                for row in group
            ],
            dtype=np.float64,
        )
        finite_overlap = overlaps[np.isfinite(overlaps)]
        finite_rank = ranks[np.isfinite(ranks)]
        summaries.append(
            {
                "model_name": model_name,
                "layer": layer,
                "channel": channel,
                "repeats": len(group),
                "n_examples_per_half": int(group[0]["n_examples_per_half"]),
                "mean_subspace_overlap": (
                    float(finite_overlap.mean())
                    if finite_overlap.size
                    else float("nan")
                ),
                "subspace_overlap_q05": (
                    float(np.quantile(finite_overlap, 0.05))
                    if finite_overlap.size
                    else float("nan")
                ),
                "subspace_overlap_q95": (
                    float(np.quantile(finite_overlap, 0.95))
                    if finite_overlap.size
                    else float("nan")
                ),
                "mean_effective_rank": (
                    float(finite_rank.mean()) if finite_rank.size else float("nan")
                ),
            }
        )
    return summaries
