"""Deterministic figure generation from completed controllability artifacts."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _group_mean(
    rows: Sequence[Mapping[str, str]],
    *,
    group_fields: Sequence[str],
    value_field: str,
) -> dict[tuple[str, ...], float]:
    grouped: dict[tuple[str, ...], list[float]] = defaultdict(list)
    for row in rows:
        value = float(row[value_field])
        if np.isfinite(value):
            grouped[tuple(str(row[field]) for field in group_fields)].append(value)
    return {key: float(np.mean(values)) for key, values in grouped.items() if values}


def _save_line_plot(
    values: Mapping[tuple[str, str], float],
    path: Path,
    *,
    xlabel: str,
    ylabel: str,
    title: str,
    x_parser: Callable[[str], float] = float,
) -> None:
    series: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for (group, x_value), value in values.items():
        series[group].append((x_parser(x_value), value))
    figure, axis = plt.subplots(figsize=(6.4, 4.0))
    for group, points in sorted(series.items()):
        ordered = sorted(points)
        axis.plot(
            [point[0] for point in ordered],
            [point[1] for point in ordered],
            marker="o",
            linewidth=1.5,
            markersize=3.5,
            label=group.replace("_", " "),
        )
    axis.set(xlabel=xlabel, ylabel=ylabel, title=title)
    axis.grid(alpha=0.2)
    if series:
        axis.legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=220)
    plt.close(figure)


def _scatter(
    rows: Sequence[Mapping[str, str]],
    path: Path,
    *,
    x_field: str,
    y_field: str,
    group_field: str,
    xlabel: str,
    ylabel: str,
    title: str,
) -> None:
    figure, axis = plt.subplots(figsize=(5.2, 4.2))
    groups = sorted({str(row[group_field]) for row in rows})
    for group in groups:
        selected = [row for row in rows if str(row[group_field]) == group]
        x = np.asarray([float(row[x_field]) for row in selected])
        y = np.asarray([float(row[y_field]) for row in selected])
        finite = np.isfinite(x) & np.isfinite(y)
        axis.scatter(
            x[finite],
            y[finite],
            s=24,
            alpha=0.7,
            label=group.replace("_", " "),
        )
    axis.set(xlabel=xlabel, ylabel=ylabel, title=title)
    axis.grid(alpha=0.2)
    if groups:
        axis.legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=220)
    plt.close(figure)


def _atlas_heatmap(
    rows: Sequence[Mapping[str, str]],
    path: Path,
) -> None:
    facets = {str(row["facet"]) for row in rows}
    facet = next(
        (candidate for candidate in ("concept", "category", "source") if candidate in facets),
        min(facets),
    )
    selected = [row for row in rows if str(row["facet"]) == facet]
    layers = sorted({int(row["layer"]) for row in selected})
    labels = sorted(
        {
            (str(row["channel"]), str(row["facet_value"]))
            for row in selected
        }
    )
    layer_index = {layer: index for index, layer in enumerate(layers)}
    label_index = {label: index for index, label in enumerate(labels)}
    values: dict[tuple[int, int], list[float]] = defaultdict(list)
    for row in selected:
        value = float(row["maximum_state_displacement"])
        if np.isfinite(value):
            values[
                (
                    label_index[(str(row["channel"]), str(row["facet_value"]))],
                    layer_index[int(row["layer"])],
                )
            ].append(value)
    matrix = np.full((len(labels), len(layers)), np.nan)
    for index, cell_values in values.items():
        matrix[index] = float(np.mean(cell_values))
    height = max(3.6, min(9.0, 0.34 * len(labels) + 1.8))
    figure, axis = plt.subplots(figsize=(7.2, height))
    image = axis.imshow(matrix, aspect="auto", interpolation="nearest")
    axis.set(
        xlabel="Layer",
        ylabel=facet.replace("_", " ").title(),
        title="Behavior preserving controllability atlas",
    )
    axis.set_xticks(np.arange(len(layers)), labels=layers)
    axis.set_yticks(
        np.arange(len(labels)),
        labels=[f"{channel}: {value}" for channel, value in labels],
    )
    axis.tick_params(axis="x", labelrotation=45, labelsize=7)
    axis.tick_params(axis="y", labelsize=7)
    figure.colorbar(image, ax=axis, label="Maximum preserved displacement")
    figure.tight_layout()
    figure.savefig(path, dpi=220)
    plt.close(figure)


def render_study_figures(
    run_dir: str | Path,
    out_dir: str | Path,
) -> dict[str, Any]:
    """Render the predeclared core figures without selecting favorable results."""

    run_dir = Path(run_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    artifacts = []

    geometry = [
        row
        for row in _read_csv(run_dir / "reachable" / "geometry.csv")
        if row["channel"] != "prompt_activation_overlap"
    ]
    if geometry:
        values = _group_mean(
            geometry,
            group_fields=("channel", "layer"),
            value_field="effective_rank",
        )
        path = out_dir / "reachable_rank.png"
        _save_line_plot(
            values,
            path,
            xlabel="Layer",
            ylabel="Effective rank",
            title="Behavior preserving reachable rank",
        )
        artifacts.append(path.name)

    growth = _read_csv(run_dir / "reachable" / "budget_growth.csv")
    if growth:
        values = _group_mean(
            growth,
            group_fields=("channel", "budget"),
            value_field="maximum_radius",
        )
        path = out_dir / "budget_growth.png"
        _save_line_plot(
            values,
            path,
            xlabel="Control budget",
            ylabel="Maximum displacement",
            title="Reachable set growth",
        )
        artifacts.append(path.name)

    boundaries = _read_csv(
        run_dir / "reachable" / "controllability_boundary_summary.csv"
    )
    if boundaries:
        values = _group_mean(
            boundaries,
            group_fields=("channel", "layer"),
            value_field="mean_maximum_state_displacement",
        )
        path = out_dir / "controllability_boundary.png"
        _save_line_plot(
            values,
            path,
            xlabel="Layer",
            ylabel="Maximum preserved displacement",
            title="Observed behavior boundary",
        )
        artifacts.append(path.name)

    accessibility = _read_csv(
        run_dir / "reachable" / "directed_accessibility_summary.csv"
    )
    if accessibility:
        values = _group_mean(
            accessibility,
            group_fields=("direction", "layer"),
            value_field="mean_normalized_gap",
        )
        path = out_dir / "directed_accessibility.png"
        _save_line_plot(
            values,
            path,
            xlabel="Layer",
            ylabel="Normalized nearest set gap",
            title="Directed cross channel accessibility",
        )
        artifacts.append(path.name)

    detection_control = _read_csv(run_dir / "reachable" / "detection_control_gap.csv")
    if detection_control:
        path = out_dir / "detection_control_gap.png"
        _scatter(
            detection_control,
            path,
            x_field="natural_oriented_projection_auc",
            y_field="control_to_detection_ratio",
            group_field="channel",
            xlabel="Natural projection AUROC",
            ylabel="Control span / natural gap",
            title="Detection and control are distinct",
        )
        artifacts.append(path.name)

    atlas = _read_csv(run_dir / "reachable" / "controllability_atlas.csv")
    if atlas:
        path = out_dir / "controllability_atlas.png"
        _atlas_heatmap(atlas, path)
        artifacts.append(path.name)

    survival = _read_csv(run_dir / "reachable" / "boundary_survival.csv")
    if survival:
        values = _group_mean(
            survival,
            group_fields=("channel", "dose_fraction"),
            value_field="preservation_rate",
        )
        path = out_dir / "boundary_survival.png"
        _save_line_plot(
            values,
            path,
            xlabel="Normalized sampled dose",
            ylabel="Behavior preservation rate",
            title="Boundary sharpness diagnostic",
        )
        artifacts.append(path.name)

    cmap_queries = _read_csv(run_dir / "reachable" / "cmap_queries.csv")
    if cmap_queries:
        path = out_dir / "cmap_discovery.png"
        _scatter(
            cmap_queries,
            path,
            x_field="query_index",
            y_field="state_displacement",
            group_field="role",
            xlabel="Sequential model query",
            ylabel="Residual state displacement",
            title="Active controllability manifold discovery",
        )
        artifacts.append(path.name)

    jacobians = _read_csv(run_dir / "jacobians.csv")
    if jacobians:
        values = _group_mean(
            jacobians,
            group_fields=("capture_layer", "control_dimension"),
            value_field="rank_fraction",
        )
        path = out_dir / "jacobian_convergence.png"
        _save_line_plot(
            values,
            path,
            xlabel="Control basis dimension",
            ylabel="Jacobian rank fraction",
            title="Local residual control convergence",
        )
        artifacts.append(path.name)

    control = _read_csv(run_dir / "control" / "control_summary.csv")
    if control:
        values = _group_mean(
            control,
            group_fields=("channel", "setpoint"),
            value_field="mean_minimum_cost",
        )
        path = out_dir / "control_energy.png"
        _save_line_plot(
            values,
            path,
            xlabel="Requested setpoint",
            ylabel="Minimum control cost",
            title="Setpoint control energy",
        )
        artifacts.append(path.name)

    transfer = _read_csv(run_dir / "transfer_source.csv")
    if transfer:
        path = out_dir / "overlap_transfer.png"
        _scatter(
            transfer,
            path,
            x_field="subspace_overlap",
            y_field="transfer_score",
            group_field="channel",
            xlabel="Subspace overlap",
            ylabel="Transfer score",
            title="Geometry and held out transfer",
        )
        artifacts.append(path.name)

    monitor = _read_csv(run_dir / "monitors" / "monitor_invariance.csv")
    if monitor:
        values = _group_mean(
            monitor,
            group_fields=("training", "layer"),
            value_field="worst_case_consistency",
        )
        path = out_dir / "monitor_invariance.png"
        _save_line_plot(
            values,
            path,
            xlabel="Layer",
            ylabel="Worst case consistency",
            title="Monitor invariance",
            x_parser=lambda value: float(value.split(":")[-1].split(",")[0]),
        )
        artifacts.append(path.name)

    patching = _read_csv(run_dir / "causal" / "patching.csv")
    if patching:
        values = _group_mean(
            patching,
            group_fields=("direction", "layer"),
            value_field="logit_recovery",
        )
        path = out_dir / "causal_recovery.png"
        _save_line_plot(
            values,
            path,
            xlabel="Patched layer",
            ylabel="Logit recovery",
            title="Cross channel causal recovery",
        )
        artifacts.append(path.name)

    manifest = {
        "run_dir": str(run_dir),
        "artifacts": artifacts,
        "n_artifacts": len(artifacts),
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def render_matrix_figures(
    matrix_dir: str | Path,
    out_dir: str | Path,
) -> dict[str, Any]:
    """Render predeclared cross-model discovery figures."""

    matrix_dir = Path(matrix_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    artifacts = []
    summaries = _read_csv(matrix_dir / "model_summaries.csv")
    revisions = {
        row["slug"]: row for row in _read_csv(matrix_dir / "revisions.csv")
    }
    scaling_rows = [
        row
        for row in summaries
        if row["table"] == "geometry"
        and row["metric"] == "effective_rank"
        and row.get("slug") in revisions
    ]
    if scaling_rows:
        figure, axis = plt.subplots(figsize=(6.4, 4.2))
        grouped: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for row in scaling_rows:
            revision = revisions[row["slug"]]
            group = ": ".join(
                (
                    revision["family"],
                    revision["training_regime"],
                    row.get("channel", ""),
                )
            )
            grouped[group].append(
                (
                    float(revision["parameter_count_billions"]),
                    float(row["mean"]),
                )
            )
        for group, points in sorted(grouped.items()):
            ordered = sorted(points)
            axis.plot(
                [point[0] for point in ordered],
                [point[1] for point in ordered],
                marker="o",
                linewidth=1.2,
                label=group,
            )
        axis.set_xscale("log")
        axis.set(
            xlabel="Nominal parameter count (billions)",
            ylabel="Mean reachable effective rank",
            title="Predeclared controllability scaling diagnostic",
        )
        axis.grid(alpha=0.2)
        axis.legend(frameon=False, fontsize=7)
        figure.tight_layout()
        path = out_dir / "controllability_scaling.png"
        figure.savefig(path, dpi=220)
        plt.close(figure)
        artifacts.append(path.name)

    representation_control = _read_csv(
        matrix_dir / "representation_control_gap.csv"
    )
    if representation_control:
        path = out_dir / "representation_control_gap.png"
        _scatter(
            representation_control,
            path,
            x_field="standardized_detection_margin",
            y_field="standardized_control_margin",
            group_field="slug",
            xlabel="Standardized detection margin",
            ylabel="Standardized control margin",
            title="Cross-model representation control gap",
        )
        artifacts.append(path.name)

    manifest = {
        "matrix_dir": str(matrix_dir),
        "artifacts": artifacts,
        "n_artifacts": len(artifacts),
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest
