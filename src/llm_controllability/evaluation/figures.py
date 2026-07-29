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
    return {
        key: float(np.mean(values))
        for key, values in grouped.items()
        if values
    }


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
            x_parser=lambda value: float(
                value.split(":")[-1].split(",")[0]
            ),
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
