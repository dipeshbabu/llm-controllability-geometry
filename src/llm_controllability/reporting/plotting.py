"""Plotting helpers for suppression experiments."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt

from llm_controllability.reporting.results import (
    CandidateRecord,
    pareto_frontier,
    summarize_by_method,
)


def plot_scatter(
    records: Sequence[CandidateRecord],
    path: str | Path,
    *,
    title: str = "",
    minimize: bool = True,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.0, 4.2), dpi=160)
    methods = sorted({r.method for r in records})
    for method in methods:
        group = [r for r in records if r.method == method]
        ax.scatter(
            [r.xentropy for r in group],
            [r.target for r in group],
            s=14,
            alpha=0.55,
            label=method,
        )
    front = pareto_frontier(records, minimize=minimize)
    if front:
        ax.plot(
            [r.xentropy for r in front],
            [r.target for r in front],
            color="black",
            linewidth=1.2,
            label="pareto",
        )
    ax.set_xlabel("self cross entropy")
    ax.set_ylabel("target")
    if title:
        ax.set_title(title)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_method_bars(
    records: Sequence[CandidateRecord],
    path: str | Path,
    *,
    title: str = "",
    minimize: bool = True,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = summarize_by_method(records, minimize=minimize)
    labels = [row["method"] for row in rows]
    targets = [row["best_target"] for row in rows]
    xentropy = [row["best_target_xentropy"] for row in rows]
    x = list(range(len(rows)))

    fig, ax1 = plt.subplots(figsize=(6.0, 4.0), dpi=160)
    width = 0.36
    ax1.bar([v - width / 2 for v in x], targets, width=width, label="target")
    ax1.set_ylabel("target")
    ax2 = ax1.twinx()
    ax2.bar([v + width / 2 for v in x], xentropy, width=width, color="#999999", label="XE")
    ax2.set_ylabel("self cross entropy")
    ax1.set_xticks(x, labels, rotation=20, ha="right")
    if title:
        ax1.set_title(title)
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_robustness_deltas(records: Sequence[CandidateRecord], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    variants = [r.extra.get("variant", r.method) for r in records]
    deltas = [r.target - float(r.extra.get("base_target", r.target)) for r in records]
    fig, ax = plt.subplots(figsize=(7.0, 4.0), dpi=160)
    ax.bar(range(len(records)), deltas)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(range(len(records)), variants, rotation=35, ha="right")
    ax.set_ylabel("target change from original")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
