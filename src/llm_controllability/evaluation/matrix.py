"""Strict aggregation for the declared multi-model experiment matrix."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

_TABLES = {
    "geometry": "reachable/geometry.csv",
    "target_geometry": "reachable/target_geometry.csv",
    "budget_growth": "reachable/budget_growth.csv",
    "control": "control/control_summary.csv",
    "dose_response": "control/dose_response.csv",
    "transfer_source": "transfer_source.csv",
    "transfer_category": "transfer_category.csv",
    "patching": "causal/patching.csv",
    "component_ablation": "causal/component_ablation.csv",
    "head_ablation": "causal/head_ablation.csv",
    "path_mediation": "causal/path_mediation.csv",
    "cross_prompt_patching": "causal/cross_prompt_patching.csv",
    "monitor_invariance": "monitors/monitor_invariance.csv",
    "monitor_comparisons": "monitors/monitor_comparisons.csv",
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(f"required nonempty artifact is missing: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"required artifact has no rows: {path}")
    return rows


def _write_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({field for row in rows for field in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _metric_summaries(
    tables: Mapping[str, Sequence[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    declarations = {
        "geometry": (
            ("channel",),
            (
                "preservation_rate",
                "effective_rank",
                "participation_ratio",
                "mean_displacement",
            ),
        ),
        "control": (
            ("channel",),
            ("reach_rate", "mean_minimum_cost", "mean_control_error"),
        ),
        "monitor_invariance": (
            ("monitor", "training", "channel"),
            ("worst_case_consistency", "accuracy", "maximum_score_drift"),
        ),
        "patching": (
            ("direction",),
            ("logit_recovery", "patched_source_js"),
        ),
    }
    summaries = []
    for table_name, (group_fields, metrics) in declarations.items():
        grouped: dict[tuple[str, ...], list[Mapping[str, Any]]] = defaultdict(list)
        for row in tables[table_name]:
            grouped[
                (
                    str(row["slug"]),
                    *(str(row.get(field, "")) for field in group_fields),
                )
            ].append(row)
        for key, group in sorted(grouped.items()):
            slug, *group_values = key
            for metric in metrics:
                values = np.asarray(
                    [float(row[metric]) for row in group],
                    dtype=np.float64,
                )
                finite = values[np.isfinite(values)]
                if not finite.size:
                    continue
                summary = {
                    "slug": slug,
                    "table": table_name,
                    "metric": metric,
                    "n_rows": int(finite.size),
                    "mean": float(finite.mean()),
                    "standard_deviation": (
                        float(finite.std(ddof=1))
                        if finite.size > 1
                        else 0.0
                    ),
                }
                summary.update(dict(zip(group_fields, group_values)))
                summaries.append(summary)
    return summaries


def _matched_contrasts(
    summaries: Sequence[Mapping[str, Any]],
    comparisons: Sequence[Mapping[str, str]],
) -> list[dict[str, Any]]:
    index = {
        tuple(
            (field, str(row.get(field, "")))
            for field in (
                "slug",
                "table",
                "metric",
                "channel",
                "monitor",
                "training",
                "direction",
            )
        ): row
        for row in summaries
    }
    rows = []
    for comparison in comparisons:
        reference_slug = comparison["reference"]
        comparison_slug = comparison["comparison"]
        reference_rows = [
            row for row in summaries if row["slug"] == reference_slug
        ]
        for reference in reference_rows:
            fields = {
                "table": str(reference["table"]),
                "metric": str(reference["metric"]),
                "channel": str(reference.get("channel", "")),
                "monitor": str(reference.get("monitor", "")),
                "training": str(reference.get("training", "")),
                "direction": str(reference.get("direction", "")),
            }
            key = tuple(
                (field, comparison_slug if field == "slug" else fields[field])
                for field in (
                    "slug",
                    "table",
                    "metric",
                    "channel",
                    "monitor",
                    "training",
                    "direction",
                )
            )
            candidate = index.get(key)
            if candidate is None:
                continue
            reference_mean = float(reference["mean"])
            comparison_mean = float(candidate["mean"])
            rows.append(
                {
                    "comparison": comparison["name"],
                    "reference_slug": reference_slug,
                    "comparison_slug": comparison_slug,
                    **fields,
                    "reference_mean": reference_mean,
                    "comparison_mean": comparison_mean,
                    "difference": comparison_mean - reference_mean,
                    "relative_difference": (
                        (comparison_mean - reference_mean) / abs(reference_mean)
                        if reference_mean != 0
                        else float("nan")
                    ),
                }
            )
    return rows


def aggregate_matrix(
    run_root: str | Path,
    matrix_path: str | Path,
    out_dir: str | Path,
) -> dict[str, Any]:
    run_root = Path(run_root)
    matrix = json.loads(Path(matrix_path).read_text(encoding="utf-8"))
    models = matrix["models"]
    slugs = [str(model["slug"]) for model in models]
    if len(slugs) != len(set(slugs)):
        raise ValueError("matrix slugs must be unique")

    combined: dict[str, list[dict[str, Any]]] = {
        name: [] for name in _TABLES
    }
    revisions = []
    for model in models:
        slug = str(model["slug"])
        run_dir = run_root / slug
        manifest_path = run_dir / "reachable" / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"missing run manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        revisions.append(
            {
                "slug": slug,
                "model_name": model["model_name"],
                "protocol": model["protocol"],
                "family": model["family"],
                "scale_group": model["scale_group"],
                "model_revision": manifest.get("runtime", {}).get(
                    "model_revision"
                ),
                "tokenizer_revision": manifest.get("runtime", {}).get(
                    "tokenizer_revision"
                ),
                "repository_commit": manifest.get("runtime", {}).get(
                    "repository_commit"
                ),
                "uv_lock_sha256": manifest.get("runtime", {}).get(
                    "uv_lock_sha256"
                ),
            }
        )
        for name, relative in _TABLES.items():
            for row in _read_csv(run_dir / relative):
                combined[name].append(
                    {
                        "slug": slug,
                        "protocol": model["protocol"],
                        "family": model["family"],
                        "scale_group": model["scale_group"],
                        **row,
                    }
                )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in combined.items():
        _write_csv(rows, out_dir / f"{name}.csv")
    summaries = _metric_summaries(combined)
    contrasts = _matched_contrasts(
        summaries,
        matrix.get("matched_comparisons", []),
    )
    _write_csv(summaries, out_dir / "model_summaries.csv")
    _write_csv(contrasts, out_dir / "matched_contrasts.csv")
    _write_csv(revisions, out_dir / "revisions.csv")
    manifest = {
        "matrix": str(matrix_path),
        "run_root": str(run_root),
        "n_models": len(models),
        "models": slugs,
        "table_rows": {
            name: len(rows) for name, rows in combined.items()
        },
        "n_summary_rows": len(summaries),
        "n_matched_contrasts": len(contrasts),
        "complete": True,
        "artifacts": [
            *(f"{name}.csv" for name in combined),
            "model_summaries.csv",
            "matched_contrasts.csv",
            "revisions.csv",
        ],
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest
