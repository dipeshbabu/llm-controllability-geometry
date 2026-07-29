"""Result records and summary helpers for suppression experiments."""

from __future__ import annotations

import csv
import dataclasses
import json
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np


@dataclasses.dataclass(frozen=True)
class CandidateRecord:
    target_name: str
    method: str
    seed: int
    text: str
    target: float
    xentropy: float
    source: str = ""
    extra: dict = dataclasses.field(default_factory=dict)

    def to_dict(self) -> dict:
        out = dataclasses.asdict(self)
        out["extra"] = json.dumps(out["extra"], sort_keys=True)
        return out


def records_to_csv(records: Sequence[CandidateRecord], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(CandidateRecord.__dataclass_fields__.keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow(record.to_dict())


def records_from_csv(path: str | Path) -> list[CandidateRecord]:
    records = []
    with Path(path).open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            extra = row.get("extra") or "{}"
            records.append(
                CandidateRecord(
                    target_name=row["target_name"],
                    method=row["method"],
                    seed=int(row["seed"]),
                    text=row["text"],
                    target=float(row["target"]),
                    xentropy=float(row["xentropy"]),
                    source=row.get("source", ""),
                    extra=json.loads(extra),
                )
            )
    return records


def pareto_frontier(
    records: Sequence[CandidateRecord],
    *,
    minimize: bool = True,
) -> list[CandidateRecord]:
    ordered = sorted(records, key=lambda r: (r.xentropy, r.target))
    frontier = []
    best_target = np.inf if minimize else -np.inf
    for record in ordered:
        improves = record.target < best_target if minimize else record.target > best_target
        if improves:
            frontier.append(record)
            best_target = record.target
    return frontier


def best_target_at_fluent(
    records: Sequence[CandidateRecord],
    *,
    quantile: float = 0.25,
    minimize: bool = True,
) -> CandidateRecord | None:
    if not records:
        return None
    threshold = float(np.quantile([r.xentropy for r in records], quantile))
    fluent = [r for r in records if r.xentropy <= threshold]
    if not fluent:
        return None
    key = (lambda r: r.target) if minimize else (lambda r: -r.target)
    return min(fluent, key=key)


def best_fluency_at_threshold(
    records: Sequence[CandidateRecord],
    *,
    threshold: float,
    minimize: bool = True,
) -> CandidateRecord | None:
    if minimize:
        eligible = [r for r in records if r.target <= threshold]
    else:
        eligible = [r for r in records if r.target >= threshold]
    if not eligible:
        return None
    return min(eligible, key=lambda r: r.xentropy)


def summarize_by_method(
    records: Iterable[CandidateRecord],
    *,
    minimize: bool = True,
    fluent_quantile: float = 0.25,
    threshold: float | None = None,
) -> list[dict]:
    grouped: dict[tuple[str, str], list[CandidateRecord]] = {}
    for record in records:
        grouped.setdefault((record.target_name, record.method), []).append(record)

    rows = []
    for (target_name, method), group in sorted(grouped.items()):
        group_minimize = (
            False if target_name.endswith("_increase")
            else True if target_name.endswith("_decrease")
            else minimize
        )
        best_target = (
            min(group, key=lambda r: r.target)
            if group_minimize
            else max(group, key=lambda r: r.target)
        )
        best_fluent = best_target_at_fluent(
            group,
            quantile=fluent_quantile,
            minimize=group_minimize,
        )
        row = {
            "target_name": target_name,
            "method": method,
            "n": len(group),
            "best_target": best_target.target,
            "best_target_xentropy": best_target.xentropy,
            "median_target": float(np.median([r.target for r in group])),
            "median_xentropy": float(np.median([r.xentropy for r in group])),
        }
        if best_fluent is not None:
            row["best_target_at_fluent"] = best_fluent.target
            row["best_target_at_fluent_xentropy"] = best_fluent.xentropy
        if threshold is not None:
            best_near = best_fluency_at_threshold(
                group,
                threshold=threshold,
                minimize=group_minimize,
            )
            if best_near is not None:
                row["best_fluency_at_threshold"] = best_near.xentropy
                row["best_fluency_at_threshold_target"] = best_near.target
        rows.append(row)
    return rows


def rows_to_csv(rows: Sequence[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
