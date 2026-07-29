"""Robustness checks for suppression prompts."""

from __future__ import annotations

from collections.abc import Callable, Iterable

import numpy as np

from llm_controllability.optimization.benchmarks import score_texts
from llm_controllability.reporting.results import CandidateRecord


def deterministic_variants(text: str) -> list[tuple[str, str]]:
    stripped = " ".join(text.split())
    variants = [
        ("original", text),
        ("trimmed", stripped),
        ("lower", stripped.lower()),
        ("upper", stripped.upper()),
        ("period", stripped.rstrip(".!?") + "."),
        ("instruction_wrap", "Please answer the following: " + stripped),
        ("context_wrap", "Context: " + stripped + "\nAnswer:"),
    ]
    seen = set()
    out = []
    for name, variant in variants:
        if variant not in seen:
            out.append((name, variant))
            seen.add(variant)
    return out


def evaluate_robustness(
    cache_run: Callable,
    model,
    tokenizer,
    records: Iterable[CandidateRecord],
    *,
    batch_size: int = 32,
    max_length: int = 128,
) -> list[CandidateRecord]:
    texts = []
    metadata = []
    for record in records:
        for variant_name, variant_text in deterministic_variants(record.text):
            texts.append(variant_text)
            metadata.append((record, variant_name))

    scored = score_texts(
        cache_run,
        model,
        tokenizer,
        texts,
        target_name="robustness",
        method="variant",
        seed=0,
        batch_size=batch_size,
        max_length=max_length,
        source="deterministic_variants",
    )

    out = []
    for score, (base, variant_name) in zip(scored, metadata):
        out.append(
            CandidateRecord(
                target_name=base.target_name,
                method=f"{base.method}:{variant_name}",
                seed=base.seed,
                text=score.text,
                target=score.target,
                xentropy=score.xentropy,
                source="robustness",
                extra={
                    "base_method": base.method,
                    "base_target": base.target,
                    "base_xentropy": base.xentropy,
                    "variant": variant_name,
                },
            )
        )
    return out


def robustness_rows(records: Iterable[CandidateRecord]) -> list[dict]:
    rows = []
    for record in records:
        base_target = float(record.extra["base_target"])
        base_xentropy = float(record.extra["base_xentropy"])
        rows.append(
            {
                "target_name": record.target_name,
                "method": record.method,
                "seed": record.seed,
                "variant": record.extra["variant"],
                "target": record.target,
                "xentropy": record.xentropy,
                "delta_target": record.target - base_target,
                "delta_xentropy": record.xentropy - base_xentropy,
            }
        )
    return rows


def robustness_summary_rows(
    records: Iterable[CandidateRecord],
    *,
    target_tolerance: float = 0.0,
) -> list[dict]:
    grouped: dict[tuple[str, str, str], list[CandidateRecord]] = {}
    for record in records:
        base_method = record.extra.get("base_method", record.method.split(":", 1)[0])
        variant = record.extra.get("variant", record.method)
        grouped.setdefault((record.target_name, base_method, variant), []).append(record)

    rows = []
    for (target_name, base_method, variant), group in sorted(grouped.items()):
        deltas = [
            r.target - float(r.extra.get("base_target", r.target))
            for r in group
        ]
        xe_deltas = [
            r.xentropy - float(r.extra.get("base_xentropy", r.xentropy))
            for r in group
        ]
        survived = [delta <= target_tolerance for delta in deltas]
        rows.append(
            {
                "target_name": target_name,
                "base_method": base_method,
                "variant": variant,
                "n": len(group),
                "survival_rate": float(np.mean(survived)),
                "median_delta_target": float(np.median(deltas)),
                "median_delta_xentropy": float(np.median(xe_deltas)),
                "worst_delta_target": float(np.max(deltas)),
            }
        )
    return rows
