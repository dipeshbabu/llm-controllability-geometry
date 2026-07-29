"""Lossless storage for state arrays and their experiment metadata."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from llm_controllability.controllability.types import (
    ControlChannel,
    InterventionMetadata,
    StateSample,
)


def save_state_samples(samples: Sequence[StateSample], out_dir: str | Path) -> None:
    if not samples:
        raise ValueError("cannot save an empty state sample collection")
    widths = {sample.state.shape[0] for sample in samples}
    if len(widths) != 1:
        raise ValueError("one state archive cannot mix hidden-state widths")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    states = np.stack([sample.state for sample in samples]).astype(np.float32, copy=False)
    token_arrays = [sample.token_states for sample in samples]
    archive = {"states": states}
    if any(tokens is not None for tokens in token_arrays):
        width = states.shape[1]
        normalized = [
            tokens
            if tokens is not None
            else np.empty((0, width), dtype=np.float32)
            for tokens in token_arrays
        ]
        offsets = np.zeros(len(normalized) + 1, dtype=np.int64)
        offsets[1:] = np.cumsum([tokens.shape[0] for tokens in normalized])
        archive["token_states"] = np.concatenate(normalized, axis=0)
        archive["token_offsets"] = offsets
    np.savez_compressed(out_dir / "states.npz", **archive)
    with (out_dir / "samples.jsonl").open("w", encoding="utf-8") as handle:
        for index, sample in enumerate(samples):
            handle.write(json.dumps(sample.metadata_row(index), sort_keys=True) + "\n")


def load_state_samples(out_dir: str | Path) -> list[StateSample]:
    out_dir = Path(out_dir)
    with np.load(out_dir / "states.npz") as archive:
        states = archive["states"]
        token_values = archive.get("token_states")
        token_offsets = archive.get("token_offsets")
    rows = [
        json.loads(line)
        for line in (out_dir / "samples.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != states.shape[0]:
        raise ValueError("state array and metadata row counts do not match")

    samples = []
    for row in rows:
        index = int(row["state_index"])
        tokens = None
        if token_values is not None and token_offsets is not None:
            start, end = int(token_offsets[index]), int(token_offsets[index + 1])
            if end > start:
                tokens = token_values[start:end]
        samples.append(
            StateSample(
                example_id=str(row["example_id"]),
                model_name=str(row["model_name"]),
                layer=int(row["layer"]),
                intervention=InterventionMetadata(
                    name=str(row["intervention"]),
                    channel=ControlChannel(row["channel"]),
                    control_cost=float(row["control_cost"]),
                    parameters=row.get("parameters", {}),
                ),
                state=states[index],
                token_states=tokens,
                prompt=str(row["prompt"]),
                output=str(row["output"]),
                behavior_preserved=bool(row["behavior_preserved"]),
                constraint_results={
                    str(key): bool(value)
                    for key, value in row.get("constraint_results", {}).items()
                },
                metrics={
                    str(key): float(value)
                    for key, value in row.get("metrics", {}).items()
                },
                tags={
                    str(key): str(value)
                    for key, value in row.get("tags", {}).items()
                },
                seed=int(row.get("seed", 0)),
            )
        )
    return samples
