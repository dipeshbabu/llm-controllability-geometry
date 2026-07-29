"""Residual direction fitting and layer sweep utilities."""

from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

from llm_controllability.activations.probes import (
    collect_residual_states,
    collect_residual_states_many,
)


def load_contrast_pairs(path: str | Path) -> tuple[list[str], list[str]]:
    path = Path(path)
    a_texts = []
    b_texts = []
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
    else:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        rows = data["pairs"] if isinstance(data, dict) else data

    for row in rows:
        if "a" in row and "b" in row:
            a_texts.append(row["a"])
            b_texts.append(row["b"])
        else:
            a_texts.append(row["positive"])
            b_texts.append(row["negative"])
    if len(a_texts) != len(b_texts) or not a_texts:
        raise ValueError("Contrast file must contain nonempty paired texts")
    return a_texts, b_texts


def mean_difference_direction(
    a_states: np.ndarray,
    b_states: np.ndarray,
) -> np.ndarray:
    direction = a_states.mean(axis=0) - b_states.mean(axis=0)
    norm = np.linalg.norm(direction)
    if norm == 0:
        raise ValueError("Contrast direction has zero norm")
    return direction / norm


def projection_gap(
    a_states: np.ndarray,
    b_states: np.ndarray,
    direction: np.ndarray,
) -> float:
    return float((a_states @ direction).mean() - (b_states @ direction).mean())


def fit_direction_for_layer(
    model,
    tokenizer,
    layer: int,
    a_texts: Sequence[str],
    b_texts: Sequence[str],
    *,
    pooling: str = "last",
    max_len: int = 256,
    batch_size: int = 16,
) -> tuple[np.ndarray, dict]:
    a_states = collect_residual_states(
        model,
        layer,
        tokenizer,
        a_texts,
        max_len=max_len,
        pooling=pooling,
        batch_size=batch_size,
    )
    b_states = collect_residual_states(
        model,
        layer,
        tokenizer,
        b_texts,
        max_len=max_len,
        pooling=pooling,
        batch_size=batch_size,
    )
    direction = mean_difference_direction(a_states, b_states)
    gap = projection_gap(a_states, b_states, direction)
    row = {
        "layer": int(layer),
        "projection_gap": gap,
        "a_mean_projection": float((a_states @ direction).mean()),
        "b_mean_projection": float((b_states @ direction).mean()),
        "n_pairs": len(a_texts),
        "pooling": pooling,
    }
    return direction, row


def fit_direction_sweep(
    model,
    tokenizer,
    contrast_path: str | Path,
    layers: Sequence[int],
    out_dir: str | Path,
    *,
    name: str,
    eval_contrast_path: str | Path | None = None,
    pooling: str = "last",
    max_len: int = 256,
    batch_size: int = 16,
) -> list[dict]:
    a_texts, b_texts = load_contrast_pairs(contrast_path)
    eval_a_texts = eval_b_texts = None
    if eval_contrast_path is not None:
        eval_a_texts, eval_b_texts = load_contrast_pairs(eval_contrast_path)
    out_dir = Path(out_dir)
    vector_dir = out_dir / "vectors"
    vector_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    selected_layers = tuple(dict.fromkeys(int(layer) for layer in layers))
    a_states_by_layer = collect_residual_states_many(
        model,
        selected_layers,
        tokenizer,
        a_texts,
        max_len=max_len,
        pooling=pooling,
        batch_size=batch_size,
    )
    b_states_by_layer = collect_residual_states_many(
        model,
        selected_layers,
        tokenizer,
        b_texts,
        max_len=max_len,
        pooling=pooling,
        batch_size=batch_size,
    )
    eval_a_by_layer = eval_b_by_layer = None
    if eval_a_texts is not None and eval_b_texts is not None:
        eval_a_by_layer = collect_residual_states_many(
            model,
            selected_layers,
            tokenizer,
            eval_a_texts,
            max_len=max_len,
            pooling=pooling,
            batch_size=batch_size,
        )
        eval_b_by_layer = collect_residual_states_many(
            model,
            selected_layers,
            tokenizer,
            eval_b_texts,
            max_len=max_len,
            pooling=pooling,
            batch_size=batch_size,
        )

    for layer in selected_layers:
        a_states = a_states_by_layer[layer]
        b_states = b_states_by_layer[layer]
        direction = mean_difference_direction(a_states, b_states)
        row = {
            "layer": layer,
            "projection_gap": projection_gap(a_states, b_states, direction),
            "a_mean_projection": float((a_states @ direction).mean()),
            "b_mean_projection": float((b_states @ direction).mean()),
            "n_pairs": len(a_texts),
            "pooling": pooling,
        }
        if eval_a_by_layer is not None and eval_b_by_layer is not None:
            row["eval_projection_gap"] = projection_gap(
                eval_a_by_layer[layer],
                eval_b_by_layer[layer],
                direction,
            )
            row["n_eval_pairs"] = len(eval_a_texts)
        vector_path = vector_dir / f"{name}_L{layer}.npy"
        np.save(vector_path, direction.astype(np.float32))
        row["name"] = name
        row["vector_path"] = str(vector_path)
        rows.append(row)
    write_direction_rows(rows, out_dir / f"{name}_layer_sweep.csv")
    return rows


def write_direction_rows(rows: Sequence[dict], path: str | Path) -> None:
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


def top_direction_specs(
    rows: Sequence[dict],
    *,
    top_k: int = 3,
    bidirectional: bool = False,
) -> list[dict]:
    metric = (
        "eval_projection_gap"
        if rows and all("eval_projection_gap" in row for row in rows)
        else "projection_gap"
    )
    ranked = sorted(
        rows,
        key=lambda row: (
            float(row[metric])
            if metric == "eval_projection_gap"
            else abs(float(row[metric]))
        ),
        reverse=True,
    )
    if (
        metric == "eval_projection_gap"
        and ranked
        and float(ranked[0][metric]) <= 0
    ):
        raise ValueError("no fitted direction has a positive held-out projection gap")
    specs = []
    for row in ranked[:top_k]:
        layer = int(row["layer"])
        base = {
            "type": "residual",
            "layer": layer,
            "vector_path": row["vector_path"],
        }
        if bidirectional:
            specs.extend(
                [
                    {
                        **base,
                        "name": f"{row['name']}_residual_L{layer}_decrease",
                        "minimize": True,
                    },
                    {
                        **base,
                        "name": f"{row['name']}_residual_L{layer}_increase",
                        "minimize": False,
                    },
                ]
            )
        else:
            specs.append(
                {
                    **base,
                    "name": f"{row['name']}_residual_L{layer}",
                    "minimize": True,
                }
            )
    return specs


def save_torch_vector(vector: np.ndarray, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(torch.tensor(vector, dtype=torch.float32), path)
