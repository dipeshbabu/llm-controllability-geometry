"""Build reproducible study specifications from fitted directions and prompt searches."""

from __future__ import annotations

import csv
import json
import os
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from llm_controllability.models.adapters import profile_manifest, resolve_model_profile
from llm_controllability.reporting.results import pareto_frontier, records_from_csv


def export_prompt_controls(
    records_path: str | Path,
    out_path: str | Path,
    *,
    target_name: str | None = None,
    methods: Sequence[str] | None = None,
    top_n: int = 32,
    minimize: bool = True,
) -> list[str]:
    records = records_from_csv(records_path)
    if target_name is not None:
        records = [record for record in records if record.target_name == target_name]
    if methods is not None:
        allowed = set(methods)
        records = [record for record in records if record.method in allowed]
    frontier = pareto_frontier(records, minimize=minimize)
    ranked = sorted(
        frontier,
        key=lambda record: (
            record.target if minimize else -record.target,
            record.xentropy,
        ),
    )
    texts = []
    seen = set()
    for record in ranked:
        text = " ".join(record.text.split())
        if text and text not in seen:
            texts.append(text)
            seen.add(text)
        if len(texts) >= top_n:
            break
    if not texts:
        raise ValueError("no prompt controls matched the requested filters")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(texts) + "\n", encoding="utf-8")
    return texts


def export_bidirectional_prompt_controls(
    records_path: str | Path,
    out_path: str | Path,
    *,
    target_base: str,
    methods: Sequence[str] | None = None,
    top_n_per_direction: int = 8,
) -> list[str]:
    """Export balanced Pareto controls for projection decrease and increase."""

    if top_n_per_direction <= 0:
        raise ValueError("top_n_per_direction must be positive")
    out_path = Path(out_path)
    decrease_path = out_path.with_name(f"{out_path.stem}_decrease.txt")
    increase_path = out_path.with_name(f"{out_path.stem}_increase.txt")
    decrease = export_prompt_controls(
        records_path,
        decrease_path,
        target_name=f"{target_base}_decrease",
        methods=methods,
        top_n=top_n_per_direction,
        minimize=True,
    )
    increase = export_prompt_controls(
        records_path,
        increase_path,
        target_name=f"{target_base}_increase",
        methods=methods,
        top_n=top_n_per_direction,
        minimize=False,
    )
    combined = []
    seen = set()
    for index in range(max(len(decrease), len(increase))):
        for values in (decrease, increase):
            if index < len(values) and values[index] not in seen:
                combined.append(values[index])
                seen.add(values[index])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(combined) + "\n", encoding="utf-8")
    return combined


def best_direction_target(direction_sweep: str | Path) -> str:
    path = Path(direction_sweep)
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("direction sweep is empty")
    metric = (
        "eval_projection_gap"
        if all(row.get("eval_projection_gap") not in (None, "") for row in rows)
        else "projection_gap"
    )
    best = max(
        rows,
        key=lambda row: (
            float(row[metric])
            if metric == "eval_projection_gap"
            else abs(float(row[metric]))
        ),
    )
    if metric == "eval_projection_gap" and float(best[metric]) <= 0:
        raise ValueError("no fitted direction has a positive held-out projection gap")
    return f"{best['name']}_residual_L{int(best['layer'])}"


def _path_for_spec(path: Path, spec_path: Path) -> str:
    return Path(os.path.relpath(path.resolve(), spec_path.resolve().parent)).as_posix()


def _resolve_vector_path(value: str, sweep_path: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    if path.exists():
        return path.resolve()
    candidate = sweep_path.parent / path
    if candidate.exists():
        return candidate.resolve()
    raise FileNotFoundError(f"direction vector does not exist: {value}")


def build_study_spec(
    *,
    out_path: str | Path,
    model_name: str,
    data_path: str | Path,
    direction_sweep: str | Path,
    capture_layers: Sequence[int] | None,
    prompt_controls_path: str | Path,
    natural_controls_path: str | Path,
    example_limit: int | None = None,
    dtype: str = "bfloat16",
    device_map: str = "auto",
    attn_implementation: str | None = None,
    revision: str | None = None,
    strengths: Sequence[float] = (
        -16.0,
        -8.0,
        -4.0,
        -2.0,
        -1.0,
        -0.5,
        0.5,
        1.0,
        2.0,
        4.0,
        8.0,
        16.0,
    ),
    ablation_fractions: Sequence[float] = (0.25, 0.5, 0.75, 1.0),
    max_new_tokens: int = 128,
    maximum_quality_drop: float = 0.75,
    maximum_control_cost: float = 64.0,
    semantic_model: str | None = None,
    minimum_semantic_similarity: float = 0.80,
    store_token_states: bool = False,
    cmap_direction_budget: int = 0,
    cmap_query_budget: int = 512,
    cmap_validation_examples: int = 8,
    cmap_test_examples: int = 16,
    seed: int = 0,
) -> dict:
    out_path = Path(out_path)
    sweep_path = Path(direction_sweep)
    if revision is None:
        manifest_path = sweep_path.parent / "direction_manifest.json"
        if manifest_path.exists():
            direction_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            revision = direction_manifest.get("resolved_revision")
    with sweep_path.open("r", encoding="utf-8", newline="") as handle:
        sweep_rows = list(csv.DictReader(handle))
    if not sweep_rows:
        raise ValueError("direction sweep is empty")
    metric = (
        "eval_projection_gap"
        if all(row.get("eval_projection_gap") not in (None, "") for row in sweep_rows)
        else "projection_gap"
    )
    best = max(
        sweep_rows,
        key=lambda row: (
            float(row[metric])
            if metric == "eval_projection_gap"
            else abs(float(row[metric]))
        ),
    )
    if metric == "eval_projection_gap" and float(best[metric]) <= 0:
        raise ValueError("no fitted direction has a positive held-out projection gap")
    direction_path = _resolve_vector_path(best["vector_path"], sweep_path)
    direction = np.load(direction_path).astype(np.float64)
    direction /= max(np.linalg.norm(direction), 1e-12)
    rng = np.random.default_rng(seed)
    random_control = rng.normal(size=direction.shape)
    random_control -= np.dot(random_control, direction) * direction
    random_control /= max(np.linalg.norm(random_control), 1e-12)
    random_path = out_path.parent / f"{out_path.stem}_orthogonal_random.npy"
    random_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(random_path, random_control.astype(np.float32))

    layer = int(best["layer"])
    if capture_layers is None:
        available = sorted({int(row["layer"]) for row in sweep_rows})
        positions = np.linspace(0, len(available) - 1, num=min(5, len(available)))
        capture_layers = [available[round(position)] for position in positions]
    layers = sorted({int(value) for value in capture_layers} | {layer})
    a_mean = float(best["a_mean_projection"])
    b_mean = float(best["b_mean_projection"])
    lower = min(a_mean, b_mean)
    upper = max(a_mean, b_mean)
    span = max(upper - lower, 1e-6)
    setpoints = np.linspace(lower - span, upper + span, num=9).tolist()
    direction_value = _path_for_spec(direction_path, out_path)
    random_value = _path_for_spec(random_path, out_path)
    prompt_value = _path_for_spec(Path(prompt_controls_path), out_path)
    natural_value = _path_for_spec(Path(natural_controls_path), out_path)

    activation = {
        "name": "activation_addition",
        "type": "activation_addition",
        "layer": layer,
        "direction": direction_value,
        "strengths": list(strengths),
        "token_scope": "last",
    }
    model_profile = profile_manifest(resolve_model_profile(model_name))
    spec = {
        "model": {
            "name": model_name,
            "dtype": dtype,
            "device_map": device_map,
            "attn_implementation": attn_implementation,
            "revision": revision,
            **model_profile,
        },
        "data": {
            "path": _path_for_spec(Path(data_path), out_path),
            "limit": example_limit,
        },
        "layers": layers,
        "pooling": "last",
        "max_length": 2048,
        "store_token_states": store_token_states,
        "generation": {
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "num_beams": 1,
        },
        "constraints": {
            "task": {
                "require_correct": True,
                "minimum_score": 1.0,
                "maximum_drop": 0.0,
            },
            "quality": {"maximum_drop": maximum_quality_drop},
            "budget": {"maximum_cost": maximum_control_cost},
        },
        "target_directions": {str(layer): direction_value},
        "interventions": [
            {
                "name": "optimized_prompt",
                "type": "prompt",
                "texts_path": prompt_value,
                "mode": "suffix",
            },
            {
                "name": "surface_control",
                "type": "mapped_prompt",
                "path": natural_value,
                "control_names": [
                    "natural_paraphrase",
                    "style_concise",
                    "capitalization",
                    "topic_matched",
                ],
                "strict": True,
            },
            activation,
            {
                "name": "directional_ablation",
                "type": "directional_ablation",
                "layer": layer,
                "direction": direction_value,
                "fractions": list(ablation_fractions),
                "token_scope": "last",
            },
            {
                "name": "adaptive_controller",
                "type": "adaptive",
                "layer": layer,
                "direction": direction_value,
                "setpoints": setpoints,
                "kp": 0.5,
                "ki": 0.05,
                "kd": 0.1,
                "max_update": max(abs(value) for value in strengths),
                "token_scope": "last",
            },
            {
                "name": "orthogonal_random",
                "type": "activation_addition",
                "channel": "random",
                "layer": layer,
                "direction": random_value,
                "strengths": list(strengths),
                "token_scope": "last",
            },
            {
                "name": "hybrid",
                "type": "hybrid",
                "prompt": {
                    "name": "optimized",
                    "type": "prompt",
                    "texts_path": prompt_value,
                    "mode": "suffix",
                },
                "activation": {
                    "name": "activation",
                    "type": "activation_addition",
                    "layer": layer,
                    "direction": direction_value,
                    "strength": float(strengths[len(strengths) // 2]),
                    "token_scope": "last",
                },
            },
        ],
        "seed": seed,
    }
    if cmap_direction_budget > 0:
        spec["cmap"] = {
            "enabled": True,
            "layer": layer,
            "direction_budget": cmap_direction_budget,
            "query_budget": cmap_query_budget,
            "validation_examples": cmap_validation_examples,
            "test_examples": cmap_test_examples,
            "candidate_pool_size": 64,
            "initial_strength": 0.5,
            "maximum_strength": min(
                maximum_control_cost,
                max(abs(float(value)) for value in strengths),
            ),
            "expansion_factor": 2.0,
            "boundary_steps": 3,
            "required_preservation_rate": 0.75,
            "rank_tolerance": 1e-4,
            "minimum_novelty": 0.05,
            "stagnation_patience": 2,
            "token_scope": "last",
            "seed": seed,
        }
    if semantic_model is not None:
        spec["constraints"]["semantic"] = {
            "model_name": semantic_model,
            "device": "cpu",
            "minimum_similarity": minimum_semantic_similarity,
            "minimum_prompt_similarity": minimum_semantic_similarity,
            "check_prompt": True,
            "batch_size": 16,
        }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
    return spec
