"""Finite difference local controllability study for residual controls."""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from llm_controllability.interventions.core import _hidden_from_output, _replace_hidden
from llm_controllability.models.adapters import (
    ensure_padding_token,
    model_device,
    tokenize_prompts,
)
from llm_controllability.models.architecture import get_layers
from llm_controllability.models.loading import load_model
from llm_controllability.reachability.collection import load_examples
from llm_controllability.reachability.jacobians import local_controllability


def _resolve(path: str, base_dir: Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else base_dir / value


def _orthonormal_control_basis(
    anchors: list[np.ndarray],
    *,
    dimension: int,
    seed: int,
) -> torch.Tensor:
    if dimension <= 0:
        raise ValueError("control basis dimension must be positive")
    width = int(np.asarray(anchors[0]).size)
    if dimension > width:
        raise ValueError(
            f"control basis dimension {dimension} exceeds residual width {width}"
        )
    rng = np.random.default_rng(seed)
    candidates = [np.asarray(value, dtype=np.float64).reshape(-1) for value in anchors]
    candidates.extend(rng.normal(size=width) for _ in range(4 * dimension))
    basis: list[np.ndarray] = []
    for candidate in candidates:
        if candidate.size != width:
            raise ValueError("all control basis anchors must have the same width")
        residual = candidate.copy()
        for vector in basis:
            residual -= np.dot(residual, vector) * vector
        norm = np.linalg.norm(residual)
        if norm <= 1e-10:
            continue
        basis.append(residual / norm)
        if len(basis) == dimension:
            break
    if len(basis) != dimension:
        raise RuntimeError(
            "failed to construct the requested orthonormal control basis"
        )
    return torch.as_tensor(np.stack(basis), dtype=torch.float32)


def _load_basis(
    spec: dict[str, Any],
    base_dir: Path,
    *,
    dimension: int,
    seed: int,
) -> tuple[int, torch.Tensor]:
    target_directions = spec.get("target_directions", {})
    if len(target_directions) != 1:
        raise ValueError("Jacobian study requires exactly one target direction")
    layer_text, concept_path = next(iter(target_directions.items()))
    random_config = next(
        (
            item
            for item in spec["interventions"]
            if item.get("name") == "orthogonal_random"
            and item.get("type") == "activation_addition"
        ),
        None,
    )
    if random_config is None:
        raise ValueError("study spec does not contain the orthogonal random control")
    directions = [
        np.load(_resolve(str(concept_path), base_dir)),
        np.load(_resolve(str(random_config["direction"]), base_dir)),
    ]
    basis = _orthonormal_control_basis(
        directions,
        dimension=dimension,
        seed=seed,
    )
    return int(layer_text), basis


def capture_controlled_states(
    model: torch.nn.Module,
    model_inputs: dict[str, torch.Tensor],
    *,
    injection_layer: int,
    capture_layers: list[int],
    basis: torch.Tensor,
    coefficients: torch.Tensor,
) -> dict[int, np.ndarray]:
    """Inject low dimensional residual controls and capture last token states."""

    blocks = get_layers(model)
    captured: dict[int, torch.Tensor] = {}

    def inject(module, inputs, output):
        hidden = _hidden_from_output(output)
        vectors = basis.to(device=hidden.device, dtype=hidden.dtype)
        control = coefficients.to(device=hidden.device, dtype=hidden.dtype)
        delta = control @ vectors
        updated = hidden.clone()
        updated[:, -1, :] = updated[:, -1, :] + delta
        return _replace_hidden(output, updated)

    handles = [blocks[injection_layer].register_forward_hook(inject)]
    for layer in capture_layers:

        def capture(module, inputs, output, layer=layer):
            captured[layer] = _hidden_from_output(output)[:, -1]

        handles.append(blocks[layer].register_forward_hook(capture))
    try:
        with torch.no_grad():
            model(**model_inputs, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    return {
        layer: value[0].detach().float().cpu().numpy()
        for layer, value in captured.items()
    }


def finite_difference_jacobians(
    model: torch.nn.Module,
    model_inputs: dict[str, torch.Tensor],
    *,
    injection_layer: int,
    capture_layers: list[int],
    basis: torch.Tensor,
    epsilon: float,
) -> dict[int, np.ndarray]:
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    columns: dict[int, list[np.ndarray]] = {layer: [] for layer in capture_layers}
    for coordinate in range(basis.shape[0]):
        positive = torch.zeros(basis.shape[0], dtype=torch.float32)
        negative = torch.zeros(basis.shape[0], dtype=torch.float32)
        positive[coordinate] = epsilon
        negative[coordinate] = -epsilon
        plus = capture_controlled_states(
            model,
            model_inputs,
            injection_layer=injection_layer,
            capture_layers=capture_layers,
            basis=basis,
            coefficients=positive,
        )
        minus = capture_controlled_states(
            model,
            model_inputs,
            injection_layer=injection_layer,
            capture_layers=capture_layers,
            basis=basis,
            coefficients=negative,
        )
        for layer in capture_layers:
            columns[layer].append((plus[layer] - minus[layer]) / (2.0 * epsilon))
    return {layer: np.stack(values, axis=1) for layer, values in columns.items()}


def run_jacobian_study(
    spec_path: str | Path,
    out_path: str | Path,
    *,
    example_limit: int = 16,
    epsilon: float = 0.25,
    basis_dimensions: tuple[int, ...] = (8, 16, 32),
    seed: int = 0,
) -> dict[str, Any]:
    spec_path = Path(spec_path)
    base_dir = spec_path.resolve().parent
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    dimensions = sorted({int(value) for value in basis_dimensions})
    if not dimensions or dimensions[0] <= 0:
        raise ValueError("basis_dimensions must contain positive integers")
    injection_layer, basis = _load_basis(
        spec,
        base_dir,
        dimension=dimensions[-1],
        seed=seed,
    )
    capture_layers = [int(layer) for layer in spec["layers"]]
    examples = load_examples(_resolve(spec["data"]["path"], base_dir))
    if any(example.get("split") == "test" for example in examples):
        examples = [example for example in examples if example.get("split") == "test"]
    random.Random(seed).shuffle(examples)
    examples = examples[:example_limit]

    model_config = spec["model"]
    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[model_config.get("dtype", "bfloat16")]
    model, tokenizer = load_model(
        model_name=model_config["name"],
        tokenizer_name=model_config.get("tokenizer_name"),
        attn_implementation=model_config.get("attn_implementation"),
        device_map=model_config.get("device_map", "auto"),
        torch_dtype=dtype,
        prompt_format=model_config.get("prompt_format", "auto"),
        enable_thinking=model_config.get("enable_thinking"),
        trust_remote_code=model_config.get("trust_remote_code"),
        revision=model_config.get("revision"),
    )
    tokenizer, _ = ensure_padding_token(tokenizer)
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    rows = []
    spectrum_rows = []
    for index, example in enumerate(examples):
        example_id = str(example.get("id", index))
        model_inputs = tokenize_prompts(
            tokenizer,
            str(example["prompt"]),
            return_tensors="pt",
            truncation=True,
            max_length=int(spec.get("max_length", 2048)),
        ).to(model_device(model))
        jacobians = finite_difference_jacobians(
            model,
            dict(model_inputs),
            injection_layer=injection_layer,
            capture_layers=capture_layers,
            basis=basis,
            epsilon=epsilon,
        )
        for layer, full_jacobian in jacobians.items():
            for dimension in dimensions:
                jacobian = full_jacobian[:, :dimension]
                summary = local_controllability(jacobian)
                singular = np.linalg.svd(jacobian, compute_uv=False)
                row = {
                    "model_name": model_config["name"],
                    "example_id": example_id,
                    "injection_layer": injection_layer,
                    "capture_layer": layer,
                    "control_dimension": dimension,
                    "epsilon": epsilon,
                    **summary,
                    "rank_fraction": float(summary["rank"]) / dimension,
                    "squared_gain": float(np.square(singular).sum()),
                }
                rows.append(row)
                for component, singular_value in enumerate(singular):
                    spectrum_rows.append(
                        {
                            "model_name": model_config["name"],
                            "example_id": example_id,
                            "injection_layer": injection_layer,
                            "capture_layer": layer,
                            "control_dimension": dimension,
                            "component": component,
                            "singular_value": float(singular_value),
                        }
                    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with out_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    else:
        out_path.write_text("", encoding="utf-8")
    spectrum_path = out_path.with_name(f"{out_path.stem}_spectrum.csv")
    if spectrum_rows:
        with spectrum_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(spectrum_rows[0]))
            writer.writeheader()
            writer.writerows(spectrum_rows)
    else:
        spectrum_path.write_text("", encoding="utf-8")
    basis_path = out_path.with_name(f"{out_path.stem}_control_basis.npy")
    np.save(basis_path, basis.detach().cpu().numpy())
    manifest = {
        "model": model_config["name"],
        "injection_layer": injection_layer,
        "capture_layers": capture_layers,
        "control_dimensions": dimensions,
        "maximum_control_dimension": int(basis.shape[0]),
        "basis_construction": "target_then_declared_orthogonal_then_seeded_gram_schmidt",
        "epsilon": epsilon,
        "n_examples": len(examples),
        "n_rows": len(rows),
        "n_spectrum_rows": len(spectrum_rows),
        "artifact": str(out_path),
        "spectrum_artifact": str(spectrum_path),
        "basis_artifact": str(basis_path),
    }
    out_path.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest
