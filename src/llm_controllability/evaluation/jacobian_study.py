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


def _load_basis(spec: dict[str, Any], base_dir: Path) -> tuple[int, torch.Tensor]:
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
    basis = torch.as_tensor(np.stack(directions), dtype=torch.float32)
    basis = torch.nn.functional.normalize(basis, dim=1)
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
    return {
        layer: np.stack(values, axis=1)
        for layer, values in columns.items()
    }


def run_jacobian_study(
    spec_path: str | Path,
    out_path: str | Path,
    *,
    example_limit: int = 16,
    epsilon: float = 0.25,
    seed: int = 0,
) -> dict[str, Any]:
    spec_path = Path(spec_path)
    base_dir = spec_path.resolve().parent
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    injection_layer, basis = _load_basis(spec, base_dir)
    capture_layers = [int(layer) for layer in spec["layers"]]
    examples = load_examples(_resolve(spec["data"]["path"], base_dir))
    if any(example.get("split") == "test" for example in examples):
        examples = [
            example for example in examples if example.get("split") == "test"
        ]
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
        for layer, jacobian in jacobians.items():
            row = {
                "model_name": model_config["name"],
                "example_id": example_id,
                "injection_layer": injection_layer,
                "capture_layer": layer,
                "control_dimension": basis.shape[0],
                "epsilon": epsilon,
            }
            row.update(local_controllability(jacobian))
            rows.append(row)
            for component, singular_value in enumerate(
                np.linalg.svd(jacobian, compute_uv=False)
            ):
                spectrum_rows.append(
                    {
                        "model_name": model_config["name"],
                        "example_id": example_id,
                        "injection_layer": injection_layer,
                        "capture_layer": layer,
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
    manifest = {
        "model": model_config["name"],
        "injection_layer": injection_layer,
        "capture_layers": capture_layers,
        "control_dimension": int(basis.shape[0]),
        "epsilon": epsilon,
        "n_examples": len(examples),
        "n_rows": len(rows),
        "n_spectrum_rows": len(spectrum_rows),
        "artifact": str(out_path),
        "spectrum_artifact": str(spectrum_path),
    }
    out_path.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest
