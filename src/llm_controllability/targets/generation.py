"""Generate JSON target specs for suppression experiments."""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from pathlib import Path

from llm_controllability.models.adapters import profile_manifest, resolve_model_profile


def logit_specs(tokens: Iterable[str], *, prefix: str = "logit") -> list[dict]:
    specs = []
    for token in tokens:
        clean = token.strip().replace(" ", "_") or "blank"
        specs.append(
            {
                "name": f"{prefix}_{clean}",
                "type": "logit",
                "token_text": token,
                "banned_text": token.strip(),
                "minimize": True,
            }
        )
    return specs


def neuron_specs(
    layers: Sequence[int],
    neurons: Sequence[int],
    *,
    prefix: str = "neuron",
) -> list[dict]:
    return [
        {
            "name": f"{prefix}_L{layer}_N{neuron}",
            "type": "neuron",
            "layer": int(layer),
            "neuron": int(neuron),
            "minimize": True,
        }
        for layer in layers
        for neuron in neurons
    ]


def residual_specs(
    vector_paths: Sequence[str | Path],
    *,
    layer_by_file: dict[str, int] | None = None,
    default_layer: int | None = None,
    prefix: str = "residual",
) -> list[dict]:
    specs = []
    for path_like in vector_paths:
        path = Path(path_like)
        layer = None
        if layer_by_file is not None:
            layer = layer_by_file.get(path.name) or layer_by_file.get(path.stem)
        if layer is None:
            layer = default_layer
        if layer is None:
            raise ValueError(f"No layer provided for residual vector {path}")
        specs.append(
            {
                "name": f"{prefix}_L{layer}_{path.stem}",
                "type": "residual",
                "layer": int(layer),
                "vector_path": str(path),
                "minimize": True,
            }
        )
    return specs


def write_spec(
    targets: Sequence[dict],
    path: str | Path,
    *,
    model_name: str | None = None,
    model_size: str | None = None,
    texts_path: str | None = None,
    attn_implementation: str | None = None,
    device_map: str | None = None,
    revision: str | None = None,
) -> None:
    spec = {"targets": list(targets)}
    if model_name:
        spec["model_name"] = model_name
        spec["model_profile"] = profile_manifest(resolve_model_profile(model_name))
    if model_size:
        spec["model_size"] = model_size
    if texts_path:
        spec["texts_path"] = texts_path
    if attn_implementation:
        spec["attn_implementation"] = attn_implementation
    if device_map:
        spec["device_map"] = device_map
    if revision:
        spec["revision"] = revision
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")


def parse_int_list(value: str) -> list[int]:
    out = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return out
