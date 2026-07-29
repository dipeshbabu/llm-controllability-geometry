"""Build target runners from JSON compatible specifications."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from llm_controllability.models.adapters import model_device, model_dtype
from llm_controllability.optimization.runners import (
    logit_diff_runner,
    neuron_runner,
    residual_runner,
)


def token_id_from_text(tokenizer, text: str) -> int:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError(
            f"Expected {text!r} to encode to one token, got {len(ids)} tokens: {ids}"
        )
    return ids[0]


def load_vector(path: str | Path, model) -> torch.Tensor:
    path = Path(path)
    if path.suffix == ".pt":
        vector = torch.load(path, map_location=model_device(model))
    else:
        vector = torch.tensor(np.load(path), device=model_device(model))
    vector = vector.to(dtype=model_dtype(model), device=model_device(model))
    return vector / vector.norm().clamp_min(1e-8)


def build_runner_from_spec(model, tokenizer, spec: dict[str, Any]):
    target_type = spec["type"]
    if target_type == "logit":
        token_id = spec.get("token_id")
        if token_id is None:
            token_id = token_id_from_text(tokenizer, spec["token_text"])
        runner = logit_diff_runner(
            model,
            tokenizer,
            token_id=int(token_id),
            banned_text=spec.get("banned_text", spec.get("token_text", "")),
            check_retokenization=bool(spec.get("check_retokenization", False)),
        )
    elif target_type == "neuron":
        runner = neuron_runner(
            model,
            tokenizer,
            layer=int(spec["layer"]),
            neuron=int(spec["neuron"]),
            check_retokenization=bool(spec.get("check_retokenization", False)),
        )
    elif target_type == "residual":
        vector = load_vector(spec["vector_path"], model)
        runner = residual_runner(
            model,
            tokenizer,
            layer=int(spec["layer"]),
            vector=vector,
            check_retokenization=bool(spec.get("check_retokenization", False)),
            minimize=bool(spec.get("minimize", True)),
        )
    else:
        raise ValueError(f"Unknown target type: {target_type}")

    runner.minimize = bool(spec.get("minimize", True))
    return runner


def target_name(spec: dict[str, Any]) -> str:
    if "name" in spec:
        return str(spec["name"])
    if spec["type"] == "logit":
        return f"logit:{spec.get('token_text', spec.get('token_id'))}"
    if spec["type"] == "neuron":
        return f"neuron:L{spec['layer']}:N{spec['neuron']}"
    if spec["type"] == "residual":
        return f"residual:L{spec['layer']}:{Path(spec['vector_path']).stem}"
    return str(spec["type"])
