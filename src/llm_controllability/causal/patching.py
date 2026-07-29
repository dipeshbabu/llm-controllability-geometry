"""Activation patching, component ablation, and route comparison."""

from __future__ import annotations

import contextlib
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import torch

from llm_controllability.interventions.core import _hidden_from_output, _replace_hidden
from llm_controllability.models.architecture import (
    get_attention_head_layout,
    get_attention_output_projection,
    get_layers,
)


@dataclass
class ActivationCache:
    """Capture block outputs for later counterfactual replacement."""

    layers: Sequence[int]
    detach_to_cpu: bool = True
    values: dict[int, torch.Tensor] = field(default_factory=dict, init=False)

    @contextlib.contextmanager
    def capture(self, model: torch.nn.Module) -> Iterator[ActivationCache]:
        self.values = {}
        handles = []
        for layer in self.layers:
            def hook(module, inputs, output, layer=layer):
                hidden = _hidden_from_output(output).detach()
                self.values[layer] = hidden.cpu() if self.detach_to_cpu else hidden

            handles.append(get_layers(model)[layer].register_forward_hook(hook))
        try:
            yield self
        finally:
            for handle in handles:
                handle.remove()


@dataclass
class StatePatching:
    """Replace selected token states at one transformer layer."""

    layer: int
    source: torch.Tensor
    token_scope: Literal["all", "last"] = "all"
    blend: float = 1.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.blend <= 1.0:
            raise ValueError("blend must lie in [0, 1]")

    @contextlib.contextmanager
    def apply(self, model: torch.nn.Module) -> Iterator[None]:
        block = get_layers(model)[self.layer]

        def hook(module, inputs, output):
            hidden = _hidden_from_output(output)
            source = self.source.to(device=hidden.device, dtype=hidden.dtype)
            if source.shape[0] == 1 and hidden.shape[0] > 1:
                source = source.expand(hidden.shape[0], -1, -1)
            if self.token_scope == "last":
                if source.shape[-1] != hidden.shape[-1]:
                    raise ValueError("source and target hidden widths differ")
                source_last = source[:, -1:, :]
                patched = hidden.clone()
                patched[:, -1:, :] = (
                    (1.0 - self.blend) * hidden[:, -1:, :] + self.blend * source_last
                )
            else:
                if source.shape != hidden.shape:
                    raise ValueError(
                        f"all-token patch requires matching shapes, got {source.shape} and {hidden.shape}"
                    )
                patched = (1.0 - self.blend) * hidden + self.blend * source
            return _replace_hidden(output, patched)

        handle = block.register_forward_hook(hook)
        try:
            yield
        finally:
            handle.remove()


@dataclass
class ComponentAblation:
    """Ablate a module output with zeros or a supplied reference mean."""

    module: torch.nn.Module
    mode: Literal["zero", "mean"] = "zero"
    reference_mean: torch.Tensor | None = None
    scale: float = 1.0

    @contextlib.contextmanager
    def apply(self) -> Iterator[None]:
        if self.mode == "mean" and self.reference_mean is None:
            raise ValueError("mean ablation requires reference_mean")

        def hook(module, inputs, output):
            hidden = _hidden_from_output(output)
            if self.mode == "zero":
                replacement = torch.zeros_like(hidden)
            else:
                assert self.reference_mean is not None
                replacement = self.reference_mean.to(
                    device=hidden.device,
                    dtype=hidden.dtype,
                )
                replacement = torch.broadcast_to(replacement, hidden.shape)
            updated = (1.0 - self.scale) * hidden + self.scale * replacement
            return _replace_hidden(output, updated)

        handle = self.module.register_forward_hook(hook)
        try:
            yield
        finally:
            handle.remove()


@dataclass
class ModuleOutputCache:
    """Cache the tensor-valued output of explicitly selected modules."""

    modules: Mapping[str, torch.nn.Module]
    detach_to_cpu: bool = True
    values: dict[str, torch.Tensor] = field(default_factory=dict, init=False)

    @contextlib.contextmanager
    def capture(self) -> Iterator[ModuleOutputCache]:
        handles = []
        self.values = {}
        for name, module in self.modules.items():
            def hook(current_module, inputs, output, name=name):
                hidden = _hidden_from_output(output).detach()
                self.values[name] = hidden.cpu() if self.detach_to_cpu else hidden

            handles.append(module.register_forward_hook(hook))
        try:
            yield self
        finally:
            for handle in handles:
                handle.remove()


@dataclass
class ModuleOutputPatching:
    """Replace one module output with a cached counterfactual value."""

    module: torch.nn.Module
    source: torch.Tensor
    blend: float = 1.0

    @contextlib.contextmanager
    def apply(self) -> Iterator[None]:
        if not 0.0 <= self.blend <= 1.0:
            raise ValueError("blend must lie in [0, 1]")

        def hook(module, inputs, output):
            hidden = _hidden_from_output(output)
            source = self.source.to(device=hidden.device, dtype=hidden.dtype)
            if source.shape != hidden.shape:
                raise ValueError(
                    f"module patch requires matching shapes, got {source.shape} and {hidden.shape}"
                )
            patched = (1.0 - self.blend) * hidden + self.blend * source
            return _replace_hidden(output, patched)

        handle = self.module.register_forward_hook(hook)
        try:
            yield
        finally:
            handle.remove()


@dataclass
class AttentionHeadAblation:
    """Zero one query head immediately before the attention output projection."""

    block: torch.nn.Module
    head: int
    scale: float = 1.0

    @contextlib.contextmanager
    def apply(self) -> Iterator[None]:
        if not 0.0 <= self.scale <= 1.0:
            raise ValueError("scale must lie in [0, 1]")
        head_count, head_width = get_attention_head_layout(self.block)
        if not 0 <= self.head < head_count:
            raise ValueError(f"head must lie in [0, {head_count})")
        projection = get_attention_output_projection(self.block)
        start = self.head * head_width
        end = start + head_width

        def pre_hook(module, inputs):
            if not inputs or not isinstance(inputs[0], torch.Tensor):
                raise TypeError("attention output projection input is not a tensor")
            hidden = inputs[0]
            updated = hidden.clone()
            updated[..., start:end] *= 1.0 - self.scale
            return (updated, *inputs[1:])

        handle = projection.register_forward_pre_hook(pre_hook)
        try:
            yield
        finally:
            handle.remove()


def trajectory_similarity(
    first: Sequence[np.ndarray],
    second: Sequence[np.ndarray],
) -> dict[str, float]:
    """Compare two layerwise displacement trajectories."""

    a = np.asarray(first, dtype=np.float64)
    b = np.asarray(second, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 2:
        raise ValueError("trajectories must have the same [layers, width] shape")
    row_denominator = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    layer_cosines = np.divide(
        np.einsum("ij,ij->i", a, b),
        row_denominator,
        out=np.zeros(a.shape[0], dtype=np.float64),
        where=row_denominator > 0,
    )
    flat_denominator = np.linalg.norm(a) * np.linalg.norm(b)
    route_cosine = float(np.vdot(a, b) / flat_denominator) if flat_denominator > 0 else 0.0
    return {
        "mean_layer_cosine": float(layer_cosines.mean()) if layer_cosines.size else float("nan"),
        "minimum_layer_cosine": float(layer_cosines.min()) if layer_cosines.size else float("nan"),
        "route_cosine": route_cosine,
        "mean_layer_distance": float(np.linalg.norm(a - b, axis=1).mean())
        if a.shape[0]
        else float("nan"),
    }
