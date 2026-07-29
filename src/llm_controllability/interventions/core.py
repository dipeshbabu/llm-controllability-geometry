"""Composable prompt and residual-stream interventions."""

from __future__ import annotations

import contextlib
import math
from abc import ABC
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

import torch

from llm_controllability.controllability.types import (
    ControlChannel,
    InterventionMetadata,
)
from llm_controllability.models.architecture import get_layers

TokenScope = Literal["all", "last"]


def _edit_distance(first: list[int], second: list[int]) -> int:
    """Levenshtein distance using one dynamic-programming row."""

    if len(first) < len(second):
        first, second = second, first
    previous = list(range(len(second) + 1))
    for row, left in enumerate(first, start=1):
        current = [row]
        for column, right in enumerate(second, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (left != right),
                )
            )
        previous = current
    return previous[-1]


def _replace_hidden(output: Any, hidden: torch.Tensor) -> Any:
    if isinstance(output, torch.Tensor):
        return hidden
    if isinstance(output, tuple):
        return (hidden, *output[1:])
    if isinstance(output, list):
        return [hidden, *output[1:]]
    raise TypeError(f"unsupported transformer block output type: {type(output)!r}")


def _hidden_from_output(output: Any) -> torch.Tensor:
    hidden = output[0] if isinstance(output, (tuple, list)) else output
    if not isinstance(hidden, torch.Tensor):
        raise TypeError("transformer block output does not contain a hidden-state tensor")
    return hidden


def _selected(hidden: torch.Tensor, scope: TokenScope) -> tuple[torch.Tensor, Callable[[torch.Tensor], torch.Tensor]]:
    if hidden.ndim != 3:
        raise ValueError("activation interventions expect hidden states shaped [batch, tokens, width]")
    if scope == "all":
        return hidden, lambda value: value
    if scope == "last":
        selected = hidden[:, -1:, :]

        def restore(value: torch.Tensor) -> torch.Tensor:
            result = hidden.clone()
            result[:, -1:, :] = value
            return result

        return selected, restore
    raise ValueError(f"unknown token scope: {scope}")


class Intervention(ABC):
    """Common lifecycle for all control channels."""

    name: str
    channel: ControlChannel

    def prepare_prompt(self, prompt: str) -> str:
        return prompt

    def reset(self) -> None:
        pass

    @contextlib.contextmanager
    def apply(self, model: torch.nn.Module) -> Iterator[None]:
        yield

    def control_cost(self, tokenizer=None) -> float:
        return 0.0

    def parameters(self) -> dict[str, Any]:
        return {}

    def diagnostics(self) -> dict[str, float]:
        return {}

    def metadata(self, tokenizer=None) -> InterventionMetadata:
        return InterventionMetadata(
            name=self.name,
            channel=self.channel,
            control_cost=self.control_cost(tokenizer),
            parameters=self.parameters(),
        )


@dataclass
class NoIntervention(Intervention):
    name: str = "baseline"
    channel: ControlChannel = field(default=ControlChannel.BASELINE, init=False)


@dataclass
class PromptIntervention(Intervention):
    """Append, prepend, or replace text through the external prompt channel."""

    name: str
    text: str | None = None
    mode: Literal["suffix", "prefix", "replace"] = "suffix"
    transform: Callable[[str], str] | None = None
    separator: str = "\n"
    channel: ControlChannel = field(default=ControlChannel.PROMPT, init=False)

    def __post_init__(self) -> None:
        if self.text is None and self.transform is None:
            raise ValueError("a prompt intervention requires text or a transform")

    def prepare_prompt(self, prompt: str) -> str:
        if self.transform is not None:
            return self.transform(prompt)
        assert self.text is not None
        if self.mode == "suffix":
            return f"{prompt}{self.separator}{self.text}"
        if self.mode == "prefix":
            return f"{self.text}{self.separator}{prompt}"
        if self.mode == "replace":
            return self.text
        raise ValueError(f"unknown prompt mode: {self.mode}")

    def control_cost(self, tokenizer=None) -> float:
        if self.text is None:
            return 1.0
        if tokenizer is None:
            return float(len(self.text.split()))
        return float(len(tokenizer.encode(self.text, add_special_tokens=False)))

    def parameters(self) -> dict[str, Any]:
        return {"mode": self.mode, "text": self.text}


@dataclass
class MappedPromptIntervention(Intervention):
    """Rewrite each source prompt with a predeclared behavior-equivalent prompt."""

    name: str
    rewrites: Mapping[str, str]
    rewrite_kind: str = "natural_paraphrase"
    strict: bool = True
    channel: ControlChannel = field(default=ControlChannel.PROMPT, init=False)
    _source_prompt: str | None = field(default=None, init=False, repr=False)
    _prepared_prompt: str | None = field(default=None, init=False, repr=False)

    def prepare_prompt(self, prompt: str) -> str:
        if prompt not in self.rewrites:
            if self.strict:
                raise KeyError(
                    f"{self.name!r} has no declared rewrite for the current prompt"
                )
            rewritten = prompt
        else:
            rewritten = self.rewrites[prompt]
        self._source_prompt = prompt
        self._prepared_prompt = rewritten
        return rewritten

    def control_cost(self, tokenizer=None) -> float:
        if self._source_prompt is None or self._prepared_prompt is None:
            return 0.0
        if tokenizer is None:
            return float(
                _edit_distance(
                    self._source_prompt.split(),
                    self._prepared_prompt.split(),
                )
            )
        source = tokenizer.encode(self._source_prompt, add_special_tokens=False)
        prepared = tokenizer.encode(self._prepared_prompt, add_special_tokens=False)
        return float(_edit_distance(source, prepared))

    def parameters(self) -> dict[str, Any]:
        return {
            "mode": "rewrite",
            "rewrite_kind": self.rewrite_kind,
            "source_prompt": self._source_prompt,
        }


@dataclass
class ActivationAddition(Intervention):
    """Add a fixed direction to a residual-stream layer."""

    name: str
    layer: int
    direction: torch.Tensor
    strength: float
    token_scope: TokenScope = "last"
    normalize_direction: bool = True
    channel: ControlChannel = ControlChannel.ACTIVATION

    def _vector(self, hidden: torch.Tensor) -> torch.Tensor:
        vector = self.direction.to(device=hidden.device, dtype=hidden.dtype).reshape(-1)
        if vector.shape[0] != hidden.shape[-1]:
            raise ValueError(
                f"direction width {vector.shape[0]} does not match hidden width {hidden.shape[-1]}"
            )
        if self.normalize_direction:
            vector = vector / vector.float().norm().clamp_min(1e-12).to(vector.dtype)
        return vector

    @contextlib.contextmanager
    def apply(self, model: torch.nn.Module) -> Iterator[None]:
        block = get_layers(model)[self.layer]

        def hook(module, inputs, output):
            hidden = _hidden_from_output(output)
            selected, restore = _selected(hidden, self.token_scope)
            updated = selected + self.strength * self._vector(hidden).view(1, 1, -1)
            return _replace_hidden(output, restore(updated))

        handle = block.register_forward_hook(hook)
        try:
            yield
        finally:
            handle.remove()

    def control_cost(self, tokenizer=None) -> float:
        norm = float(self.direction.detach().float().norm().cpu())
        if self.normalize_direction:
            norm = 1.0 if norm > 0 else 0.0
        return abs(self.strength) * norm

    def parameters(self) -> dict[str, Any]:
        return {
            "layer": self.layer,
            "strength": self.strength,
            "token_scope": self.token_scope,
            "normalize_direction": self.normalize_direction,
        }


@dataclass
class DirectionalAblation(Intervention):
    """Remove a configurable fraction of one residual-stream projection."""

    name: str
    layer: int
    direction: torch.Tensor
    fraction: float = 1.0
    token_scope: TokenScope = "last"
    channel: ControlChannel = field(default=ControlChannel.ACTIVATION, init=False)

    def __post_init__(self) -> None:
        if not 0.0 <= self.fraction <= 1.0:
            raise ValueError("fraction must lie in [0, 1]")

    @contextlib.contextmanager
    def apply(self, model: torch.nn.Module) -> Iterator[None]:
        block = get_layers(model)[self.layer]

        def hook(module, inputs, output):
            hidden = _hidden_from_output(output)
            selected, restore = _selected(hidden, self.token_scope)
            vector = self.direction.to(device=hidden.device, dtype=hidden.dtype).reshape(-1)
            vector = vector / vector.float().norm().clamp_min(1e-12).to(vector.dtype)
            projection = torch.einsum("btd,d->bt", selected, vector).unsqueeze(-1)
            updated = selected - self.fraction * projection * vector.view(1, 1, -1)
            return _replace_hidden(output, restore(updated))

        handle = block.register_forward_hook(hook)
        try:
            yield
        finally:
            handle.remove()

    def control_cost(self, tokenizer=None) -> float:
        return float(self.fraction)

    def parameters(self) -> dict[str, Any]:
        return {
            "layer": self.layer,
            "fraction": self.fraction,
            "token_scope": self.token_scope,
        }


@dataclass
class AdaptiveActivationController(Intervention):
    """PID feedback on a residual projection at one layer."""

    name: str
    layer: int
    direction: torch.Tensor
    setpoint: float
    kp: float = 1.0
    ki: float = 0.0
    kd: float = 0.0
    max_update: float | None = None
    token_scope: TokenScope = "last"
    channel: ControlChannel = field(default=ControlChannel.ACTIVATION, init=False)
    _integral: torch.Tensor | None = field(default=None, init=False, repr=False)
    _previous_error: torch.Tensor | None = field(default=None, init=False, repr=False)
    _energy_squared: float = field(default=0.0, init=False, repr=False)
    _tracking_errors: list[float] = field(default_factory=list, init=False, repr=False)
    _updates: list[float] = field(default_factory=list, init=False, repr=False)

    def reset(self) -> None:
        self._integral = None
        self._previous_error = None
        self._energy_squared = 0.0
        self._tracking_errors = []
        self._updates = []

    @contextlib.contextmanager
    def apply(self, model: torch.nn.Module) -> Iterator[None]:
        block = get_layers(model)[self.layer]

        def hook(module, inputs, output):
            hidden = _hidden_from_output(output)
            selected, restore = _selected(hidden, self.token_scope)
            vector = self.direction.to(device=hidden.device, dtype=hidden.dtype).reshape(-1)
            vector = vector / vector.float().norm().clamp_min(1e-12).to(vector.dtype)
            projection = torch.einsum("btd,d->bt", selected, vector)
            error = self.setpoint - projection

            if self._integral is None or self._integral.shape != error.shape:
                self._integral = torch.zeros_like(error)
                self._previous_error = torch.zeros_like(error)
            self._integral = self._integral + error
            derivative = error - self._previous_error
            update = self.kp * error + self.ki * self._integral + self.kd * derivative
            if self.max_update is not None:
                update = update.clamp(-self.max_update, self.max_update)
            self._previous_error = error.detach()
            self._energy_squared += float(update.detach().float().square().sum().cpu())
            self._tracking_errors.append(float(error.detach().float().abs().mean().cpu()))
            self._updates.append(float(update.detach().float().abs().mean().cpu()))
            updated = selected + update.unsqueeze(-1) * vector.view(1, 1, -1)
            return _replace_hidden(output, restore(updated))

        self.reset()
        handle = block.register_forward_hook(hook)
        try:
            yield
        finally:
            handle.remove()

    def control_cost(self, tokenizer=None) -> float:
        return math.sqrt(self._energy_squared)

    def parameters(self) -> dict[str, Any]:
        return {
            "layer": self.layer,
            "setpoint": self.setpoint,
            "kp": self.kp,
            "ki": self.ki,
            "kd": self.kd,
            "max_update": self.max_update,
            "token_scope": self.token_scope,
        }

    def diagnostics(self) -> dict[str, float]:
        if not self._tracking_errors:
            return {
                "generation_tracking_steps": 0.0,
                "generation_tracking_mae": float("nan"),
                "generation_tracking_max_error": float("nan"),
                "generation_tracking_final_error": float("nan"),
                "generation_mean_abs_update": float("nan"),
            }
        return {
            "generation_tracking_steps": float(len(self._tracking_errors)),
            "generation_tracking_mae": float(
                sum(self._tracking_errors) / len(self._tracking_errors)
            ),
            "generation_tracking_max_error": float(max(self._tracking_errors)),
            "generation_tracking_final_error": float(self._tracking_errors[-1]),
            "generation_mean_abs_update": float(
                sum(self._updates) / len(self._updates)
            ),
        }


@dataclass
class HybridIntervention(Intervention):
    """Combine one prompt intervention with one activation intervention."""

    name: str
    prompt: PromptIntervention
    activation: Intervention
    channel: ControlChannel = field(default=ControlChannel.HYBRID, init=False)

    def prepare_prompt(self, prompt: str) -> str:
        return self.prompt.prepare_prompt(prompt)

    def reset(self) -> None:
        self.activation.reset()

    @contextlib.contextmanager
    def apply(self, model: torch.nn.Module) -> Iterator[None]:
        with self.activation.apply(model):
            yield

    def control_cost(self, tokenizer=None) -> float:
        prompt_cost = self.prompt.control_cost(tokenizer)
        activation_cost = self.activation.control_cost(tokenizer)
        return math.sqrt(prompt_cost**2 + activation_cost**2)

    def parameters(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt.parameters(),
            "activation": self.activation.parameters(),
        }

    def diagnostics(self) -> dict[str, float]:
        return self.activation.diagnostics()
