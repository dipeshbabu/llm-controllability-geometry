"""Data contracts shared by interventions, reachability, and monitors."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class ControlChannel(str, Enum):
    """The interface through which an intervention enters the model."""

    BASELINE = "baseline"
    PROMPT = "prompt"
    ACTIVATION = "activation"
    HYBRID = "hybrid"
    RANDOM = "random"


@dataclass(frozen=True)
class InterventionMetadata:
    """Serializable description of one intervention setting."""

    name: str
    channel: ControlChannel
    control_cost: float
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.control_cost < 0 or not np.isfinite(self.control_cost):
            raise ValueError("control_cost must be finite and nonnegative")


@dataclass
class StateSample:
    """One layer state reached under a concrete intervention."""

    example_id: str
    model_name: str
    layer: int
    intervention: InterventionMetadata
    state: np.ndarray
    prompt: str
    output: str
    behavior_preserved: bool
    token_states: np.ndarray | None = None
    constraint_results: Mapping[str, bool] = field(default_factory=dict)
    metrics: Mapping[str, float] = field(default_factory=dict)
    tags: Mapping[str, str] = field(default_factory=dict)
    seed: int = 0

    def __post_init__(self) -> None:
        state = np.asarray(self.state, dtype=np.float32)
        if state.ndim != 1:
            raise ValueError("state must be a one-dimensional activation vector")
        if not np.isfinite(state).all():
            raise ValueError("state contains a nonfinite value")
        self.state = state
        if self.token_states is not None:
            token_states = np.asarray(self.token_states, dtype=np.float32)
            if token_states.ndim != 2 or token_states.shape[1] != state.shape[0]:
                raise ValueError("token_states must be shaped [tokens, state width]")
            if not np.isfinite(token_states).all():
                raise ValueError("token_states contain a nonfinite value")
            self.token_states = token_states

    def metadata_row(self, state_index: int) -> dict[str, Any]:
        return {
            "state_index": state_index,
            "example_id": self.example_id,
            "model_name": self.model_name,
            "layer": self.layer,
            "intervention": self.intervention.name,
            "channel": self.intervention.channel.value,
            "control_cost": self.intervention.control_cost,
            "parameters": dict(self.intervention.parameters),
            "prompt": self.prompt,
            "output": self.output,
            "behavior_preserved": self.behavior_preserved,
            "constraint_results": dict(self.constraint_results),
            "metrics": dict(self.metrics),
            "tags": dict(self.tags),
            "seed": self.seed,
        }


@dataclass
class ReachableSet:
    """A behavior-preserving collection of states for one layer and example."""

    example_id: str
    model_name: str
    layer: int
    samples: Sequence[StateSample]

    def __post_init__(self) -> None:
        if not self.samples:
            raise ValueError("a reachable set requires at least one sample")
        dimensions = {sample.state.shape[0] for sample in self.samples}
        if len(dimensions) != 1:
            raise ValueError("all states in a reachable set must have the same dimension")
        for sample in self.samples:
            if sample.example_id != self.example_id:
                raise ValueError("sample example_id does not match reachable set")
            if sample.model_name != self.model_name:
                raise ValueError("sample model_name does not match reachable set")
            if sample.layer != self.layer:
                raise ValueError("sample layer does not match reachable set")

    @property
    def preserved_samples(self) -> list[StateSample]:
        return [sample for sample in self.samples if sample.behavior_preserved]

    def matrix(self, *, preserved_only: bool = True) -> np.ndarray:
        samples = self.preserved_samples if preserved_only else list(self.samples)
        if not samples:
            width = self.samples[0].state.shape[0]
            return np.empty((0, width), dtype=np.float32)
        return np.stack([sample.state for sample in samples]).astype(np.float32, copy=False)

    def channels(self, *, preserved_only: bool = True) -> np.ndarray:
        samples = self.preserved_samples if preserved_only else list(self.samples)
        return np.asarray([sample.intervention.channel.value for sample in samples])
