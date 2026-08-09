"""Active discovery of behavior-preserving controllability manifolds."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from llm_controllability.constraints import BehaviorGate, BehaviorRecord
from llm_controllability.constraints.verification import (
    TransformerSentenceEmbedder,
    verify_output,
)
from llm_controllability.controllability.types import (
    ControlChannel,
    InterventionMetadata,
    StateSample,
)
from llm_controllability.interventions import ActivationAddition, NoIntervention
from llm_controllability.models.adapters import ensure_padding_token
from llm_controllability.reachability.collection import _quality_score, run_and_capture
from llm_controllability.reachability.geometry import spectral_metrics


def _stable_seed(*parts: object, seed: int) -> int:
    payload = "\x1f".join(str(part) for part in (*parts, seed)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


@dataclass(frozen=True)
class CMapConfig:
    """Finite-budget controls for Controllability Manifold Discovery."""

    layer: int
    direction_budget: int = 8
    query_budget: int = 512
    candidate_pool_size: int = 64
    initial_strength: float = 0.5
    maximum_strength: float = 16.0
    expansion_factor: float = 2.0
    boundary_steps: int = 3
    required_preservation_rate: float = 0.75
    rank_tolerance: float = 1e-4
    minimum_novelty: float = 0.05
    stagnation_patience: int = 2
    validation_examples: int = 8
    test_examples: int = 16
    token_scope: str = "last"
    seed: int = 0

    def __post_init__(self) -> None:
        if self.layer < 0:
            raise ValueError("C-MAP layer must be nonnegative")
        for name in (
            "direction_budget",
            "query_budget",
            "candidate_pool_size",
            "validation_examples",
            "test_examples",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"C-MAP {name} must be positive")
        if self.initial_strength <= 0:
            raise ValueError("C-MAP initial_strength must be positive")
        if self.maximum_strength < self.initial_strength:
            raise ValueError("C-MAP maximum_strength must cover initial_strength")
        if self.expansion_factor <= 1:
            raise ValueError("C-MAP expansion_factor must exceed one")
        if self.boundary_steps < 0:
            raise ValueError("C-MAP boundary_steps must be nonnegative")
        if not 0 < self.required_preservation_rate <= 1:
            raise ValueError("C-MAP required_preservation_rate must lie in (0, 1]")
        if self.rank_tolerance <= 0:
            raise ValueError("C-MAP rank_tolerance must be positive")
        if not 0 <= self.minimum_novelty <= 1:
            raise ValueError("C-MAP minimum_novelty must lie in [0, 1]")
        if self.stagnation_patience <= 0:
            raise ValueError("C-MAP stagnation_patience must be positive")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> CMapConfig:
        ignored = {"enabled", "seeds"}
        return cls(**{key: value for key, value in values.items() if key not in ignored})


@dataclass(frozen=True)
class BoundaryTrial:
    strength: float
    preservation_rate: float


@dataclass(frozen=True)
class BoundaryEstimate:
    lower: float
    upper: float
    status: str
    trials: tuple[BoundaryTrial, ...]


def adaptive_control_boundary(
    evaluate: Callable[[float], float],
    *,
    initial_strength: float,
    maximum_strength: float,
    expansion_factor: float,
    boundary_steps: int,
    required_preservation_rate: float,
    max_trials: int,
) -> BoundaryEstimate:
    """Bracket the largest accepted dose using expansion followed by bisection."""

    if max_trials <= 0:
        return BoundaryEstimate(0.0, float("nan"), "budget_exhausted", ())
    trials: list[BoundaryTrial] = []

    def query(strength: float) -> bool:
        rate = float(evaluate(strength))
        if not np.isfinite(rate) or not 0 <= rate <= 1:
            raise ValueError("boundary evaluator must return a finite rate in [0, 1]")
        trials.append(BoundaryTrial(strength, rate))
        return rate >= required_preservation_rate

    lower = 0.0
    upper = float("nan")
    strength = min(initial_strength, maximum_strength)
    while len(trials) < max_trials:
        if query(strength):
            lower = strength
            if strength >= maximum_strength:
                return BoundaryEstimate(
                    lower,
                    float("nan"),
                    "right_censored",
                    tuple(trials),
                )
            strength = min(maximum_strength, strength * expansion_factor)
            continue
        upper = strength
        break

    if not np.isfinite(upper):
        return BoundaryEstimate(lower, upper, "budget_limited", tuple(trials))

    for _ in range(boundary_steps):
        if len(trials) >= max_trials:
            return BoundaryEstimate(lower, upper, "budget_limited", tuple(trials))
        midpoint = 0.5 * (lower + upper)
        if query(midpoint):
            lower = midpoint
        else:
            upper = midpoint
    return BoundaryEstimate(lower, upper, "bracketed", tuple(trials))


class ActiveTangentExplorer:
    """Select residual directions outside the currently observed tangent span."""

    def __init__(self, width: int, config: CMapConfig, *, seed: int) -> None:
        if width <= 0:
            raise ValueError("state width must be positive")
        self.width = width
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.accepted_displacements: list[np.ndarray] = []
        self.attempted_directions: list[np.ndarray] = []
        self.stagnant_rounds = 0
        self._basis_cache = np.empty((0, self.width), dtype=np.float64)
        self._basis_dirty = False

    def tangent_basis(self) -> np.ndarray:
        if not self._basis_dirty:
            return self._basis_cache
        if not self.accepted_displacements:
            self._basis_cache = np.empty((0, self.width), dtype=np.float64)
            self._basis_dirty = False
            return self._basis_cache
        matrix = np.stack(self.accepted_displacements).astype(np.float64, copy=False)
        _, singular_values, right = np.linalg.svd(matrix, full_matrices=False)
        if not singular_values.size or singular_values[0] <= 1e-12:
            self._basis_cache = np.empty((0, self.width), dtype=np.float64)
            self._basis_dirty = False
            return self._basis_cache
        rank = int(
            np.count_nonzero(
                singular_values > singular_values[0] * self.config.rank_tolerance
            )
        )
        self._basis_cache = right[:rank]
        self._basis_dirty = False
        return self._basis_cache

    @property
    def rank(self) -> int:
        return int(self.tangent_basis().shape[0])

    def propose(self) -> tuple[np.ndarray, float, float] | None:
        basis = self.tangent_basis()
        candidates = self.rng.normal(
            size=(self.config.candidate_pool_size, self.width)
        )
        candidates /= np.linalg.norm(candidates, axis=1, keepdims=True).clip(1e-12)
        residuals = candidates.copy()
        if basis.size:
            residuals -= (residuals @ basis.T) @ basis
        residual_norms = np.linalg.norm(residuals, axis=1)
        novelty = residual_norms.copy()
        if self.attempted_directions:
            attempted = np.stack(self.attempted_directions)
            separation = 1.0 - np.max(np.abs(candidates @ attempted.T), axis=1)
        else:
            separation = np.ones(candidates.shape[0], dtype=np.float64)
        scores = 0.75 * novelty + 0.25 * separation
        index = int(np.argmax(scores))
        if novelty[index] < self.config.minimum_novelty:
            return None
        direction = residuals[index] / max(residual_norms[index], 1e-12)
        self.attempted_directions.append(direction)
        return direction.astype(np.float32), float(novelty[index]), float(scores[index])

    def observe(self, displacement: np.ndarray, *, preserved: bool) -> None:
        value = np.asarray(displacement, dtype=np.float64).reshape(-1)
        if value.shape[0] != self.width:
            raise ValueError("observed displacement width does not match explorer width")
        if preserved and np.linalg.norm(value) > 1e-10:
            self.accepted_displacements.append(value)
            self._basis_dirty = True

    def finish_direction(self, previous_rank: int) -> None:
        if self.rank > previous_rank:
            self.stagnant_rounds = 0
        else:
            self.stagnant_rounds += 1

    @property
    def converged(self) -> bool:
        return self.stagnant_rounds >= self.config.stagnation_patience


@dataclass
class CMapResult:
    samples: list[StateSample] = field(default_factory=list)
    query_rows: list[dict[str, Any]] = field(default_factory=list)
    direction_rows: list[dict[str, Any]] = field(default_factory=list)
    summary_rows: list[dict[str, Any]] = field(default_factory=list)
    directions: dict[str, np.ndarray] = field(default_factory=dict)


@dataclass
class _ReferenceExecution:
    example: Mapping[str, Any]
    example_id: str
    prompt: str
    output: str
    state: np.ndarray
    record: BehaviorRecord
    sample: StateSample
    eligible: bool


def _record(
    model: torch.nn.Module,
    tokenizer,
    prompt: str,
    output: str,
    example: Mapping[str, Any],
    semantic_embedder: TransformerSentenceEmbedder | None,
) -> BehaviorRecord:
    task_score, task_correct = verify_output(output, example)
    if semantic_embedder is not None:
        prompt_embedding, output_embedding = semantic_embedder.encode([prompt, output])
    else:
        prompt_embedding = None
        output_embedding = None
    return BehaviorRecord(
        output=output,
        task_score=task_score,
        task_correct=task_correct,
        quality_score=_quality_score(model, tokenizer, prompt, output),
        embedding=output_embedding,
        metadata={"prompt_embedding": prompt_embedding},
    )


def _tags(example: Mapping[str, Any], *, role: str) -> dict[str, str]:
    tags = {
        key: str(example[key])
        for key in ("source", "category", "split", "concept", "pair_id")
        if example.get(key) is not None
    }
    tags.update({"algorithm": "cmap", "cmap_role": role})
    return tags


def _metrics(
    record: BehaviorRecord,
    example: Mapping[str, Any],
    state: np.ndarray,
    target_direction: np.ndarray | None,
    execution: Mapping[str, float],
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    if record.task_score is not None:
        metrics["task_score"] = float(record.task_score)
    if record.quality_score is not None:
        metrics["quality_score"] = float(record.quality_score)
    if "monitor_label" in example:
        metrics["monitor_label"] = float(example["monitor_label"])
    if target_direction is not None:
        direction = np.asarray(target_direction, dtype=np.float64).reshape(-1)
        direction /= max(np.linalg.norm(direction), 1e-12)
        metrics["target_projection"] = float(np.dot(state, direction))
    metrics.update(
        {key: float(value) for key, value in execution.items() if key != "control_cost"}
    )
    return metrics


def _prepare_references(
    model: torch.nn.Module,
    tokenizer,
    examples: Sequence[Mapping[str, Any]],
    *,
    model_name: str,
    layer: int,
    behavior_gate: BehaviorGate,
    generation: Mapping[str, Any],
    max_length: int,
    semantic_embedder: TransformerSentenceEmbedder | None,
    target_direction: np.ndarray | None,
    role: str,
    seed: int,
) -> list[_ReferenceExecution]:
    references = []
    baseline = NoIntervention()
    for index, example in enumerate(examples):
        example_id = str(example.get("id", index))
        prompt = str(example["prompt"])
        prepared, output, states, _, execution = run_and_capture(
            model,
            tokenizer,
            prompt,
            baseline,
            [layer],
            generation,
            pooling="last",
            max_length=max_length,
            store_token_states=False,
        )
        record = _record(
            model,
            tokenizer,
            prepared,
            output,
            example,
            semantic_embedder,
        )
        eligible, verdicts = behavior_gate.evaluate(record, record, control_cost=0.0)
        metadata = InterventionMetadata(
            name=f"cmap_{role}_baseline",
            channel=ControlChannel.BASELINE,
            control_cost=0.0,
            parameters={"algorithm": "cmap", "role": role},
        )
        sample = StateSample(
            example_id=example_id,
            model_name=model_name,
            layer=layer,
            intervention=metadata,
            state=states[layer],
            prompt=prepared,
            output=output,
            behavior_preserved=eligible,
            constraint_results={name: value.passed for name, value in verdicts.items()},
            metrics=_metrics(record, example, states[layer], target_direction, execution),
            tags=_tags(example, role=role),
            seed=seed,
        )
        references.append(
            _ReferenceExecution(
                example=example,
                example_id=example_id,
                prompt=prompt,
                output=output,
                state=states[layer],
                record=record,
                sample=sample,
                eligible=eligible,
            )
        )
    return references


def discover_controllability_manifold(
    model: torch.nn.Module,
    tokenizer,
    *,
    model_name: str,
    validation_examples: Sequence[Mapping[str, Any]],
    test_examples: Sequence[Mapping[str, Any]],
    behavior_gate: BehaviorGate,
    generation: Mapping[str, Any],
    config: CMapConfig,
    max_length: int = 2048,
    semantic_embedder: TransformerSentenceEmbedder | None = None,
    target_direction: np.ndarray | None = None,
) -> CMapResult:
    """Actively map a local tangent space, then test it on held-out examples."""

    validation_examples = list(validation_examples[: config.validation_examples])
    test_examples = list(test_examples[: config.test_examples])
    if not validation_examples or not test_examples:
        raise ValueError("C-MAP requires nonempty validation and test splits")
    tokenizer, _ = ensure_padding_token(tokenizer)

    validation = _prepare_references(
        model,
        tokenizer,
        validation_examples,
        model_name=model_name,
        layer=config.layer,
        behavior_gate=behavior_gate,
        generation=generation,
        max_length=max_length,
        semantic_embedder=semantic_embedder,
        target_direction=target_direction,
        role="validation",
        seed=config.seed,
    )
    holdout = _prepare_references(
        model,
        tokenizer,
        test_examples,
        model_name=model_name,
        layer=config.layer,
        behavior_gate=behavior_gate,
        generation=generation,
        max_length=max_length,
        semantic_embedder=semantic_embedder,
        target_direction=target_direction,
        role="test",
        seed=config.seed,
    )
    eligible_validation = [reference for reference in validation if reference.eligible]
    eligible_holdout = [reference for reference in holdout if reference.eligible]
    result = CMapResult(samples=[*(item.sample for item in validation), *(item.sample for item in holdout)])
    if not eligible_validation:
        result.summary_rows.append(
            {
                "model_name": model_name,
                "layer": config.layer,
                "seed": config.seed,
                "role": "validation",
                "n_examples": len(validation),
                "n_eligible_examples": 0,
                "n_queries": 0,
                "n_directions": 0,
                "preservation_rate": float("nan"),
                "effective_rank": 0.0,
                "participation_ratio": 0.0,
                "maximum_state_displacement": 0.0,
                "mean_boundary_lower": 0.0,
                "stopping_reason": "no_eligible_validation_examples",
            }
        )
        return result

    explorer = ActiveTangentExplorer(
        eligible_validation[0].state.shape[0],
        config,
        seed=_stable_seed(model_name, config.layer, seed=config.seed),
    )
    discovery_queries = 0
    accepted_strengths: list[tuple[int, float, np.ndarray, float]] = []
    stopping_reason = "direction_budget"

    for direction_index in range(config.direction_budget):
        if config.query_budget - discovery_queries < len(eligible_validation):
            stopping_reason = "query_budget"
            break
        if explorer.converged:
            stopping_reason = "rank_stagnation"
            break
        proposal = explorer.propose()
        if proposal is None:
            stopping_reason = "tangent_span_saturated"
            break
        direction, novelty, acquisition_score = proposal
        vector_key = f"seed_{config.seed:04d}_direction_{direction_index:04d}"
        result.directions[vector_key] = direction
        previous_rank = explorer.rank

        def evaluate(
            strength: float,
            *,
            direction_index: int = direction_index,
            direction: np.ndarray = direction,
            novelty: float = novelty,
            acquisition_score: float = acquisition_score,
            vector_key: str = vector_key,
        ) -> float:
            nonlocal discovery_queries
            preserved_count = 0
            intervention = ActivationAddition(
                name=f"cmap_s{config.seed:04d}_d{direction_index:04d}_a{strength:g}",
                layer=config.layer,
                direction=torch.from_numpy(direction),
                strength=strength,
                token_scope=config.token_scope,
            )
            control_cost = intervention.control_cost(tokenizer)
            parameters = {
                **intervention.parameters(),
                "algorithm": "cmap",
                "direction_index": direction_index,
                "vector_key": vector_key,
            }
            for reference in eligible_validation:
                prepared, output, states, _, execution = run_and_capture(
                    model,
                    tokenizer,
                    reference.prompt,
                    intervention,
                    [config.layer],
                    generation,
                    pooling="last",
                    max_length=max_length,
                    store_token_states=False,
                )
                record = _record(
                    model,
                    tokenizer,
                    prepared,
                    output,
                    reference.example,
                    semantic_embedder,
                )
                preserved, verdicts = behavior_gate.evaluate(
                    reference.record,
                    record,
                    control_cost=control_cost,
                )
                state = states[config.layer]
                displacement = state - reference.state
                displacement_norm = float(np.linalg.norm(displacement))
                explorer.observe(displacement, preserved=preserved)
                preserved_count += int(preserved)
                discovery_queries += 1
                metrics = _metrics(
                    record,
                    reference.example,
                    state,
                    target_direction,
                    execution,
                )
                metrics.update(
                    {
                        "cmap_direction_index": float(direction_index),
                        "cmap_query_index": float(discovery_queries),
                        "cmap_proposal_uncertainty": novelty,
                        "cmap_acquisition_score": acquisition_score,
                        "state_displacement": displacement_norm,
                    }
                )
                sample = StateSample(
                    example_id=reference.example_id,
                    model_name=model_name,
                    layer=config.layer,
                    intervention=InterventionMetadata(
                        name=intervention.name,
                        channel=ControlChannel.ACTIVATION,
                        control_cost=control_cost,
                        parameters=parameters,
                    ),
                    state=state,
                    prompt=prepared,
                    output=output,
                    behavior_preserved=preserved,
                    constraint_results={
                        name: value.passed for name, value in verdicts.items()
                    },
                    metrics=metrics,
                    tags=_tags(reference.example, role="validation"),
                    seed=config.seed,
                )
                result.samples.append(sample)
                result.query_rows.append(
                    {
                        "model_name": model_name,
                        "layer": config.layer,
                        "seed": config.seed,
                        "role": "validation",
                        "example_id": reference.example_id,
                        "direction_index": direction_index,
                        "query_index": discovery_queries,
                        "strength": strength,
                        "proposal_uncertainty": novelty,
                        "acquisition_score": acquisition_score,
                        "behavior_preserved": int(preserved),
                        "state_displacement": displacement_norm,
                        "binding_constraints": ";".join(
                            sorted(name for name, value in verdicts.items() if not value.passed)
                        ),
                    }
                )
            return preserved_count / len(eligible_validation)

        remaining = config.query_budget - discovery_queries
        max_trials = remaining // len(eligible_validation)
        boundary = adaptive_control_boundary(
            evaluate,
            initial_strength=config.initial_strength,
            maximum_strength=config.maximum_strength,
            expansion_factor=config.expansion_factor,
            boundary_steps=config.boundary_steps,
            required_preservation_rate=config.required_preservation_rate,
            max_trials=max_trials,
        )
        explorer.finish_direction(previous_rank)
        result.direction_rows.append(
            {
                "model_name": model_name,
                "layer": config.layer,
                "seed": config.seed,
                "direction_index": direction_index,
                "vector_key": vector_key,
                "proposal_uncertainty": novelty,
                "acquisition_score": acquisition_score,
                "n_trials": len(boundary.trials),
                "n_queries": len(boundary.trials) * len(eligible_validation),
                "lower_control_bound": boundary.lower,
                "upper_control_bound": boundary.upper,
                "boundary_status": boundary.status,
                "rank_before": previous_rank,
                "rank_after": explorer.rank,
            }
        )
        if boundary.lower > 0:
            accepted_strengths.append(
                (direction_index, boundary.lower, direction, novelty)
            )
        if discovery_queries >= config.query_budget:
            stopping_reason = "query_budget"
            break

    holdout_queries = 0
    for direction_index, strength, direction, novelty in accepted_strengths:
        vector_key = f"seed_{config.seed:04d}_direction_{direction_index:04d}"
        intervention = ActivationAddition(
            name=f"cmap_holdout_s{config.seed:04d}_d{direction_index:04d}_a{strength:g}",
            layer=config.layer,
            direction=torch.from_numpy(direction),
            strength=strength,
            token_scope=config.token_scope,
        )
        control_cost = intervention.control_cost(tokenizer)
        parameters = {
            **intervention.parameters(),
            "algorithm": "cmap",
            "direction_index": direction_index,
            "vector_key": vector_key,
        }
        for reference in eligible_holdout:
            prepared, output, states, _, execution = run_and_capture(
                model,
                tokenizer,
                reference.prompt,
                intervention,
                [config.layer],
                generation,
                pooling="last",
                max_length=max_length,
                store_token_states=False,
            )
            record = _record(
                model,
                tokenizer,
                prepared,
                output,
                reference.example,
                semantic_embedder,
            )
            preserved, verdicts = behavior_gate.evaluate(
                reference.record,
                record,
                control_cost=control_cost,
            )
            state = states[config.layer]
            displacement = state - reference.state
            displacement_norm = float(np.linalg.norm(displacement))
            holdout_queries += 1
            metrics = _metrics(
                record,
                reference.example,
                state,
                target_direction,
                execution,
            )
            metrics.update(
                {
                    "cmap_direction_index": float(direction_index),
                    "cmap_query_index": float(holdout_queries),
                    "cmap_proposal_uncertainty": novelty,
                    "state_displacement": displacement_norm,
                }
            )
            result.samples.append(
                StateSample(
                    example_id=reference.example_id,
                    model_name=model_name,
                    layer=config.layer,
                    intervention=InterventionMetadata(
                        name=intervention.name,
                        channel=ControlChannel.ACTIVATION,
                        control_cost=control_cost,
                        parameters=parameters,
                    ),
                    state=state,
                    prompt=prepared,
                    output=output,
                    behavior_preserved=preserved,
                    constraint_results={
                        name: value.passed for name, value in verdicts.items()
                    },
                    metrics=metrics,
                    tags=_tags(reference.example, role="test"),
                    seed=config.seed,
                )
            )
            result.query_rows.append(
                {
                    "model_name": model_name,
                    "layer": config.layer,
                    "seed": config.seed,
                    "role": "test",
                    "example_id": reference.example_id,
                    "direction_index": direction_index,
                    "query_index": holdout_queries,
                    "strength": strength,
                    "proposal_uncertainty": novelty,
                    "acquisition_score": float("nan"),
                    "behavior_preserved": int(preserved),
                    "state_displacement": displacement_norm,
                    "binding_constraints": ";".join(
                        sorted(name for name, value in verdicts.items() if not value.passed)
                    ),
                }
            )

    for role, references in (("validation", validation), ("test", holdout)):
        role_queries = [row for row in result.query_rows if row["role"] == role]
        role_samples = [
            sample
            for sample in result.samples
            if sample.tags.get("cmap_role") == role
            and sample.intervention.channel is not ControlChannel.BASELINE
            and sample.behavior_preserved
        ]
        baseline_index = {reference.example_id: reference.state for reference in references}
        displacements = [
            sample.state - baseline_index[sample.example_id]
            for sample in role_samples
            if sample.example_id in baseline_index
        ]
        matrix = (
            np.stack(displacements)
            if displacements
            else np.empty((0, explorer.width), dtype=np.float64)
        )
        norms = np.linalg.norm(matrix, axis=1) if matrix.size else np.empty(0)
        spectrum = spectral_metrics(matrix, center=False)
        result.summary_rows.append(
            {
                "model_name": model_name,
                "layer": config.layer,
                "seed": config.seed,
                "role": role,
                "n_examples": len(references),
                "n_eligible_examples": sum(reference.eligible for reference in references),
                "n_queries": len(role_queries),
                "n_directions": len(accepted_strengths),
                "preservation_rate": (
                    float(np.mean([row["behavior_preserved"] for row in role_queries]))
                    if role_queries
                    else float("nan")
                ),
                "effective_rank": spectrum["effective_rank"],
                "participation_ratio": spectrum["participation_ratio"],
                "maximum_state_displacement": float(norms.max()) if norms.size else 0.0,
                "mean_boundary_lower": (
                    float(np.mean([strength for _, strength, _, _ in accepted_strengths]))
                    if accepted_strengths
                    else 0.0
                ),
                "stopping_reason": stopping_reason if role == "validation" else "held_out_evaluation",
            }
        )
    return result
