"""Behavior constraints evaluated before a state enters a reachable set."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np


@dataclass(frozen=True)
class BehaviorRecord:
    """Observable behavior associated with one model execution."""

    output: str
    task_score: float | None = None
    task_correct: bool | None = None
    quality_score: float | None = None
    embedding: np.ndarray | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ConstraintResult:
    name: str
    passed: bool
    value: float | bool | None
    threshold: float | bool | None
    details: str = ""


class BehaviorConstraint(Protocol):
    name: str

    def evaluate(
        self,
        reference: BehaviorRecord,
        candidate: BehaviorRecord,
        *,
        control_cost: float,
    ) -> ConstraintResult:
        ...


@dataclass(frozen=True)
class TaskPreservationConstraint:
    """Require verified correctness and bound any drop from the reference."""

    minimum_score: float | None = None
    maximum_drop: float | None = 0.0
    require_correct: bool = True
    require_reference_correct: bool = True
    name: str = "task"

    def evaluate(
        self,
        reference: BehaviorRecord,
        candidate: BehaviorRecord,
        *,
        control_cost: float,
    ) -> ConstraintResult:
        checks: list[bool] = []
        details: list[str] = []
        value = candidate.task_score

        if self.require_correct:
            if self.require_reference_correct:
                if reference.task_correct is None:
                    checks.append(False)
                    details.append("reference correctness is unavailable")
                else:
                    checks.append(reference.task_correct)
                    details.append(f"reference_correct={reference.task_correct}")
            if candidate.task_correct is None:
                checks.append(False)
                details.append("candidate correctness is unavailable")
            else:
                checks.append(candidate.task_correct)
                details.append(f"correct={candidate.task_correct}")

        if self.minimum_score is not None:
            if candidate.task_score is None:
                checks.append(False)
                details.append("candidate task score is unavailable")
            else:
                checks.append(candidate.task_score >= self.minimum_score)
                details.append(f"score={candidate.task_score:.6g}")

        if self.maximum_drop is not None and reference.task_score is not None:
            if candidate.task_score is None:
                checks.append(False)
            else:
                drop = reference.task_score - candidate.task_score
                checks.append(drop <= self.maximum_drop)
                details.append(f"drop={drop:.6g}")

        if not checks:
            checks.append(True)
            details.append("no task criterion configured")
        return ConstraintResult(
            name=self.name,
            passed=all(checks),
            value=value if value is not None else candidate.task_correct,
            threshold=self.minimum_score if self.minimum_score is not None else self.require_correct,
            details=", ".join(details),
        )


@dataclass(frozen=True)
class SemanticEquivalenceConstraint:
    minimum_similarity: float = 0.85
    name: str = "semantic_equivalence"

    def evaluate(
        self,
        reference: BehaviorRecord,
        candidate: BehaviorRecord,
        *,
        control_cost: float,
    ) -> ConstraintResult:
        if reference.embedding is None or candidate.embedding is None:
            return ConstraintResult(
                self.name,
                False,
                None,
                self.minimum_similarity,
                "reference or candidate embedding is unavailable",
            )
        a = np.asarray(reference.embedding, dtype=np.float64).reshape(-1)
        b = np.asarray(candidate.embedding, dtype=np.float64).reshape(-1)
        if a.shape != b.shape:
            raise ValueError("semantic embeddings must have the same shape")
        denominator = np.linalg.norm(a) * np.linalg.norm(b)
        similarity = float(np.dot(a, b) / denominator) if denominator > 0 else 0.0
        return ConstraintResult(
            self.name,
            similarity >= self.minimum_similarity,
            similarity,
            self.minimum_similarity,
        )


@dataclass(frozen=True)
class PromptSemanticEquivalenceConstraint:
    """Require the controlled prompt to retain the baseline prompt semantics."""

    minimum_similarity: float = 0.85
    metadata_key: str = "prompt_embedding"
    name: str = "prompt_semantic_equivalence"

    def evaluate(
        self,
        reference: BehaviorRecord,
        candidate: BehaviorRecord,
        *,
        control_cost: float,
    ) -> ConstraintResult:
        reference_value = reference.metadata.get(self.metadata_key)
        candidate_value = candidate.metadata.get(self.metadata_key)
        if reference_value is None or candidate_value is None:
            return ConstraintResult(
                self.name,
                False,
                None,
                self.minimum_similarity,
                "reference or candidate prompt embedding is unavailable",
            )
        a = np.asarray(reference_value, dtype=np.float64).reshape(-1)
        b = np.asarray(candidate_value, dtype=np.float64).reshape(-1)
        if a.shape != b.shape:
            raise ValueError("prompt semantic embeddings must have the same shape")
        denominator = np.linalg.norm(a) * np.linalg.norm(b)
        similarity = float(np.dot(a, b) / denominator) if denominator > 0 else 0.0
        return ConstraintResult(
            self.name,
            similarity >= self.minimum_similarity,
            similarity,
            self.minimum_similarity,
        )


@dataclass(frozen=True)
class OutputQualityConstraint:
    maximum_drop: float = 0.5
    minimum_score: float | None = None
    name: str = "output_quality"

    def evaluate(
        self,
        reference: BehaviorRecord,
        candidate: BehaviorRecord,
        *,
        control_cost: float,
    ) -> ConstraintResult:
        if reference.quality_score is None or candidate.quality_score is None:
            return ConstraintResult(
                self.name,
                False,
                None,
                self.maximum_drop,
                "reference or candidate quality score is unavailable",
            )
        drop = reference.quality_score - candidate.quality_score
        passed = drop <= self.maximum_drop
        if self.minimum_score is not None:
            passed = passed and candidate.quality_score >= self.minimum_score
        return ConstraintResult(
            self.name,
            passed,
            candidate.quality_score,
            self.minimum_score if self.minimum_score is not None else self.maximum_drop,
            f"drop={drop:.6g}",
        )


@dataclass(frozen=True)
class BudgetConstraint:
    maximum_cost: float
    name: str = "control_budget"

    def evaluate(
        self,
        reference: BehaviorRecord,
        candidate: BehaviorRecord,
        *,
        control_cost: float,
    ) -> ConstraintResult:
        return ConstraintResult(
            self.name,
            control_cost <= self.maximum_cost,
            control_cost,
            self.maximum_cost,
        )


@dataclass(frozen=True)
class BehaviorGate:
    constraints: Sequence[BehaviorConstraint]

    def evaluate(
        self,
        reference: BehaviorRecord,
        candidate: BehaviorRecord,
        *,
        control_cost: float,
    ) -> tuple[bool, dict[str, ConstraintResult]]:
        results = {
            constraint.name: constraint.evaluate(
                reference,
                candidate,
                control_cost=control_cost,
            )
            for constraint in self.constraints
        }
        return all(result.passed for result in results.values()), results
