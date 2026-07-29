"""Token pooling and monitor implementations used in invariance tests."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from llm_controllability.models.runtime import resolve_device

Pooling = Literal["last", "mean", "max"]


def pool_hidden_states(
    hidden: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    pooling: Pooling = "last",
) -> np.ndarray:
    values = np.asarray(hidden, dtype=np.float32)
    if values.ndim != 3:
        raise ValueError("hidden states must be shaped [samples, tokens, width]")
    if mask is None:
        mask = np.ones(values.shape[:2], dtype=bool)
    else:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != values.shape[:2]:
            raise ValueError("mask shape must match the first two hidden-state dimensions")
    if np.any(mask.sum(axis=1) == 0):
        raise ValueError("every sequence must contain at least one unmasked token")

    if pooling == "last":
        positions = np.broadcast_to(np.arange(mask.shape[1]), mask.shape)
        indices = np.where(mask, positions, -1).max(axis=1)
        return values[np.arange(values.shape[0]), indices]
    if pooling == "mean":
        weights = mask[..., None].astype(np.float32)
        return (values * weights).sum(axis=1) / weights.sum(axis=1)
    if pooling == "max":
        masked = np.where(mask[..., None], values, -np.inf)
        return masked.max(axis=1)
    raise ValueError(f"unknown pooling method: {pooling}")


@dataclass
class LinearMonitor:
    regularization: float = 1.0
    seed: int = 0
    model: object = field(default=None, init=False, repr=False)

    def fit(
        self,
        states: np.ndarray,
        labels: np.ndarray,
        *,
        sample_weight: np.ndarray | None = None,
    ) -> LinearMonitor:
        self.model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=self.regularization,
                max_iter=1000,
                random_state=self.seed,
            ),
        )
        self.model.fit(states, labels, logisticregression__sample_weight=sample_weight)
        return self

    def predict_proba(self, states: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("monitor has not been fitted")
        return self.model.predict_proba(states)[:, 1]

    def predict(self, states: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return self.predict_proba(states) >= threshold


@dataclass
class NonlinearMonitor:
    hidden_width: int = 128
    regularization: float = 1e-4
    seed: int = 0
    model: object = field(default=None, init=False, repr=False)

    def fit(self, states: np.ndarray, labels: np.ndarray) -> NonlinearMonitor:
        self.model = make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(self.hidden_width,),
                alpha=self.regularization,
                max_iter=500,
                early_stopping=True,
                random_state=self.seed,
            ),
        )
        self.model.fit(states, labels)
        return self

    def predict_proba(self, states: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("monitor has not been fitted")
        return self.model.predict_proba(states)[:, 1]

    def predict(self, states: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return self.predict_proba(states) >= threshold


@dataclass
class MultiLayerMonitor:
    """Linear monitor over standardized concatenated layer states."""

    layers: tuple[int, ...]
    regularization: float = 1.0
    seed: int = 0
    monitor: LinearMonitor = field(init=False)

    def __post_init__(self) -> None:
        if not self.layers:
            raise ValueError("at least one layer is required")
        self.monitor = LinearMonitor(self.regularization, self.seed)

    def _concatenate(self, states: Mapping[int, np.ndarray]) -> np.ndarray:
        missing = set(self.layers) - set(states)
        if missing:
            raise ValueError(f"missing layers: {sorted(missing)}")
        arrays = [np.asarray(states[layer], dtype=np.float32) for layer in self.layers]
        counts = {array.shape[0] for array in arrays}
        if len(counts) != 1:
            raise ValueError("all layer matrices must have the same sample count")
        return np.concatenate(arrays, axis=1)

    def fit(
        self,
        states: Mapping[int, np.ndarray],
        labels: np.ndarray,
        *,
        sample_weight: np.ndarray | None = None,
    ) -> MultiLayerMonitor:
        self.monitor.fit(
            self._concatenate(states),
            labels,
            sample_weight=sample_weight,
        )
        return self

    def predict_proba(self, states: Mapping[int, np.ndarray]) -> np.ndarray:
        return self.monitor.predict_proba(self._concatenate(states))

    def predict(self, states: Mapping[int, np.ndarray], threshold: float = 0.5) -> np.ndarray:
        return self.predict_proba(states) >= threshold


class _AttentionPoolClassifier(torch.nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.query = torch.nn.Parameter(torch.zeros(width))
        self.classifier = torch.nn.Linear(width, 1)
        torch.nn.init.normal_(self.query, std=width**-0.5)

    def forward(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        logits = torch.einsum("ntd,d->nt", hidden, self.query)
        logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
        weights = torch.softmax(logits, dim=1)
        pooled = torch.einsum("nt,ntd->nd", weights, hidden)
        return self.classifier(pooled).squeeze(-1)


@dataclass
class AttentionPooledMonitor:
    """Learned token pooling followed by a binary linear readout."""

    learning_rate: float = 1e-2
    weight_decay: float = 1e-4
    epochs: int = 50
    batch_size: int = 64
    seed: int = 0
    device: str = "auto"
    model: _AttentionPoolClassifier | None = field(default=None, init=False, repr=False)

    def fit(
        self,
        hidden: np.ndarray,
        labels: np.ndarray,
        *,
        mask: np.ndarray | None = None,
        sample_weight: np.ndarray | None = None,
    ) -> AttentionPooledMonitor:
        values = torch.as_tensor(hidden, dtype=torch.float32)
        targets = torch.as_tensor(labels, dtype=torch.float32)
        if values.ndim != 3:
            raise ValueError("hidden states must be shaped [samples, tokens, width]")
        if mask is None:
            token_mask = torch.ones(values.shape[:2], dtype=torch.bool)
        else:
            token_mask = torch.as_tensor(mask, dtype=torch.bool)
        weights = (
            torch.ones_like(targets)
            if sample_weight is None
            else torch.as_tensor(sample_weight, dtype=torch.float32)
        )
        if weights.shape != targets.shape:
            raise ValueError("sample weights must match labels")
        torch.manual_seed(self.seed)
        self.device = str(resolve_device(self.device))
        self.model = _AttentionPoolClassifier(values.shape[-1]).to(self.device)
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        for _ in range(self.epochs):
            order = torch.randperm(values.shape[0])
            for start in range(0, len(order), self.batch_size):
                index = order[start : start + self.batch_size]
                batch_values = values[index].to(self.device)
                batch_mask = token_mask[index].to(self.device)
                batch_targets = targets[index].to(self.device)
                batch_weights = weights[index].to(self.device)
                optimizer.zero_grad()
                logits = self.model(batch_values, batch_mask)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits,
                    batch_targets,
                    weight=batch_weights,
                )
                loss.backward()
                optimizer.step()
        return self

    @torch.no_grad()
    def predict_proba(
        self,
        hidden: np.ndarray,
        *,
        mask: np.ndarray | None = None,
    ) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("monitor has not been fitted")
        values = torch.as_tensor(hidden, dtype=torch.float32)
        if mask is None:
            token_mask = torch.ones(values.shape[:2], dtype=torch.bool)
        else:
            token_mask = torch.as_tensor(mask, dtype=torch.bool)
        scores = []
        for start in range(0, values.shape[0], self.batch_size):
            batch = values[start : start + self.batch_size].to(self.device)
            batch_mask = token_mask[start : start + self.batch_size].to(self.device)
            scores.append(torch.sigmoid(self.model(batch, batch_mask)).cpu())
        return torch.cat(scores).numpy()

    def predict(
        self,
        hidden: np.ndarray,
        *,
        mask: np.ndarray | None = None,
        threshold: float = 0.5,
    ) -> np.ndarray:
        return self.predict_proba(hidden, mask=mask) >= threshold


@dataclass
class MahalanobisOODMonitor:
    """Latent out-of-distribution score using shrinkage covariance."""

    shrinkage: float = 1e-3
    mean_: np.ndarray | None = field(default=None, init=False, repr=False)
    precision_: np.ndarray | None = field(default=None, init=False, repr=False)

    def fit(self, states: np.ndarray) -> MahalanobisOODMonitor:
        values = np.asarray(states, dtype=np.float64)
        if values.ndim != 2 or values.shape[0] < 2:
            raise ValueError("OOD fitting requires at least two state vectors")
        self.mean_ = values.mean(axis=0)
        centered = values - self.mean_
        covariance = centered.T @ centered / max(values.shape[0] - 1, 1)
        scale = float(np.trace(covariance) / covariance.shape[0])
        covariance += self.shrinkage * max(scale, 1e-12) * np.eye(covariance.shape[0])
        self.precision_ = np.linalg.pinv(covariance)
        return self

    def score(self, states: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.precision_ is None:
            raise RuntimeError("monitor has not been fitted")
        centered = np.asarray(states, dtype=np.float64) - self.mean_
        squared = np.einsum("ni,ij,nj->n", centered, self.precision_, centered)
        return np.sqrt(np.maximum(squared, 0.0))
