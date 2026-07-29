"""Reachable-state augmentation for monitor training."""

from __future__ import annotations

import numpy as np


def augment_with_reachable_states(
    natural_states: np.ndarray,
    natural_labels: np.ndarray,
    reachable_states: np.ndarray,
    reachable_labels: np.ndarray,
    *,
    reachable_weight: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    natural = np.asarray(natural_states, dtype=np.float32)
    reached = np.asarray(reachable_states, dtype=np.float32)
    labels = np.concatenate([natural_labels, reachable_labels])
    if natural.ndim != 2 or reached.ndim != 2 or natural.shape[1] != reached.shape[1]:
        raise ValueError("natural and reachable states must have the same feature width")
    if natural.shape[0] != len(natural_labels) or reached.shape[0] != len(reachable_labels):
        raise ValueError("state and label counts do not match")
    weights = np.concatenate(
        [
            np.ones(natural.shape[0], dtype=np.float32),
            np.full(reached.shape[0], reachable_weight, dtype=np.float32),
        ]
    )
    return np.concatenate([natural, reached]), labels, weights
