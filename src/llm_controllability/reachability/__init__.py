"""Reachable-state collection, storage, and geometry."""

from llm_controllability.reachability.geometry import (
    budget_growth,
    connectivity,
    effective_rank,
    layerwise_propagation,
    principal_angle_rows,
    principal_angles,
    subspace_overlap,
    summarize_reachability,
    summarize_trajectories,
    target_orthogonal_decomposition,
    trajectory_curvature,
)
from llm_controllability.reachability.io import load_state_samples, save_state_samples
from llm_controllability.reachability.jacobians import (
    control_jacobian,
    jacobian_rank,
    local_controllability,
)

__all__ = [
    "budget_growth",
    "connectivity",
    "control_jacobian",
    "effective_rank",
    "jacobian_rank",
    "layerwise_propagation",
    "load_state_samples",
    "local_controllability",
    "principal_angle_rows",
    "principal_angles",
    "save_state_samples",
    "subspace_overlap",
    "summarize_reachability",
    "summarize_trajectories",
    "target_orthogonal_decomposition",
    "trajectory_curvature",
]
