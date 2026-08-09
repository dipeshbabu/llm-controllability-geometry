"""Reachable-state collection, storage, and geometry."""

from llm_controllability.reachability.boundaries import (
    controllability_boundary_rows,
    detection_control_gap_rows,
    directed_accessibility_rows,
    representation_control_gap_rows,
    summarize_controllability_boundaries,
    summarize_directed_accessibility,
)
from llm_controllability.reachability.cmap import (
    ActiveTangentExplorer,
    BoundaryEstimate,
    BoundaryTrial,
    CMapConfig,
    CMapResult,
    adaptive_control_boundary,
    discover_controllability_manifold,
)
from llm_controllability.reachability.discovery import (
    boundary_survival_rows,
    controllability_atlas_rows,
    phase_transition_candidate_rows,
)
from llm_controllability.reachability.geometry import (
    budget_growth,
    connectivity,
    effective_rank,
    layerwise_propagation,
    principal_angle_rows,
    principal_angles,
    split_half_stability_rows,
    subspace_overlap,
    summarize_reachability,
    summarize_split_half_stability,
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
    "ActiveTangentExplorer",
    "BoundaryEstimate",
    "BoundaryTrial",
    "CMapConfig",
    "CMapResult",
    "adaptive_control_boundary",
    "boundary_survival_rows",
    "budget_growth",
    "connectivity",
    "control_jacobian",
    "controllability_atlas_rows",
    "controllability_boundary_rows",
    "detection_control_gap_rows",
    "directed_accessibility_rows",
    "discover_controllability_manifold",
    "effective_rank",
    "jacobian_rank",
    "layerwise_propagation",
    "load_state_samples",
    "local_controllability",
    "phase_transition_candidate_rows",
    "principal_angle_rows",
    "principal_angles",
    "representation_control_gap_rows",
    "save_state_samples",
    "split_half_stability_rows",
    "subspace_overlap",
    "summarize_controllability_boundaries",
    "summarize_directed_accessibility",
    "summarize_reachability",
    "summarize_split_half_stability",
    "summarize_trajectories",
    "target_orthogonal_decomposition",
    "trajectory_curvature",
]
