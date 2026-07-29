"""Local controllability measurements from differentiable control inputs."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch


def control_jacobian(
    response: Callable[[torch.Tensor], torch.Tensor],
    control: torch.Tensor,
    *,
    vectorize: bool = True,
) -> np.ndarray:
    """Differentiate a flattened state response with respect to a control tensor."""

    value = control.detach().clone().requires_grad_(True)

    def flattened(current: torch.Tensor) -> torch.Tensor:
        output = response(current)
        if not isinstance(output, torch.Tensor):
            raise TypeError("response must return a torch.Tensor")
        return output.reshape(-1)

    output_size = flattened(value).numel()
    jacobian = torch.autograd.functional.jacobian(
        flattened,
        value,
        vectorize=vectorize,
    )
    return jacobian.detach().float().cpu().numpy().reshape(output_size, -1)


def jacobian_rank(
    jacobian: np.ndarray,
    *,
    relative_tolerance: float = 1e-6,
) -> int:
    values = np.asarray(jacobian, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("jacobian must be a matrix")
    singular = np.linalg.svd(values, compute_uv=False)
    if singular.size == 0 or singular[0] == 0:
        return 0
    return int(np.count_nonzero(singular > singular[0] * relative_tolerance))


def local_controllability(
    jacobian: np.ndarray,
) -> dict[str, float | int]:
    """Rank, conditioning, and Gramian volume of a local linearization."""

    values = np.asarray(jacobian, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("jacobian must be a matrix")
    singular = np.linalg.svd(values, compute_uv=False)
    positive = singular[singular > max(singular[0] * 1e-12, 1e-15)] if singular.size else singular
    rank = jacobian_rank(values)
    condition = (
        float(positive[0] / positive[-1])
        if positive.size > 1
        else 1.0 if positive.size == 1 else float("inf")
    )
    log_pseudodeterminant = (
        float(2.0 * np.log(positive).sum()) if positive.size else float("-inf")
    )
    return {
        "rank": rank,
        "maximum_gain": float(singular[0]) if singular.size else 0.0,
        "minimum_nonzero_gain": float(positive[-1]) if positive.size else 0.0,
        "condition_number": condition,
        "log_gramian_pseudodeterminant": log_pseudodeterminant,
    }
