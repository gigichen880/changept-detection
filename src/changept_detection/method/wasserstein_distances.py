"""Wasserstein-family distances for window comparisons (docs/proposed_method.md §6)."""

from __future__ import annotations

from typing import Any

import numpy as np

from changept_detection.baselines.core import (
    as_2d,
    bures_wasserstein_cov,
    covariance_matrix,
    sinkhorn_divergence,
    sliced_wasserstein,
    w2_squared_1d,
)

SUPPORTED_DISTANCE_TYPES = frozenset(
    {"wasserstein_1d", "coordinate_w", "coordinate", "sliced", "sliced_wasserstein", "bures", "bures_wasserstein", "sinkhorn", "projected"}
)


def wasserstein_1d_avg(left: np.ndarray, right: np.ndarray) -> float:
    left = as_2d(left)
    right = as_2d(right)
    return float(np.mean([w2_squared_1d(left[:, j], right[:, j]) for j in range(left.shape[1])]))


def projected_wasserstein(
    left: np.ndarray,
    right: np.ndarray,
    n_components: int = 1,
    n_projections: int = 32,
    seed: int = 0,
) -> float:
    """PCA/factor projection then sliced Wasserstein (docs/proposed_method.md §6 projected)."""
    left = as_2d(left)
    right = as_2d(right)
    stacked = np.vstack([left, right])
    stacked = stacked - np.mean(stacked, axis=0)
    if stacked.shape[1] == 1:
        return wasserstein_1d_avg(left, right)
    _, _, vt = np.linalg.svd(stacked, full_matrices=False)
    k = min(n_components, vt.shape[0])
    basis = vt[:k].T
    left_p = left @ basis
    right_p = right @ basis
    return sliced_wasserstein(left_p, right_p, n_projections=n_projections, seed=seed)


def compute_distance(
    x_left: np.ndarray,
    x_right: np.ndarray,
    distance_type: str,
    **kwargs: Any,
) -> float:
    """Compare two empirical windows with the selected Wasserstein variant."""
    if distance_type in {"wasserstein_1d", "coordinate_w", "coordinate"}:
        return wasserstein_1d_avg(x_left, x_right)
    if distance_type in {"sliced", "sliced_wasserstein"}:
        return sliced_wasserstein(
            x_left,
            x_right,
            n_projections=int(kwargs.get("n_projections", 64)),
            seed=int(kwargs.get("seed", 0)),
        )
    if distance_type in {"bures", "bures_wasserstein"}:
        return bures_wasserstein_cov(x_left, x_right)
    if distance_type == "projected":
        return projected_wasserstein(
            x_left,
            x_right,
            n_components=int(kwargs.get("n_components", 1)),
            n_projections=int(kwargs.get("n_projections", 32)),
            seed=int(kwargs.get("seed", 0)),
        )
    if distance_type == "sinkhorn":
        try:
            return sinkhorn_divergence(x_left, x_right, reg=float(kwargs.get("reg", 0.1)))
        except Exception:
            return sliced_wasserstein(x_left, x_right, seed=int(kwargs.get("seed", 0)))
    raise ValueError(f"Unknown distance_type: {distance_type}")


def bures_between_cov(c1: np.ndarray, c2: np.ndarray) -> float:
    from scipy import linalg

    c1_sqrt = linalg.sqrtm(c1)
    middle = linalg.sqrtm(c1_sqrt @ c2 @ c1_sqrt)
    return float(np.real(np.trace(c1 + c2 - 2.0 * middle)))


def prototype_distance(
    current_window: np.ndarray,
    prototype: np.ndarray,
    distance_type: str,
    distance_fn,
    **kwargs: Any,
) -> float:
    """Distance from a current window to a prototype (window array or covariance matrix)."""
    if distance_type in {"bures", "bures_wasserstein"} and prototype.ndim == 2 and prototype.shape[0] == prototype.shape[1]:
        cur_cov = covariance_matrix(current_window, ridge=float(kwargs.get("ridge", 1e-6)))
        return bures_between_cov(cur_cov, prototype)
    return distance_fn(current_window, prototype)


def make_distance_fn(distance_type: str, **kwargs: Any):
    def _fn(left: np.ndarray, right: np.ndarray) -> float:
        return compute_distance(left, right, distance_type, **kwargs)

    return _fn
