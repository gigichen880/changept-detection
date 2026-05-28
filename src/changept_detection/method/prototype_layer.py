"""Regime prototype initialization and posterior layer (docs/proposed_method.md §7)."""

from __future__ import annotations

from typing import Callable

import numpy as np

from changept_detection.baselines.core import as_2d, covariance_matrix, simple_kmeans


def window_feature_vector(block: np.ndarray) -> np.ndarray:
    block = as_2d(block)
    mean = np.mean(block, axis=0)
    std = np.std(block, axis=0, ddof=1) if len(block) > 1 else np.zeros(block.shape[1])
    cov = covariance_matrix(block).ravel()
    return np.r_[mean, std, cov]


def list_rolling_windows(x: np.ndarray, window: int, start: int, end: int) -> tuple[list[int], list[np.ndarray]]:
    data = as_2d(x)
    centers: list[int] = []
    windows: list[np.ndarray] = []
    for t in range(max(window, start), min(len(data), end) + 1):
        centers.append(t)
        windows.append(data[t - window : t].copy())
    return centers, windows


def init_prototypes_random_windows(
    x: np.ndarray,
    window: int,
    n_prototypes: int,
    random_state: int = 0,
    warmup_end: int | None = None,
) -> list[np.ndarray]:
    rng = np.random.default_rng(random_state)
    end = warmup_end if warmup_end is not None else len(as_2d(x))
    _, windows = list_rolling_windows(x, window, 2 * window, end)
    if not windows:
        return [as_2d(x)[max(0, len(x) - window) : len(x)].copy() for _ in range(n_prototypes)]
    idx = rng.choice(len(windows), size=min(n_prototypes, len(windows)), replace=False)
    return [windows[i] for i in idx]


def init_prototypes_kmeans_windows(
    x: np.ndarray,
    window: int,
    n_prototypes: int,
    random_state: int = 0,
    warmup_end: int | None = None,
) -> list[np.ndarray]:
    end = warmup_end if warmup_end is not None else len(as_2d(x))
    _, windows = list_rolling_windows(x, window, 2 * window, end)
    if len(windows) < n_prototypes:
        return init_prototypes_random_windows(x, window, n_prototypes, random_state, warmup_end=end)
    features = np.asarray([window_feature_vector(w) for w in windows])
    labels = simple_kmeans(features, n_clusters=n_prototypes, random_state=random_state)
    prototypes: list[np.ndarray] = []
    for k in range(n_prototypes):
        assigned = [windows[i] for i in range(len(windows)) if labels[i] == k]
        if assigned:
            prototypes.append(assigned[len(assigned) // 2])
        else:
            prototypes.append(windows[0])
    return prototypes


def init_prototypes_kmeans_covariance(
    x: np.ndarray,
    window: int,
    n_prototypes: int,
    random_state: int = 0,
    warmup_end: int | None = None,
    ridge: float = 1e-6,
) -> list[np.ndarray]:
    """Bures mode: store covariance-matrix prototypes (docs/proposed_method.md §7 Option B)."""
    end = warmup_end if warmup_end is not None else len(as_2d(x))
    _, windows = list_rolling_windows(x, window, 2 * window, end)
    if len(windows) < n_prototypes:
        return [covariance_matrix(windows[0] if windows else as_2d(x), ridge=ridge) for _ in range(n_prototypes)]
    covs = [covariance_matrix(w, ridge=ridge) for w in windows]
    features = np.asarray([c.ravel() for c in covs])
    labels = simple_kmeans(features, n_clusters=n_prototypes, random_state=random_state)
    prototypes: list[np.ndarray] = []
    for k in range(n_prototypes):
        assigned = [covs[i] for i in range(len(covs)) if labels[i] == k]
        prototypes.append(assigned[len(assigned) // 2] if assigned else covs[0])
    return prototypes


def init_prototypes(
    x: np.ndarray,
    window: int,
    n_prototypes: int,
    distance_type: str,
    prototype_init: str = "kmeans_windows",
    random_state: int = 0,
    warmup_end: int | None = None,
) -> list[np.ndarray]:
    if distance_type in {"bures", "bures_wasserstein"}:
        return init_prototypes_kmeans_covariance(
            x, window, n_prototypes, random_state=random_state, warmup_end=warmup_end
        )
    if prototype_init == "random_windows":
        return init_prototypes_random_windows(
            x, window, n_prototypes, random_state=random_state, warmup_end=warmup_end
        )
    return init_prototypes_kmeans_windows(
        x, window, n_prototypes, random_state=random_state, warmup_end=warmup_end
    )


def prototype_posterior(
    current_window: np.ndarray,
    prototypes: list[np.ndarray],
    distance_fn: Callable[[np.ndarray, np.ndarray], float],
    temperature: float,
    distance_type: str = "sliced",
) -> np.ndarray:
    from changept_detection.method.wasserstein_distances import prototype_distance

    distances = np.array(
        [
            prototype_distance(current_window, proto, distance_type, distance_fn)
            for proto in prototypes
        ],
        dtype=float,
    )
    logits = -(distances - np.min(distances)) / max(temperature, 1e-8)
    weights = np.exp(logits)
    weights /= np.sum(weights) + 1e-12
    return weights


def posterior_shift(pi_t: np.ndarray, pi_lagged: np.ndarray) -> float:
    """L1 distance between posteriors (docs/proposed_method.md §7)."""
    return float(0.5 * np.sum(np.abs(pi_t - pi_lagged)))


def regime_label_from_posterior(pi_t: np.ndarray) -> int:
    return int(np.argmax(pi_t))


def posterior_entropy(pi_t: np.ndarray) -> float:
    return float(-np.sum(pi_t * np.log(pi_t + 1e-12)))


def update_prototypes_medoid(
    assigned_windows: list[list[np.ndarray]],
    distance_fn: Callable[[np.ndarray, np.ndarray], float],
    distance_type: str = "sliced",
    ridge: float = 1e-6,
) -> list[np.ndarray]:
    """Medoid update per prototype cluster (docs/proposed_method.md §12)."""
    updated: list[np.ndarray] = []
    for windows in assigned_windows:
        if not windows:
            if distance_type in {"bures", "bures_wasserstein"}:
                updated.append(np.eye(1))
            else:
                updated.append(np.zeros((1, 1)))
            continue
        if len(windows) == 1:
            medoid = windows[0].copy()
        else:
            best_idx = 0
            best_cost = np.inf
            for i, cand in enumerate(windows):
                cost = float(
                    np.mean([distance_fn(cand, other) for j, other in enumerate(windows) if j != i])
                )
                if cost < best_cost:
                    best_cost = cost
                    best_idx = i
            medoid = windows[best_idx].copy()
        if distance_type in {"bures", "bures_wasserstein"}:
            updated.append(covariance_matrix(medoid, ridge=ridge))
        else:
            updated.append(medoid)
    return updated
