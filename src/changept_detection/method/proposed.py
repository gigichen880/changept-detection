"""
Proposed local–global Wasserstein regime filter (experiment_plan.md §3.1).

Ablation keys:
  - proposed_local_only: adjacent-window alert only
  - proposed_local_persistence_proxy: peak refine + persistence (old stub)
  - proposed_local_global_no_proto: local alert + horizon global refinement, no prototypes
  - proposed_local_proto_no_global: local alert + prototype posterior shift, no global refinement
  - proposed_full: local alert + prototype posterior + global refinement + persistence
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from changept_detection.baselines.core import (
    DetectionResult,
    WINDOW_METRICS,
    adjacent_window_scores,
    as_2d,
    resource_table,
    select_peaks,
    sliced_wasserstein,
    threshold_from_quantile,
)
from changept_detection.method.wpcg import coordinate_sweep_optimize, w2_squared_1d


def _metric_fn(name: str) -> Callable[..., float]:
    return WINDOW_METRICS.get(name, sliced_wasserstein)


def _window_block(x: np.ndarray, t: int, window: int) -> np.ndarray:
    data = as_2d(x)
    return data[max(0, t - window) : t]


def _init_prototypes(x: np.ndarray, window: int, n_prototypes: int, seed: int = 0) -> list[np.ndarray]:
    """Empirical windows as initial Wasserstein prototypes."""
    data = as_2d(x)
    rng = np.random.default_rng(seed)
    starts = []
    for t in range(window, len(data), max(1, window // 2)):
        starts.append(t)
    if len(starts) < n_prototypes:
        return [_window_block(x, min(len(data), window + i * window), window) for i in range(n_prototypes)]
    chosen = rng.choice(starts, size=n_prototypes, replace=False)
    return [_window_block(x, int(t), window) for t in chosen]


def prototype_posterior(
    cur: np.ndarray,
    prototypes: list[np.ndarray],
    metric: Callable[..., float],
    temperature: float,
) -> np.ndarray:
    distances = np.array([metric(cur, proto) for proto in prototypes], dtype=float)
    logits = -distances / max(temperature, 1e-6)
    logits -= np.max(logits)
    weights = np.exp(logits)
    weights /= np.sum(weights) + 1e-12
    return weights


def posterior_shift(pi_t: np.ndarray, pi_tm1: np.ndarray) -> float:
    return float(0.5 * np.sum(np.abs(pi_t - pi_tm1)))


def matched_filter_1d(scores: np.ndarray, width: int) -> np.ndarray:
    """Triangular matched filter (Cheng et al. style post-processing proxy)."""
    width = max(3, width)
    kernel = np.bartlett(width)
    kernel /= np.sum(kernel)
    filled = np.where(np.isfinite(scores), scores, 0.0)
    filtered = np.convolve(filled, kernel, mode="same")
    mask = np.isfinite(scores)
    out = np.full_like(scores, np.nan)
    out[mask] = filtered[mask]
    return out


def global_refinement_subset(
    x: np.ndarray,
    candidates: list[int],
    horizon_end: int,
    horizon: int,
    window: int,
    penalty: float,
    min_seg_len: int,
    metric: Callable[..., float],
) -> list[int]:
    """
    Select a subset of candidates on [horizon_end-H, horizon_end] maximizing
    segment separation minus change penalties (greedy add/drop).
    """
    lo = max(window, horizon_end - horizon)
    hi = horizon_end
    cand = sorted({c for c in candidates if lo <= c <= hi})
    if not cand:
        return []

    def objective(selected: list[int]) -> float:
        if not selected:
            return 0.0
        bounds = [lo] + selected + [hi]
        if any(bounds[i + 1] - bounds[i] < min_seg_len for i in range(len(bounds) - 1)):
            return -np.inf
        total = 0.0
        for i in range(len(bounds) - 2):
            a, b, c = bounds[i], bounds[i + 1], bounds[i + 2]
            total += metric(_window_block(x, b, window), _window_block(x, c, window))
        total -= penalty * len(selected)
        return total

    selected: list[int] = []
    improved = True
    while improved:
        improved = False
        for c in cand:
            if c in selected:
                continue
            trial = sorted(selected + [c])
            if objective(trial) > objective(selected):
                selected = trial
                improved = True
        for c in list(selected):
            trial = [t for t in selected if t != c]
            if objective(trial) > objective(selected):
                selected = trial
                improved = True
    return selected


def run_proposed(
    key: str,
    x: np.ndarray,
    window: int = 50,
    threshold: float | None = None,
    alert_threshold: float | None = None,
    shift_threshold: float | None = None,
    horizon: int | None = None,
    penalty: float = 0.05,
    min_seg_len: int = 10,
    min_persistence: int = 2,
    min_distance: int | None = None,
    metric: str = "sliced_wasserstein",
    n_prototypes: int = 3,
    temperature: float = 0.5,
    use_matched_filter: bool = True,
    use_global: bool = True,
    use_prototypes: bool = True,
    use_persistence: bool = True,
    use_wpcg_refine: bool = False,
    n_changepoints_hint: int | None = None,
) -> DetectionResult:
    data = as_2d(x)
    T = len(data)
    metric_fn = _metric_fn(metric)
    H = horizon or max(2 * window, 100)
    min_dist = min_distance or window

    # --- Local alert A_t (adjacent windows) ---
    alert_scores = adjacent_window_scores(
        x, window=window, metric=metric_fn, smooth=1, step=1
    )
    if use_matched_filter:
        alert_scores = matched_filter_1d(alert_scores, width=window)

    alert_th = alert_threshold if alert_threshold is not None else threshold
    if alert_th is None:
        alert_th = threshold_from_quantile(alert_scores, 0.98)

    shift_scores = np.zeros(T)
    pi_history: list[np.ndarray] = []
    prototypes = _init_prototypes(x, window, n_prototypes) if use_prototypes else []

    for t in range(window, T):
        cur = data[t - window : t]
        if use_prototypes and prototypes:
            pi_t = prototype_posterior(cur, prototypes, metric_fn, temperature)
            pi_history.append(pi_t)
            if len(pi_history) >= 2:
                shift_scores[t] = posterior_shift(pi_history[-1], pi_history[-2])
            # slow prototype update (exponential blend toward current window)
            k = int(np.argmax(pi_t))
            blend = 0.05
            prototypes[k] = (1 - blend) * prototypes[k] + blend * cur

    shift_th = shift_threshold if shift_threshold is not None else (
        threshold_from_quantile(shift_scores[shift_scores > 0], 0.98) if np.any(shift_scores > 0) else alert_th
    )

    local_candidates = set(select_peaks(alert_scores, alert_th, min_distance=max(1, window // 4)))
    if use_prototypes:
        local_candidates |= set(select_peaks(shift_scores, shift_th, min_distance=max(1, window // 4)))

    if key == "proposed_local_only":
        confirmed = select_peaks(alert_scores, alert_th, min_distance=min_dist)
        return DetectionResult(
            key, confirmed, alert_scores, alert_th,
            {"resource": resource_table([key])[0], "layer": "local_only"},
        )

    if key == "proposed_local_persistence_proxy":
        radius = max(window // 2, 1)
        refined: list[int] = []
        for candidate in sorted(local_candidates):
            lo = max(window, candidate - radius)
            hi = min(T - window, candidate + radius)
            if lo > hi:
                continue
            best = lo + int(np.nanargmax(alert_scores[lo : hi + 1]))
            nb = alert_scores[max(0, best - min_persistence) : best + min_persistence + 1]
            if np.sum(np.isfinite(nb) & (nb >= alert_th)) >= min_persistence:
                refined.append(best)
        selected = []
        for cp in sorted(set(refined), key=lambda i: alert_scores[i], reverse=True):
            if all(abs(cp - p) >= min_dist for p in selected):
                selected.append(cp)
        return DetectionResult(
            key, sorted(selected), alert_scores, alert_th,
            {"resource": resource_table([key])[0], "layer": "persistence_proxy"},
        )

    # Global layer over rolling horizon
    confirmed: list[int] = []
    if use_global:
        for t in range(H, T, max(1, window // 2)):
            horizon_cands = [c for c in local_candidates if t - H <= c <= t]
            if not horizon_cands:
                continue
            subset = global_refinement_subset(
                x, horizon_cands, t, H, window, penalty, min_seg_len, metric_fn
            )
            confirmed.extend(subset)
    else:
        confirmed = list(local_candidates)

    if not use_prototypes:
        pass  # candidates already from alerts only
    if not use_global:
        confirmed = list(local_candidates)

    # Persistence filter
    if use_persistence:
        kept = []
        for cp in sorted(set(confirmed), key=lambda i: alert_scores[i], reverse=True):
            nb = alert_scores[max(0, cp - min_persistence) : cp + min_persistence + 1]
            if np.sum(np.isfinite(nb) & (nb >= alert_th)) >= min_persistence:
                kept.append(cp)
        confirmed = kept

    # Duplicate suppression
    final: list[int] = []
    for cp in sorted(set(confirmed), key=lambda i: alert_scores[i], reverse=True):
        if all(abs(cp - p) >= min_dist for p in final):
            final.append(cp)

    # Optional offline WPCG polish when K is known (evaluation oracle / hint)
    if use_wpcg_refine and n_changepoints_hint and n_changepoints_hint > 0:
        y = data.mean(axis=1)
        tau_init = final[:n_changepoints_hint]
        while len(tau_init) < n_changepoints_hint:
            tau_init.append(int(len(y) * (len(tau_init) + 1) / (n_changepoints_hint + 1)))
        tau, _ = coordinate_sweep_optimize(y, tau_init, min_seg_len=min_seg_len, max_iter=15)
        final = tau

    combined = alert_scores + 0.5 * shift_scores
    return DetectionResult(
        key,
        sorted(final),
        combined,
        alert_th,
        {
            "resource": resource_table([key])[0],
            "metric": metric,
            "horizon": H,
            "penalty": penalty,
            "n_prototypes": n_prototypes,
            "use_global": use_global,
            "use_prototypes": use_prototypes,
            "use_matched_filter": use_matched_filter,
        },
    )


def regime_labels_from_prototypes(
    x: np.ndarray,
    window: int,
    n_prototypes: int,
    metric: str = "sliced_wasserstein",
    temperature: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], np.ndarray]:
    """Window-center regime assignments and posterior entropy (S7)."""
    data = as_2d(x)
    metric_fn = _metric_fn(metric)
    prototypes = _init_prototypes(x, window, n_prototypes)
    centers = np.arange(window, len(data) + 1)
    labels = np.zeros(len(centers), dtype=int)
    entropy = np.zeros(len(centers))
    for i, end in enumerate(centers):
        cur = data[end - window : end]
        pi = prototype_posterior(cur, prototypes, metric_fn, temperature)
        labels[i] = int(np.argmax(pi))
        entropy[i] = float(-np.sum(pi * np.log(pi + 1e-12)))
    return centers, labels, prototypes, entropy


PROPOSED_DISPATCH = {
    "proposed_local_only": lambda x, **kw: run_proposed(
        "proposed_local_only", x, use_global=False, use_prototypes=False, use_persistence=False, **kw
    ),
    "proposed_local_persistence_proxy": lambda x, **kw: run_proposed(
        "proposed_local_persistence_proxy", x, use_global=False, use_prototypes=False, **kw
    ),
    "proposed_local_global_no_proto": lambda x, **kw: run_proposed(
        "proposed_local_global_no_proto", x, use_prototypes=False, **kw
    ),
    "proposed_local_proto_no_global": lambda x, **kw: run_proposed(
        "proposed_local_proto_no_global", x, use_global=False, **kw
    ),
    "proposed_full": lambda x, **kw: run_proposed("proposed_full", x, **kw),
    # backward-compatible alias
    "proposed_local_global": lambda x, **kw: run_proposed(
        "proposed_local_persistence_proxy", x, **kw
    ),
}
