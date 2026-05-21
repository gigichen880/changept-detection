"""
WPCG / Eq. (13) utilities from wpcg_cpd.py.

Offline segmentation: maximize sum of adjacent-segment W2^2 via coordinate-wise
sweep. Used as an optional refinement step inside the proposed global layer, not
as the online local alert itself.
"""

from __future__ import annotations

import numpy as np


def w2_squared_1d(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if len(x) == 0 or len(y) == 0:
        return 0.0
    xs = np.sort(x)
    ys = np.sort(y)
    if len(xs) == len(ys):
        return float(np.mean((xs - ys) ** 2))
    n_q = 4 * max(len(xs), len(ys))
    u = (np.arange(n_q) + 0.5) / n_q
    fx = np.interp(u, (np.arange(len(xs)) + 0.5) / len(xs), xs)
    fy = np.interp(u, (np.arange(len(ys)) + 0.5) / len(ys), ys)
    return float(np.mean((fx - fy) ** 2))


def objective_J(x: np.ndarray, tau: list[int]) -> float:
    boundaries = [0] + list(tau) + [len(x)]
    total = 0.0
    for i in range(len(boundaries) - 2):
        a, b, c = boundaries[i], boundaries[i + 1], boundaries[i + 2]
        total += w2_squared_1d(x[a:b], x[b:c])
    return total


def coordinate_sweep_optimize(
    x: np.ndarray,
    tau_init: list[int],
    min_seg_len: int = 10,
    max_iter: int = 50,
) -> tuple[list[int], list[list[int]]]:
    T = len(x)
    tau = list(tau_init)
    history = [list(tau)]
    for _ in range(max_iter):
        prev = list(tau)
        for i in range(len(tau)):
            left_bd = tau[i - 1] if i > 0 else 0
            right_bd = tau[i + 1] if i < len(tau) - 1 else T
            lo = left_bd + min_seg_len
            hi = right_bd - min_seg_len
            if hi <= lo:
                continue
            best_val = -np.inf
            best_t = tau[i]
            for t in range(lo, hi + 1):
                val = w2_squared_1d(x[left_bd:t], x[t:right_bd])
                if val > best_val:
                    best_val = val
                    best_t = t
            tau[i] = best_t
        history.append(list(tau))
        if tau == prev:
            break
    return tau, history
