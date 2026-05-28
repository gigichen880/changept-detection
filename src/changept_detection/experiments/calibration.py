"""
Null-sequence threshold calibration (experiment_plan.md §5.1).

Thresholds are estimated on stationary (no-change) data matched to each case
configuration, then frozen when evaluating that case with changepoints.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from changept_detection.baselines.core import DetectionResult, run_baseline


@dataclass
class CalibratedThresholds:
    """Frozen thresholds keyed by (config_key..., method)."""

    thresholds: dict[tuple[Any, ...], float] = field(default_factory=dict)
    null_seeds: int = 20
    false_alarm_quantile: float = 0.95
    metadata: dict[str, Any] = field(default_factory=dict)

    def get(self, config_key: tuple[Any, ...], method: str, default: float | None = None) -> float | None:
        return self.thresholds.get((*config_key, method), default)


def calibration_config_key(case: Any, window: int) -> tuple[Any, ...]:
    """
    Hashable key for a case DGP + geometry (per-config calibration).

    Includes experiment, window, dimension, and experiment-specific difficulty knobs.
    """
    p = case.params
    x = case.x
    d = int(p.get("d", x.shape[1] if x.ndim > 1 else 1))
    key: list[Any] = [case.experiment, window, d]

    per_exp: dict[str, tuple[str, ...]] = {
        "A1": ("mean_shift", "volatility_ratio", "n_per_segment"),
        "A2": ("nu", "garch_noise", "n_changepoints"),
        "A3": ("delta", "mode_separation", "demeaned", "centered", "serial_dependence"),
        "A4": ("delta_rho", "rho1", "n_per_segment"),
        "A5": ("epsilon", "n_factors", "sparsity", "n_per_segment"),
        "A6": ("persistent", "shock_type", "magnitude", "shock_length"),
        "A7": ("shift_family", "signal_strength", "noise_level", "window_length"),
        "A_regime": ("similarity", "regime_duration"),
    }
    for name in per_exp.get(case.experiment, ()):
        val = p.get(name)
        if isinstance(val, (list, np.ndarray)):
            val = tuple(val)
        key.append(val)
    return tuple(key)


def _max_score(result: DetectionResult) -> float:
    scores = result.scores
    finite = scores[np.isfinite(scores)] if len(scores) else np.array([])
    return float(np.max(finite)) if len(finite) else 0.0


def calibrate_method_threshold(
    method: str,
    null_series: list[np.ndarray],
    run_kwargs: dict[str, Any],
    false_alarm_quantile: float = 0.95,
) -> tuple[float, list[float]]:
    max_scores = []
    for x in null_series:
        result = run_baseline(method, x, **run_kwargs)
        if result.metadata.get("unavailable"):
            return float("nan"), []
        max_scores.append(_max_score(result))
    if not max_scores:
        return float("nan"), []
    return float(np.quantile(max_scores, false_alarm_quantile)), max_scores


def calibrate_proposed_shift_threshold(
    method: str,
    null_series: list[np.ndarray],
    run_kwargs: dict[str, Any],
    false_alarm_quantile: float = 0.95,
) -> tuple[float, list[float]]:
    """Calibrate posterior-shift threshold separately from local alert scores."""
    from changept_detection.method.proposed import run_proposed

    max_shifts: list[float] = []
    for x in null_series:
        kwargs = dict(run_kwargs)
        kwargs.pop("shift_threshold", None)
        kwargs["alert_threshold"] = 1e12
        kwargs["threshold"] = 1e12
        result = run_proposed(method, x, **kwargs)
        if result.metadata.get("unavailable"):
            return float("nan"), []
        shifts = result.metadata.get("posterior_shift_scores")
        if shifts is None:
            continue
        finite = np.asarray(shifts, dtype=float)
        finite = finite[np.isfinite(finite)]
        if len(finite):
            max_shifts.append(float(np.max(finite)))
    if not max_shifts:
        return float("nan"), []
    return float(np.quantile(max_shifts, false_alarm_quantile)), max_shifts


def calibrate_for_case(
    case: Any,
    methods: list[str],
    null_series: list[np.ndarray],
    kwargs_by_method: dict[str, dict[str, Any]],
    false_alarm_quantile: float = 0.95,
) -> CalibratedThresholds:
    """Calibrate each method for one case configuration."""
    window = 50
    if methods and methods[0] in kwargs_by_method:
        window = int(kwargs_by_method[methods[0]].get("window", 50))
    config_key = calibration_config_key(case, window)
    out = CalibratedThresholds(
        null_seeds=len(null_series),
        false_alarm_quantile=false_alarm_quantile,
    )
    for method in methods:
        kwargs = dict(kwargs_by_method.get(method, {}))
        kwargs.pop("threshold", None)
        kwargs.pop("threshold_quantile", None)
        kwargs.pop("alert_threshold", None)
        th, scores = calibrate_method_threshold(
            method, null_series, kwargs, false_alarm_quantile=false_alarm_quantile
        )
        if np.isfinite(th):
            out.thresholds[(*config_key, method)] = th
            out.metadata[str((*config_key, method))] = {
                "null_max_scores": scores,
                "threshold": th,
            }
        if method.startswith("proposed"):
            shift_th, shift_scores = calibrate_proposed_shift_threshold(
                method, null_series, kwargs, false_alarm_quantile=false_alarm_quantile
            )
            if np.isfinite(shift_th):
                out.thresholds[(*config_key, f"{method}_shift")] = shift_th
                out.metadata[str((*config_key, f"{method}_shift"))] = {
                    "null_max_shift_scores": shift_scores,
                    "threshold": shift_th,
                }
    return out


# Backward-compatible alias
def calibrate_experiment_methods(
    experiment: str,
    methods: list[str],
    null_series: list[np.ndarray],
    kwargs_by_method: dict[str, dict[str, Any]],
    false_alarm_quantile: float = 0.95,
) -> CalibratedThresholds:
    del experiment
    out = CalibratedThresholds(null_seeds=len(null_series), false_alarm_quantile=false_alarm_quantile)
    for method in methods:
        kwargs = dict(kwargs_by_method.get(method, {}))
        kwargs.pop("threshold", None)
        kwargs.pop("threshold_quantile", None)
        th, scores = calibrate_method_threshold(method, null_series, kwargs, false_alarm_quantile)
        if np.isfinite(th):
            out.thresholds[(method,)] = th
    return out
