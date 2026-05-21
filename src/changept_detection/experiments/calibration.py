"""
Null-sequence threshold calibration (experiment_plan.md §5.1).

Thresholds are estimated on stationary (no-change) data from the same DGP,
then frozen when evaluating sequences with changepoints.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from changept_detection.baselines.core import DetectionResult, run_baseline


ScoreFn = Callable[[np.ndarray], np.ndarray]


@dataclass
class CalibratedThresholds:
    """Frozen thresholds keyed by (experiment, method)."""

    thresholds: dict[tuple[str, str], float] = field(default_factory=dict)
    null_seeds: int = 20
    false_alarm_quantile: float = 0.95
    metadata: dict[str, Any] = field(default_factory=dict)

    def get(self, experiment: str, method: str, default: float | None = None) -> float | None:
        return self.thresholds.get((experiment, method), default)


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
    """
    Estimate threshold from null sequences using the max score statistic.

    ``false_alarm_quantile=0.95`` => ~5% of null runs exceed the threshold.
    """
    max_scores = []
    for x in null_series:
        result = run_baseline(method, x, **run_kwargs)
        if result.metadata.get("unavailable"):
            return float("nan"), []
        max_scores.append(_max_score(result))
    if not max_scores:
        return float("nan"), []
    return float(np.quantile(max_scores, false_alarm_quantile)), max_scores


def calibrate_experiment_methods(
    experiment: str,
    methods: list[str],
    null_series: list[np.ndarray],
    kwargs_by_method: dict[str, dict[str, Any]],
    false_alarm_quantile: float = 0.95,
) -> CalibratedThresholds:
    out = CalibratedThresholds(
        null_seeds=len(null_series),
        false_alarm_quantile=false_alarm_quantile,
    )
    for method in methods:
        kwargs = dict(kwargs_by_method.get(method, {}))
        kwargs.pop("threshold", None)
        kwargs.pop("threshold_quantile", None)
        th, scores = calibrate_method_threshold(
            method, null_series, kwargs, false_alarm_quantile=false_alarm_quantile
        )
        if np.isfinite(th):
            out.thresholds[(experiment, method)] = th
            out.metadata[method] = {"null_max_scores": scores, "threshold": th}
    return out
