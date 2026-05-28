"""
Placeholder for the proposed local–global Wasserstein regime filter.

Replace ``run_proposed`` (and optionally ``regime_labels_from_prototypes``) with the
real implementation when the method is ready. The experiment runner, calibration,
metrics, and plotting code should not need changes — only this module.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from changept_detection.baselines.core import (
    DetectionResult,
    adjacent_window_scores,
    as_2d,
    bures_wasserstein_cov,
    resource_table,
    sliced_wasserstein,
    threshold_from_quantile,
)
from changept_detection.experiments.spec import PROPOSED_ABLATIONS

_METRIC_FNS: dict[str, Callable[..., float]] = {
    "sliced_wasserstein": sliced_wasserstein,
    "bures": bures_wasserstein_cov,
    "coordinate_w2": lambda left, right: float(
        np.mean(
            [
                float(np.mean((np.sort(left[:, j]) - np.sort(right[:, j])) ** 2))
                for j in range(as_2d(left).shape[1])
            ]
        )
    ),
}


def _resolve_metric(name: str) -> Callable[..., float]:
    return _METRIC_FNS.get(name, sliced_wasserstein)


def _variant_flags(key: str) -> dict[str, bool]:
    """Map registry keys to intended layer toggles (for metadata only until implemented)."""
    return {
        "proposed_local_only": {
            "use_global": False,
            "use_prototypes": False,
            "use_persistence": False,
        },
        "proposed_local_persistence_proxy": {
            "use_global": False,
            "use_prototypes": False,
            "use_persistence": True,
        },
        "proposed_local_global_no_proto": {
            "use_global": True,
            "use_prototypes": False,
            "use_persistence": True,
        },
        "proposed_local_proto_no_global": {
            "use_global": False,
            "use_prototypes": True,
            "use_persistence": True,
        },
        "proposed_full": {
            "use_global": True,
            "use_prototypes": True,
            "use_persistence": True,
        },
        "proposed_local_global": {
            "use_global": False,
            "use_prototypes": False,
            "use_persistence": True,
        },
    }.get(
        key,
        {"use_global": True, "use_prototypes": True, "use_persistence": True},
    )


def run_proposed(
    key: str,
    x: np.ndarray,
    window: int = 50,
    threshold: float | None = None,
    alert_threshold: float | None = None,
    metric: str = "sliced_wasserstein",
    **kwargs: Any,
) -> DetectionResult:
    """
    Placeholder detector: exposes score/threshold plumbing for calibration and plots.

    **Replace the body of this function** with the real local–global Wasserstein
    regime filter. Keep the ``DetectionResult`` contract (changepoints, scores,
    threshold, metadata).
    """
    del kwargs
    metric_fn = _resolve_metric(metric)
    scores = adjacent_window_scores(x, window=window, metric=metric_fn, smooth=1, step=1)
    alert_th = alert_threshold if alert_threshold is not None else threshold
    if alert_th is None:
        alert_th = threshold_from_quantile(scores, 0.98)

    # No detections until the real method is plugged in.
    changepoints: list[int] = []
    flags = _variant_flags(key)
    return DetectionResult(
        key,
        changepoints,
        scores,
        alert_th,
        {
            "resource": resource_table([key])[0],
            "placeholder": True,
            "variant": key,
            "metric": metric,
            "window": window,
            **flags,
        },
    )


def regime_labels_from_prototypes(
    x: np.ndarray,
    window: int,
    n_prototypes: int,
    metric: str = "sliced_wasserstein",
    temperature: float = 0.5,
    n_em_rounds: int = 5,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], np.ndarray]:
    """
    Placeholder A_regime regime labeling.

    Replace with prototype-posterior assignments from the real method.
    Currently uses rolling-feature k-means so A_regime metrics/plots remain wired.
    """
    del metric, temperature, n_em_rounds
    from changept_detection.baselines.core import cluster_rolling_windows

    centers, labels = cluster_rolling_windows(x, window=window, n_clusters=n_prototypes)
    entropy = np.zeros(len(centers))
    prototypes: list[np.ndarray] = []
    return centers, labels, prototypes, entropy


PROPOSED_DISPATCH = {
    key: (lambda k: lambda x, **kw: run_proposed(k, x, **kw))(key)
    for key in PROPOSED_ABLATIONS
}
PROPOSED_DISPATCH["proposed_local_global"] = lambda x, **kw: run_proposed(
    "proposed_local_persistence_proxy", x, **kw
)
