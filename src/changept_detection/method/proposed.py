"""Stable public API for the proposed local-global Wasserstein method."""

from changept_detection.method.local_global_wasserstein import (
    ABLATION_FROM_KEY,
    LocalGlobalWassersteinDetector,
    PROPOSED_DISPATCH,
    regime_labels_from_prototypes,
    run_proposed,
)

__all__ = [
    "ABLATION_FROM_KEY",
    "LocalGlobalWassersteinDetector",
    "PROPOSED_DISPATCH",
    "regime_labels_from_prototypes",
    "run_proposed",
]
