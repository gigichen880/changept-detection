"""Proposed local-global Wasserstein regime filter."""

from changept_detection.experiments.spec import PROPOSED_ABLATIONS, PROPOSED_PRIMARY
from changept_detection.method.proposed import (
    LocalGlobalWassersteinDetector,
    PROPOSED_DISPATCH,
    regime_labels_from_prototypes,
    run_proposed,
)

__all__ = [
    "LocalGlobalWassersteinDetector",
    "PROPOSED_ABLATIONS",
    "PROPOSED_DISPATCH",
    "PROPOSED_PRIMARY",
    "regime_labels_from_prototypes",
    "run_proposed",
]
