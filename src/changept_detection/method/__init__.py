"""Proposed method and WPCG offline segmentation."""

from changept_detection.method.proposed import (
    PROPOSED_DISPATCH,
    matched_filter_1d,
    regime_labels_from_prototypes,
    run_proposed,
)
from changept_detection.method.wpcg import (
    coordinate_sweep_optimize,
    objective_J,
    w2_squared_1d,
)

__all__ = [
    "PROPOSED_DISPATCH",
    "coordinate_sweep_optimize",
    "matched_filter_1d",
    "objective_J",
    "regime_labels_from_prototypes",
    "run_proposed",
    "w2_squared_1d",
]
