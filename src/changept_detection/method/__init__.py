"""Proposed method package — swap in the real detector via placeholder.py."""

from changept_detection.experiments.spec import PROPOSED_ABLATIONS, PROPOSED_PRIMARY
from changept_detection.method.placeholder import (
    PROPOSED_DISPATCH,
    regime_labels_from_prototypes,
    run_proposed,
)

__all__ = [
    "PROPOSED_ABLATIONS",
    "PROPOSED_DISPATCH",
    "PROPOSED_PRIMARY",
    "regime_labels_from_prototypes",
    "run_proposed",
]
