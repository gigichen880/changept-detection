"""Synthetic experiment suite (S0–S7), runner, calibration, and plots."""

from changept_detection.experiments.calibration import (
    CalibratedThresholds,
    calibrate_experiment_methods,
    calibrate_method_threshold,
)
from changept_detection.experiments.runner import main, parse_args
from changept_detection.experiments.synthetic import (
    BASELINE_SETS,
    EXPERIMENT_DESCRIPTIONS,
    PROPOSED_PRIMARY,
    ExperimentResult,
    SyntheticCase,
    flatten_result,
    run_case,
    run_synthetic_suite,
)

__all__ = [
    "BASELINE_SETS",
    "EXPERIMENT_DESCRIPTIONS",
    "PROPOSED_PRIMARY",
    "CalibratedThresholds",
    "ExperimentResult",
    "SyntheticCase",
    "calibrate_experiment_methods",
    "calibrate_method_threshold",
    "flatten_result",
    "main",
    "parse_args",
    "run_case",
    "run_synthetic_suite",
]
