"""Synthetic experiment suite (Set A: A1–A7, A_regime), runner, calibration, and plots."""

from changept_detection.experiments.calibration import (
    CalibratedThresholds,
    calibrate_for_case,
    calibrate_experiment_methods,
    calibrate_method_threshold,
    calibration_config_key,
)
from changept_detection.experiments.runner import main, parse_args
from changept_detection.experiments.spec import BASELINE_SETS, EXPERIMENT_DESCRIPTIONS, PROPOSED_PRIMARY
from changept_detection.experiments.synthetic import (
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
