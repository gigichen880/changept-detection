"""
Experiment specification aligned with docs/experiment_plan.md.

Set A synthetic experiments use ids S0–S7 (plan sections A1–A7 plus recurring-regime
labeling as S7). Baseline pools follow §2.3; per-experiment lists add the detectors
called out in each section's expected-outcome discussion.
"""

from __future__ import annotations

# --- Set A: synthetic experiment registry (plan §Set A) ---

EXPERIMENT_ORDER = ["S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7"]

PLAN_SECTION = {
    "S0": "A1",
    "S1": "A2",
    "S2": "A3",
    "S3": "A4",
    "S4": "A5",
    "S5": "A6",
    "S6": "A7",
    "S7": "A7+regime-labels",
}

EXPERIMENT_DESCRIPTIONS = {
    "S0": "A1: mean/variance shift sanity check.",
    "S1": "A2: variance-matched Student-t tail shift.",
    "S2": "A3: scenario-mixture weight shift.",
    "S3": "A4: fixed-marginal correlation crisis.",
    "S4": "A5: low-rank factor covariance shock.",
    "S5": "A6: transient shock vs persistent regime.",
    "S6": "A7: duplicate local peak suppression.",
    "S7": "Recurring-regime posterior / label interpretability.",
}

EXPERIMENT_TITLES = {
    "S0": "S0 (A1): mean / variance shifts",
    "S1": "S1 (A2): variance-matched tail shift",
    "S2": "S2 (A3): mixture weight shift",
    "S3": "S3 (A4): correlation crisis (fixed marginals)",
    "S4": "S4 (A5): low-rank factor shock",
    "S5": "S5 (A6): transient vs persistent",
    "S6": "S6 (A7): duplicate peak suppression",
    "S7": "S7: recurring regime labels",
}

# --- Methods (plan §2.2–2.3) ---

PROPOSED_PRIMARY = "proposed_full"

PROPOSED_ABLATIONS: dict[str, str] = {
    "proposed_local_only": "Local Wasserstein alert only (§2.2)",
    "proposed_local_persistence_proxy": "Local alert + persistence / duplicate proxy (§2.2)",
    "proposed_local_global_no_proto": "Local + global refinement, no prototype (§2.2)",
    "proposed_local_proto_no_global": "Local + prototype posterior, no global (§2.2)",
    "proposed_full": "Full method: local + prototype + global + persistence (§2.1)",
}

PROPOSED_METHOD_KEYS = frozenset(
    set(PROPOSED_ABLATIONS) | {"proposed_local_global"}  # deprecated alias
)

# Compact baseline pool from plan §2.3
PLAN_BASELINE_POOL = [
    "cusum_mean",
    "cusum_vol",
    "pelt_rbf",
    "binseg",
    "mmd",
    "energy",
    "coordinate_w2_matched_filter",
    "sliced_wasserstein",
    "bures",
    "bocpd_gaussian",
    "gaussian_hmm",
]

# Per-experiment method lists (plan section emphasis + §2.3 pool)
BASELINE_SETS: dict[str, list[str]] = {
    # A1: classical CPD should be strong; OT included for comparison
    "S0": [
        "pelt_l2",
        "pelt_normal",
        "pelt_rbf",
        "binseg",
        "cusum_mean",
        "cusum_vol",
        "bocpd_gaussian",
        "coordinate_w2_window_scan",
        "coordinate_w2_matched_filter",
        PROPOSED_PRIMARY,
        "proposed_local_persistence_proxy",
    ],
    # A2: variance-matched tails — variance monitors + distributional tests
    "S1": [
        "ewma_vol",
        "cusum_vol",
        "ks",
        "cvm",
        "mmd",
        "energy",
        "pelt_rbf",
        "coordinate_w2_matched_filter",
        PROPOSED_PRIMARY,
    ],
    # A3: scenario mixture reweighting
    "S2": [
        "ks",
        "mmd",
        "energy",
        "pelt_rbf",
        "coordinate_w2_matched_filter",
        "bocpd_gaussian",
        PROPOSED_PRIMARY,
    ],
    # A4: joint dependence — coordinate OT should degrade; joint OT/kernel methods
    "S3": [
        "coordinate_w2t",
        "ks",
        "pelt_l2",
        "pelt_rbf",
        "mmd",
        "energy",
        "bures",
        "sliced_wasserstein",
        "sinkhorn",
        PROPOSED_PRIMARY,
        "proposed_local_global_no_proto",
    ],
    # A5: low-rank factor shock
    "S4": [
        "pelt_rbf",
        "mmd",
        "sliced_wasserstein",
        "sinkhorn",
        "covariance_frobenius",
        "pca_subspace",
        "bures",
        PROPOSED_PRIMARY,
    ],
    # A6: transient vs persistent — local vs full proposed ablations
    "S5": [
        "coordinate_w2_matched_filter",
        "mmd",
        "pelt_rbf",
        "bocpd_gaussian",
        "cusum_mean",
        "ewma_vol",
        PROPOSED_PRIMARY,
        "proposed_local_only",
        "proposed_local_persistence_proxy",
    ],
    # A7 duplicate suppression
    "S6": [
        "coordinate_w2_matched_filter",
        "mmd",
        "window_rbf",
        PROPOSED_PRIMARY,
        "proposed_local_persistence_proxy",
    ],
    # Recurring regimes — online/regime baselines + proposed prototype layer
    "S7": [
        "gaussian_hmm",
        "coordinate_w2_matched_filter",
        PROPOSED_PRIMARY,
        "proposed_local_proto_no_global",
    ],
}

# --- Metrics (plan §4) ---

PRIMARY_METRIC: dict[str, tuple[str, str]] = {
    "S0": ("f1", "Mean CP-F1"),
    "S1": ("f1", "Mean CP-F1"),
    "S2": ("f1", "Mean CP-F1"),
    "S3": ("f1", "Mean CP-F1"),
    "S4": ("f1", "Mean CP-F1"),
    "S5": ("f1", "Mean CP-F1"),
    "S6": ("duplicate_rate", "False duplicate rate (lower is better)"),
    "S7": ("f1", "Mean boundary CP-F1"),
}

S7_REGIME_METRIC = ("ari", "Mean ARI (regime labeling)")

S6_SUPPLEMENTARY_METRICS = (
    "mean_num_detections",
    "event_recall",
    "duplicate_rate_conditional_on_hit",
)

AUDIT_METRICS = (
    "mean_threshold",
    "mean_max_score",
    "mean_num_detected",
    "mean_f1",
)


def detection_tolerance(window: int) -> int:
    """Plan §3.2: |tau_hat - tau*| <= w/2."""
    return max(1, window // 2)
