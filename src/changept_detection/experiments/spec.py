"""
Experiment specification aligned with docs/experiment_plan.md.

Experiment ids match the plan directly:
  - A1–A7: Set A controlled synthetic sections
  - A_regime: framework extension for recurring-regime labeling (plan §4.3 metrics)

Sets B (B1–B3) and C (C1–C3) are documented in the plan but not implemented yet.
"""

from __future__ import annotations

# --- Set A experiment registry ---

EXPERIMENT_ORDER = ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A_regime"]

EXPERIMENTS: dict[str, dict[str, str]] = {
    "A1": {"plan_set": "A", "title": "Mean/variance shift sanity check"},
    "A2": {"plan_set": "A", "title": "Variance-matched tail shift"},
    "A3": {"plan_set": "A", "title": "Scenario-mixture weight shift"},
    "A4": {"plan_set": "A", "title": "Fixed-marginal correlation crisis"},
    "A5": {"plan_set": "A", "title": "Low-rank factor shock"},
    "A6": {"plan_set": "A", "title": "Transient shock vs persistent regime"},
    "A7": {"plan_set": "A", "title": "Duplicate local peak suppression"},
    "A_regime": {
        "plan_set": "A+",
        "title": "Recurring-regime labeling (§4.3 prototype / ARI-NMI metrics)",
    },
}

PLANNED_NOT_IMPLEMENTED: dict[str, list[str]] = {
    "Set B": ["B1 block-bootstrap splice", "B2 injected factor shock", "B3 scenario reweighting"],
    "Set C": ["C1 ETF event windows", "C2 Fama-French / industry portfolios", "C3 vol surface (optional)"],
}

EXPERIMENT_DESCRIPTIONS = {eid: meta["title"] + "." for eid, meta in EXPERIMENTS.items()}

EXPERIMENT_TITLES = {
    eid: f"{eid}: {meta['title'].split('(')[0].strip().lower()}" for eid, meta in EXPERIMENTS.items()
}

# Parameter grids (--grid quick | full). Counts = DGP configs before × seeds × methods.
GRID_DESCRIPTIONS: dict[str, dict[str, str]] = {
    "quick": {
        "purpose": "Smoke tests and local iteration; 1–16 DGP configs per experiment.",
        "cli": "--grid quick",
        "approx_cases": "A1–A6: 2 each; A7: 16; A_regime: 1 (≈29 configs for all)",
    },
    "full": {
        "purpose": "Cartesian sweeps over plan Set A difficulty knobs.",
        "cli": "--grid full",
        "approx_cases": "A1: 400; A2: 48; A3: 160; A4: 72; A5: 120; A6: 90; A7: 288; A_regime: 9",
    },
}

QUICK_GRID_CASES: dict[str, int] = {
    "A1": 2,
    "A2": 2,
    "A3": 2,
    "A4": 2,
    "A5": 2,
    "A6": 2,
    "A7": 16,
    "A_regime": 1,
}

FULL_GRID_CASES: dict[str, int] = {
    "A1": 400,
    "A2": 48,
    "A3": 160,
    "A4": 72,
    "A5": 120,
    "A6": 90,
    "A7": 288,
    "A_regime": 9,
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

BASELINE_SETS: dict[str, list[str]] = {
    "A1": [
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
    "A2": [
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
    "A3": [
        "ks",
        "mmd",
        "energy",
        "pelt_rbf",
        "coordinate_w2_matched_filter",
        "bocpd_gaussian",
        PROPOSED_PRIMARY,
    ],
    "A4": [
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
    "A5": [
        "pelt_rbf",
        "mmd",
        "sliced_wasserstein",
        "sinkhorn",
        "covariance_frobenius",
        "pca_subspace",
        "bures",
        PROPOSED_PRIMARY,
    ],
    "A6": [
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
    "A7": [
        "coordinate_w2_matched_filter",
        "mmd",
        "window_rbf",
        PROPOSED_PRIMARY,
        "proposed_local_persistence_proxy",
    ],
    "A_regime": [
        "gaussian_hmm",
        "coordinate_w2_matched_filter",
        PROPOSED_PRIMARY,
        "proposed_local_proto_no_global",
    ],
}

PRIMARY_METRIC: dict[str, tuple[str, str]] = {
    "A1": ("f1", "Mean CP-F1"),
    "A2": ("f1", "Mean CP-F1"),
    "A3": ("f1", "Mean CP-F1"),
    "A4": ("f1", "Mean CP-F1"),
    "A5": ("f1", "Mean CP-F1"),
    "A6": ("f1", "Mean CP-F1"),
    "A7": ("duplicate_rate", "False duplicate rate (lower is better)"),
    "A_regime": ("f1", "Mean boundary CP-F1"),
}

A_REGIME_LABEL_METRIC = ("ari", "Mean ARI (regime labeling)")

A7_SUPPLEMENTARY_METRICS = (
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
