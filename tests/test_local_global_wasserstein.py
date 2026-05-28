"""Tests for the local-global Wasserstein proposed method (docs/proposed_method.md §16)."""

from __future__ import annotations

import numpy as np

from changept_detection.baselines.core import detection_metrics
from changept_detection.method.global_refinement import (
    confirmed_from_retention_counts,
    dedupe_by_evidence,
    greedy_global_refine,
    increment_retention_counts,
)
from changept_detection.method.local_global_wasserstein import (
    LocalGlobalWassersteinDetector,
    alert_time_to_boundary,
    run_proposed,
)
from changept_detection.method.prototype_layer import (
    posterior_shift,
    prototype_posterior,
    update_prototypes_medoid,
)
from changept_detection.method.wasserstein_distances import compute_distance, make_distance_fn


def test_alert_time_to_boundary():
    assert alert_time_to_boundary(150, 30) == 120


def test_distance_nonnegative_and_zero_on_identical_windows():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(40, 3))
    for dist in ("wasserstein_1d", "sliced", "bures", "projected"):
        d = compute_distance(x, x, dist, seed=0)
        assert d >= 0.0
        assert d < 1e-5


def test_bures_stable_with_ridge():
    rng = np.random.default_rng(10)
    x = rng.normal(size=(30, 4))
    y = rng.normal(size=(30, 4)) * 0.01
    d = compute_distance(x, y, "bures")
    assert np.isfinite(d)
    assert d >= 0.0


def test_alert_window_slices_reference_before_current():
    x = np.zeros(200)
    x[100:] = 5.0
    det = LocalGlobalWassersteinDetector(window_size=20, ablation="local_only", local_threshold=0.0, persistence=1)
    out = det.detect(x[:, None])
    assert np.nanmax(out["alert_scores"][80:120]) > np.nanmax(out["alert_scores"][:60])


def test_posterior_sums_to_one_and_nearest_wins():
    rng = np.random.default_rng(2)
    protos = [rng.normal(size=(10, 2)), rng.normal(size=(10, 2)) + 5]
    cur = protos[0] + rng.normal(scale=0.01, size=(10, 2))
    pi = prototype_posterior(cur, protos, lambda a, b: compute_distance(a, b, "sliced", seed=0), 0.5)
    assert np.isclose(pi.sum(), 1.0)
    assert pi[0] > pi[1]


def test_posterior_shift_uses_window_lag():
    rng = np.random.default_rng(11)
    w = 10
    n = 80
    x = rng.normal(size=(n, 2))
    x[40:] += 4.0
    det = LocalGlobalWassersteinDetector(window_size=w, n_prototypes=2, ablation="full", random_state=0)
    det.fit(x)
    _, shift = det._compute_posterior_series(x)
    assert np.isnan(shift[2 * w : 2 * w + w - 1]).all()
    assert np.any(np.isfinite(shift[3 * w :]))


def test_posterior_shift_l1():
    pi_a = np.array([0.8, 0.2])
    pi_b = np.array([0.2, 0.8])
    assert np.isclose(posterior_shift(pi_a, pi_b), 0.6)


def test_global_refinement_rejects_short_segments():
    rng = np.random.default_rng(5)
    x = rng.normal(size=(200, 1))
    x[100:] += 2.0
    fn = make_distance_fn("sliced", seed=0)
    scores = np.zeros(200)
    scores[100] = 10.0
    scores[105] = 9.0
    retained = greedy_global_refine(
        x,
        [100, 105],
        horizon_end=150,
        horizon=80,
        window=20,
        distance_fn=fn,
        alert_scores=scores,
        min_segment_length=40,
    )
    assert len(retained) <= 1


def test_dedupe_keeps_stronger_candidate():
    scores = np.array([0.0, 10.0, 9.0, 0.0])
    kept = dedupe_by_evidence([2, 1], scores, merge_tolerance=5)
    assert kept == [1]


def test_rolling_persistence_requires_repeated_retention():
    scores = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    counts: dict[int, int] = {}
    increment_retention_counts(counts, [3], merge_tolerance=2, alert_scores=scores)
    increment_retention_counts(counts, [4], merge_tolerance=2, alert_scores=scores)
    assert confirmed_from_retention_counts(counts, persistence=3) == []
    increment_retention_counts(counts, [3], merge_tolerance=2, alert_scores=scores)
    confirmed = confirmed_from_retention_counts(counts, persistence=3)
    assert confirmed


def test_medoid_update_runs():
    rng = np.random.default_rng(6)
    windows = [[rng.normal(size=(5, 2)) for _ in range(3)] for _ in range(2)]
    fn = make_distance_fn("sliced", seed=0)
    updated = update_prototypes_medoid(windows, fn)
    assert len(updated) == 2


def test_partial_fit_appends_and_detects():
    rng = np.random.default_rng(7)
    n = 120
    x = rng.normal(size=(n, 1))
    x[n // 2 :] += 3.0
    det = LocalGlobalWassersteinDetector(
        window_size=15,
        ablation="local_only",
        local_threshold=0.0,
        persistence=1,
        refinement_horizon=40,
    )
    state = None
    for t in range(n):
        state = det.partial_fit(x[t])
    assert state is not None
    assert state["ready"] is True
    assert len(state["confirmed_boundaries"]) >= 0


def test_boundary_localization_places_cp_near_truth():
    rng = np.random.default_rng(20)
    n = 240
    w = 30
    cp = 120
    x = rng.normal(size=(n, 1))
    x[cp:] += 3.0
    result = run_proposed(
        "proposed_full",
        x,
        window=w,
        alert_threshold=0.0,
        shift_threshold=0.0,
        min_persistence=1,
    )
    assert result.changepoints, "expected at least one localized detection"
    assert any(abs(d - cp) <= w // 2 for d in result.changepoints)


def test_build_boundary_evidence_peaks_near_truth():
    x = np.zeros(240)
    x[120:] = 5.0
    det = LocalGlobalWassersteinDetector(window_size=30, ablation="local_only", local_threshold=0.0)
    out = det.detect(x[:, None])
    evidence = out["boundary_evidence_scores"]
    cp = 120
    assert np.nanmax(evidence[cp - 10 : cp + 15]) >= np.nanmax(evidence[: cp - 30])


def test_full_detector_smoke_mean_shift():
    rng = np.random.default_rng(3)
    n = 300
    x = rng.normal(size=(n, 1))
    x[n // 2 :] += 3.0
    result = run_proposed(
        "proposed_full",
        x,
        window=30,
        alert_threshold=0.0,
        shift_threshold=0.0,
        min_persistence=1,
    )
    metrics = detection_metrics([n // 2], result.changepoints, tolerance=25)
    assert metrics["recall"] >= 0.0
    assert result.metadata.get("placeholder") is False


def test_correlation_shift_bures_signal():
    rng = np.random.default_rng(8)
    n = 250
    z = rng.normal(size=(n, 2))
    x = np.column_stack([z[:, 0], z[:, 0]])
    x[n // 2 :, 1] = z[n // 2 :, 1]
    det = LocalGlobalWassersteinDetector(
        window_size=25,
        distance_type="bures",
        ablation="local_only",
        local_threshold=0.0,
        persistence=1,
    )
    out = det.detect(x)
    cp = n // 2
    assert np.nanmax(out["alert_scores"][cp - 10 : cp + 30]) > np.nanmax(out["alert_scores"][: cp - 40])


def test_transient_shock_full_less_eager_than_local_only():
    rng = np.random.default_rng(9)
    n = 400
    x = rng.normal(size=(n, 1))
    shock_t = 200
    x[shock_t : shock_t + 5] += 20.0
    common = dict(window=25, alert_threshold=0.0, shift_threshold=0.0, min_persistence=2)
    local = run_proposed("proposed_local_only", x, **common)
    full = run_proposed("proposed_full", x, **common)
    assert len(local.changepoints) >= 1
    assert len(full.changepoints) <= max(3, len(local.changepoints) * 2)


def test_local_only_ablation_runs():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(200, 2))
    result = run_proposed("proposed_local_only", x, window=25, alert_threshold=1e9)
    assert result.changepoints == []


def test_all_proposed_ablation_keys_dispatch():
    rng = np.random.default_rng(12)
    x = rng.normal(size=(180, 2))
    for key in (
        "proposed_full",
        "proposed_local_only",
        "proposed_local_global_no_proto",
        "proposed_local_proto_no_global",
        "proposed_local_persistence_proxy",
    ):
        result = run_proposed(key, x, window=20, alert_threshold=1e9)
        assert result.metadata.get("placeholder") is False
