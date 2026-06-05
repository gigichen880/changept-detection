"""
Online local-global Wasserstein regime filter (docs/proposed_method.md).

Three layers:
  1. Local Wasserstein alert (non-overlapping reference/current windows)
  2. Regime prototype posterior + shift score
  3. Rolling greedy global refinement + persistence confirmation

--- KEY CHANGE (v2) ---
Decouple candidate generation from final-decision thresholding.

Problem in v1: the calibrated local_threshold was used as a gate for candidate
generation.  When the signal was subtle (e.g. A1 mean/variance sanity check),
NO candidates passed the gate, and the global optimization (paper eq. 19) was
never invoked -- producing zero detections even though the optimization would
have found the boundary.

Fix: when global refinement is enabled, candidates are generated permissively
(all local maxima above a low quantile).  The global optimization objective and
persistence confirmation handle false-alarm control.  The calibrated threshold
is still used for boundary-evidence scoring *within* the optimization, not as a
pre-filter that can zero out the candidate set.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from changept_detection.baselines.core import (
    DetectionResult,
    as_2d,
    resource_table,
    select_peaks,
    threshold_from_quantile,
)
from changept_detection.experiments.spec import PROPOSED_ABLATIONS
from changept_detection.method.global_refinement import (
    confirmed_from_retention_counts,
    filter_by_persistence,
    global_segmentation_score,
    greedy_global_refine,
    increment_retention_counts,
    merge_nearby,
)
from changept_detection.method.prototype_layer import (
    init_prototypes,
    posterior_shift,
    prototype_posterior,
    regime_label_from_posterior,
    update_prototypes_medoid,
)
from changept_detection.method.wasserstein_distances import make_distance_fn

ABLATION_FROM_KEY: dict[str, str] = {
    "proposed_full": "full",
    "proposed_local_only": "local_only",
    "proposed_local_global_no_proto": "local_global_no_prototype",
    "proposed_local_proto_no_global": "local_prototype_no_global",
    "proposed_local_persistence_proxy": "local_only",
    "proposed_local_global": "local_only",
}

ABLATION_DISTANCE: dict[str, str] = {
    "coordinate_w": "wasserstein_1d",
    "bures": "bures",
    "sliced": "sliced",
}

METRIC_TO_DISTANCE: dict[str, str] = {
    "sliced_wasserstein": "sliced",
    "sliced": "sliced",
    "bures": "bures",
    "bures_wasserstein": "bures",
    "coordinate_w2": "wasserstein_1d",
    "coordinate_w": "wasserstein_1d",
    "wasserstein_1d": "wasserstein_1d",
    "projected": "projected",
}


def _metric_to_distance(metric: str) -> str:
    return METRIC_TO_DISTANCE.get(metric, "sliced")


def alert_time_to_boundary(alert_t: int, window: int) -> int:
    """Map alert endpoint t to changepoint at the left edge of the current window."""
    return max(0, alert_t - window)


def build_boundary_evidence_scores(
    alert_scores: np.ndarray,
    shift_scores: np.ndarray,
    window: int,
    include_shift: bool = True,
) -> np.ndarray:
    """Max alert/shift evidence attributed to each boundary index tau."""
    n = len(alert_scores)
    boundary_scores = np.full(n, np.nan)
    for t in range(2 * window, n):
        tau = alert_time_to_boundary(t, window)
        for score in (alert_scores[t], shift_scores[t] if include_shift else np.nan):
            if np.isfinite(score):
                prev = boundary_scores[tau]
                if not np.isfinite(prev) or score > prev:
                    boundary_scores[tau] = score
    return boundary_scores


def localize_alert_times(alert_times: list[int], window: int) -> list[int]:
    return sorted(set(alert_time_to_boundary(t, window) for t in alert_times))


def _find_local_maxima(scores: np.ndarray, order: int = 1) -> list[int]:
    """Return indices of local maxima in *scores* (finite values only)."""
    maxima: list[int] = []
    for i in range(order, len(scores) - order):
        if not np.isfinite(scores[i]):
            continue
        is_max = True
        for j in range(1, order + 1):
            left = scores[i - j] if np.isfinite(scores[i - j]) else -np.inf
            right = scores[i + j] if i + j < len(scores) and np.isfinite(scores[i + j]) else -np.inf
            if scores[i] < left or scores[i] < right:
                is_max = False
                break
        if is_max:
            maxima.append(i)
    return maxima


@dataclass
class LocalGlobalWassersteinDetector:
    window_size: int = 50
    refinement_horizon: int = 250
    n_prototypes: int = 4
    distance_type: str = "sliced"
    local_threshold: float | None = None
    posterior_threshold: float | None = None
    threshold_alpha: float = 0.01
    boundary_penalty: float = 1.0
    short_segment_penalty: float = 1.0
    min_segment_length: int | None = None
    temperature: float = 1.0
    persistence: int = 3
    merge_tolerance: int | None = None
    prototype_init: str = "kmeans_windows"
    update_prototypes: bool = False
    max_candidates: int | None = None
    random_state: int = 42
    ablation: str = "full"
    n_projections: int = 64
    distance_kwargs: dict[str, Any] = field(default_factory=dict)

    # --- v2: permissive candidate generation ---
    # Quantile of alert scores used for candidate generation when global
    # refinement is active.  Default 0.5 (median) is deliberately permissive:
    # roughly half of all alert times become candidates, giving the global
    # optimization (paper eq. 19) a rich set to select from.  The calibrated
    # local_threshold is still used for evidence scoring inside the optimizer.
    candidate_quantile: float = 0.5

    prototypes: list[np.ndarray] = field(default_factory=list, init=False)
    _distance_fn: Callable[[np.ndarray, np.ndarray], float] | None = field(default=None, init=False)
    _logs: list[dict[str, Any]] = field(default_factory=list, init=False)
    _online_buffer: np.ndarray | None = field(default=None, init=False)
    _retention_counts: dict[int, int] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        if self.ablation in ABLATION_DISTANCE:
            self.distance_type = ABLATION_DISTANCE[self.ablation]
        self.min_segment_length = self.min_segment_length or self.window_size
        self.merge_tolerance = self.merge_tolerance or max(self.window_size // 2, 1)
        self._distance_fn = make_distance_fn(
            self.distance_type,
            n_projections=self.n_projections,
            seed=self.random_state,
            **self.distance_kwargs,
        )

    @property
    def use_prototypes(self) -> bool:
        return self.ablation in {"full", "local_prototype_no_global", "bures", "sliced", "coordinate_w"}

    @property
    def use_global(self) -> bool:
        return self.ablation in {"full", "local_global_no_prototype", "bures", "sliced", "coordinate_w"}

    @property
    def use_posterior_trigger(self) -> bool:
        return self.use_prototypes

    def fit(self, x: np.ndarray) -> LocalGlobalWassersteinDetector:
        data = as_2d(x)
        w = self.window_size
        warmup_end = min(len(data), max(2 * w + self.refinement_horizon, 3 * w))
        if self.use_prototypes:
            self.prototypes = init_prototypes(
                data,
                w,
                self.n_prototypes,
                self.distance_type,
                prototype_init=self.prototype_init,
                random_state=self.random_state,
                warmup_end=warmup_end,
            )
        if self.local_threshold is None:
            scores = self._compute_alert_scores(data)
            q = 1.0 - self.threshold_alpha
            self.local_threshold = threshold_from_quantile(scores, q)
        if self.use_posterior_trigger and self.posterior_threshold is None:
            _, shift_scores = self._compute_posterior_series(data)
            self.posterior_threshold = threshold_from_quantile(shift_scores, 1.0 - self.threshold_alpha)
        return self

    def _compute_alert_scores(self, data: np.ndarray) -> np.ndarray:
        w = self.window_size
        n = len(data)
        scores = np.full(n, np.nan)
        assert self._distance_fn is not None
        for t in range(2 * w, n):
            ref = data[t - 2 * w : t - w]
            cur = data[t - w : t]
            scores[t] = self._distance_fn(ref, cur)
        return scores

    def _compute_posterior_series(self, data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        w = self.window_size
        n = len(data)
        k = max(len(self.prototypes), 1)
        posteriors = np.full((n, k), np.nan)
        shift_scores = np.full(n, np.nan)
        labels = np.full(n, -1, dtype=int)
        assert self._distance_fn is not None
        for t in range(2 * w, n):
            cur = data[t - w : t]
            pi_t = prototype_posterior(
                cur,
                self.prototypes,
                self._distance_fn,
                self.temperature,
                distance_type=self.distance_type,
            )
            posteriors[t] = pi_t
            labels[t] = regime_label_from_posterior(pi_t)
            lag_t = t - w
            if lag_t >= 2 * w and np.all(np.isfinite(posteriors[lag_t])):
                shift_scores[t] = posterior_shift(pi_t, posteriors[lag_t])
        return posteriors, shift_scores

    def _maybe_update_prototypes(self, data: np.ndarray, horizon_end: int) -> None:
        if not self.update_prototypes or not self.use_prototypes or not self.prototypes:
            return
        w = self.window_size
        lo = max(2 * w, horizon_end - self.refinement_horizon)
        assigned: list[list[np.ndarray]] = [[] for _ in self.prototypes]
        assert self._distance_fn is not None
        for t in range(lo, horizon_end + 1):
            cur = data[t - w : t]
            pi_t = prototype_posterior(
                cur,
                self.prototypes,
                self._distance_fn,
                self.temperature,
                distance_type=self.distance_type,
            )
            assigned[int(np.argmax(pi_t))].append(cur.copy())
        self.prototypes = update_prototypes_medoid(
            assigned,
            self._distance_fn,
            distance_type=self.distance_type,
        )

    def detect(self, x: np.ndarray) -> dict[str, Any]:
        data = as_2d(x)
        n = len(data)
        w = self.window_size
        self._logs = []
        self._retention_counts = {}

        if self.local_threshold is None or (self.use_posterior_trigger and self.posterior_threshold is None):
            self.fit(data)

        alert_scores = self._compute_alert_scores(data)
        local_th = float(self.local_threshold or threshold_from_quantile(alert_scores, 0.98))
        posterior_th = self.posterior_threshold

        if self.use_prototypes and not self.prototypes:
            self.fit(data)

        if self.use_prototypes:
            regime_posteriors, shift_scores = self._compute_posterior_series(data)
            regime_labels = np.argmax(np.nan_to_num(regime_posteriors, nan=0.0), axis=1)
            if posterior_th is None:
                posterior_th = threshold_from_quantile(shift_scores, 1.0 - self.threshold_alpha)
        else:
            regime_posteriors = np.full((n, self.n_prototypes), np.nan)
            shift_scores = np.full(n, np.nan)
            regime_labels = np.full(n, -1, dtype=int)

        posterior_th = float(posterior_th) if posterior_th is not None else np.inf

        # ==================================================================
        # v2: ADAPTIVE PENALTY SCALING
        #
        # The boundary_penalty in eq. 19 must be commensurate with the
        # Wasserstein distances.  A penalty of 1.0 is reasonable when W2
        # values are O(1), but sliced-W2 on short windows often produces
        # values O(0.1).  We scale the penalty by the median alert score
        # so that boundary_penalty acts as a dimensionless multiplier.
        # ==================================================================
        finite_alerts = alert_scores[np.isfinite(alert_scores)]
        if len(finite_alerts) > 0:
            median_alert = float(np.median(finite_alerts))
        else:
            median_alert = 1.0
        scaled_boundary_penalty = self.boundary_penalty * median_alert
        scaled_short_penalty = self.short_segment_penalty * median_alert

        # ==================================================================
        # v2: DECOUPLED CANDIDATE GENERATION
        #
        # When global refinement is active, use a PERMISSIVE threshold for
        # candidate generation so the optimization (paper eq. 19) always has
        # material to work with.  The calibrated local_th is reserved for
        # boundary-evidence scoring *inside* the optimizer, not as a pre-gate.
        #
        # When global refinement is NOT active (local_only, etc.), the
        # candidates ARE the final output, so we keep the conservative
        # calibrated threshold.
        # ==================================================================

        if self.use_global:
            # Permissive candidate threshold: quantile of alert scores
            finite_alerts = alert_scores[np.isfinite(alert_scores)]
            if len(finite_alerts) > 0:
                cand_th = float(np.percentile(finite_alerts, self.candidate_quantile * 100))
            else:
                cand_th = 0.0
        else:
            # No global layer → candidates are final decisions → use calibrated threshold
            cand_th = local_th

        candidate_boundaries: list[int] = []
        alert_candidates: list[int] = []
        for t in range(2 * w, n):
            a_t = alert_scores[t]
            b_t = shift_scores[t]

            # Candidate generation uses the (possibly permissive) cand_th
            is_local = np.isfinite(a_t) and a_t > cand_th
            is_posterior = self.use_posterior_trigger and np.isfinite(b_t) and b_t > posterior_th

            if self.ablation == "local_global_no_prototype":
                is_candidate = is_local
            elif self.ablation == "local_only":
                is_candidate = is_local
            elif self.ablation == "local_prototype_no_global":
                is_candidate = is_local or is_posterior
            else:
                is_candidate = is_local or is_posterior

            boundary_t = alert_time_to_boundary(t, w)

            # Track whether this also exceeds the calibrated threshold (for logging)
            is_strong = np.isfinite(a_t) and a_t > local_th

            log_row = {
                "t": t,
                "boundary_t": boundary_t,
                "alert_score": float(a_t) if np.isfinite(a_t) else None,
                "posterior_shift_score": float(b_t) if np.isfinite(b_t) else None,
                "regime_posterior": regime_posteriors[t].copy() if self.use_prototypes else None,
                "regime_label": int(regime_labels[t]),
                "is_local_candidate": bool(is_local),
                "is_posterior_candidate": bool(is_posterior),
                "is_candidate": bool(is_candidate),
                "is_strong_candidate": bool(is_strong),
                "is_retained_by_global": False,
                "is_confirmed": False,
            }
            if is_candidate:
                alert_candidates.append(t)
                candidate_boundaries.append(boundary_t)
            self._logs.append(log_row)

        candidate_boundaries = sorted(set(candidate_boundaries))
        boundary_evidence = build_boundary_evidence_scores(
            alert_scores,
            shift_scores,
            w,
            include_shift=self.use_posterior_trigger,
        )

        # ==================================================================
        # v2: LOCAL-MAXIMA FALLBACK
        #
        # If even the permissive threshold produces no candidates (very quiet
        # series), fall back to all local maxima of the alert curve.  This
        # guarantees the global optimization always has *something* to
        # evaluate.  The penalty terms in eq. 19 will reject them if they
        # don't improve the segmentation.
        # ==================================================================
        if not candidate_boundaries and self.use_global:
            local_max_times = _find_local_maxima(alert_scores, order=max(1, w // 4))
            for t in local_max_times:
                boundary_t = alert_time_to_boundary(t, w)
                candidate_boundaries.append(boundary_t)
                alert_candidates.append(t)
            candidate_boundaries = sorted(set(candidate_boundaries))

        if not self.use_global:
            peaks = select_peaks(boundary_evidence, threshold=local_th, min_distance=self.merge_tolerance)
            if self.use_posterior_trigger:
                post_peaks = select_peaks(
                    build_boundary_evidence_scores(
                        np.full(n, np.nan),
                        shift_scores,
                        w,
                        include_shift=True,
                    ),
                    threshold=posterior_th,
                    min_distance=self.merge_tolerance,
                )
                peaks = sorted(set(peaks) | set(post_peaks))
            global_retained = peaks if peaks else candidate_boundaries
            confirmed = filter_by_persistence(global_retained, boundary_evidence, local_th, self.persistence)
        else:
            # === Global refinement path ===
            global_retained: list[int] = []
            step = max(1, w // 2)
            start_t = min(n - 1, 2 * w + self.refinement_horizon)
            retention_counts: dict[int, int] = {}
            n_refine_steps = max(1, len(range(start_t, n, step)))
            required_persistence = min(self.persistence, n_refine_steps)
            for t in range(start_t, n, step):
                lo_b = max(0, t - self.refinement_horizon - w)
                hi_b = max(0, t - w)
                horizon_cands = [c for c in candidate_boundaries if lo_b <= c <= hi_b]
                if not horizon_cands:
                    continue
                retained = greedy_global_refine(
                    data,
                    horizon_cands,
                    t,
                    self.refinement_horizon,
                    w,
                    self._distance_fn,
                    boundary_evidence,
                    boundary_penalty=scaled_boundary_penalty,
                    short_segment_penalty=scaled_short_penalty,
                    min_segment_length=self.min_segment_length,
                    merge_tolerance=self.merge_tolerance,
                    max_candidates=self.max_candidates,
                )
                global_retained.extend(retained)
                increment_retention_counts(
                    retention_counts,
                    retained,
                    self.merge_tolerance,
                    boundary_evidence,
                )
                self._maybe_update_prototypes(data, t)
            global_retained = merge_nearby(global_retained, boundary_evidence, self.merge_tolerance)
            self._retention_counts = retention_counts
            confirmed = confirmed_from_retention_counts(retention_counts, required_persistence)

            # ----------------------------------------------------------
            # v2: RELAXED FALLBACKS
            #
            # If persistence confirmation is too strict (e.g. short series
            # with few refinement steps), progressively relax.
            # ----------------------------------------------------------
            if not confirmed and global_retained:
                # Try with reduced persistence
                confirmed = filter_by_persistence(
                    global_retained, boundary_evidence, local_th, max(1, required_persistence - 1)
                )

            if not confirmed and global_retained:
                # Accept any globally-retained boundary
                confirmed = merge_nearby(global_retained, boundary_evidence, self.merge_tolerance)

            # ----------------------------------------------------------
            # v2: FULL-SEQUENCE FALLBACK
            #
            # If rolling refinement retained nothing (all candidates were
            # weak individually), try a single global pass over the entire
            # sequence.  This is the "just solve the optimization problem"
            # path: find the single best boundary that maximizes eq. 19.
            # ----------------------------------------------------------
            if not global_retained and candidate_boundaries:
                lo_full = 2 * w
                hi_full = n
                empty_score = global_segmentation_score(
                    data,
                    lo_full,
                    hi_full,
                    [],
                    w,
                    self._distance_fn,
                    scaled_boundary_penalty,
                    scaled_short_penalty,
                    self.min_segment_length,
                )
                # Evaluate each candidate as a single-boundary segmentation
                best_c = None
                best_score = empty_score
                for c in candidate_boundaries:
                    c_score = global_segmentation_score(
                        data,
                        lo_full,
                        hi_full,
                        [c],
                        w,
                        self._distance_fn,
                        scaled_boundary_penalty,
                        scaled_short_penalty,
                        self.min_segment_length,
                    )
                    if c_score > best_score:
                        best_score = c_score
                        best_c = c
                if best_c is not None:
                    global_retained = [best_c]
                    confirmed = [best_c]

        confirmed = merge_nearby(confirmed, boundary_evidence, self.merge_tolerance)

        confirmed_labels: dict[int, int] = {}
        for tau in confirmed:
            lo = tau
            hi = min(n, tau + w)
            if self.use_prototypes and np.any(np.isfinite(regime_posteriors[lo:hi, 0])):
                avg_pi = np.nanmean(regime_posteriors[lo:hi], axis=0)
                confirmed_labels[tau] = int(np.argmax(avg_pi))
            else:
                confirmed_labels[tau] = int(regime_labels[tau]) if regime_labels[tau] >= 0 else 0

        retained_set = set(global_retained)
        confirmed_set = set(confirmed)
        for row in self._logs:
            boundary_t = row["boundary_t"]
            row["is_retained_by_global"] = boundary_t in retained_set
            row["is_confirmed"] = boundary_t in confirmed_set

        return {
            "alert_scores": alert_scores,
            "posterior_shift_scores": shift_scores,
            "boundary_evidence_scores": boundary_evidence,
            "regime_posteriors": regime_posteriors,
            "regime_labels": regime_labels,
            "alert_candidate_times": sorted(set(alert_candidates)),
            "candidate_boundaries": candidate_boundaries,
            "global_retained_boundaries": sorted(set(global_retained)),
            "confirmed_boundaries": confirmed,
            "confirmed_labels": confirmed_labels,
            "retention_counts": dict(self._retention_counts),
            "prototype_info": {
                "n_prototypes": len(self.prototypes),
                "init": self.prototype_init,
                "update_prototypes": self.update_prototypes,
            },
            "config": {
                "window_size": w,
                "refinement_horizon": self.refinement_horizon,
                "distance_type": self.distance_type,
                "ablation": self.ablation,
                "local_threshold": local_th,
                "posterior_threshold": posterior_th,
                "candidate_threshold": cand_th,
                "boundary_penalty_raw": self.boundary_penalty,
                "boundary_penalty_scaled": scaled_boundary_penalty,
                "median_alert": median_alert,
                "persistence": self.persistence,
            },
            "logs": self._logs,
        }

    def partial_fit(self, x_t: np.ndarray) -> dict[str, Any]:
        """Append one observation and return the latest detector state (docs/proposed_method.md §3)."""
        row = as_2d(x_t)
        if row.shape[0] != 1:
            row = row.reshape(1, -1)
        if self._online_buffer is None:
            self._online_buffer = row.copy()
        else:
            self._online_buffer = np.vstack([self._online_buffer, row])

        t = len(self._online_buffer) - 1
        if t < 2 * self.window_size:
            return {"t": t, "ready": False, "confirmed_boundaries": [], "alert_score": np.nan}

        out = self.detect(self._online_buffer)
        last_alert = out["alert_scores"][t]
        return {
            "t": t,
            "ready": True,
            "confirmed_boundaries": out["confirmed_boundaries"],
            "alert_score": float(last_alert) if np.isfinite(last_alert) else np.nan,
            "posterior_shift_score": float(out["posterior_shift_scores"][t])
            if np.isfinite(out["posterior_shift_scores"][t])
            else np.nan,
            "retention_counts": out.get("retention_counts", {}),
        }

    def get_results(self) -> dict[str, Any]:
        return {
            "logs": self._logs,
            "retention_counts": dict(self._retention_counts),
            "online_buffer_length": len(self._online_buffer) if self._online_buffer is not None else 0,
        }


def run_proposed(key: str, x: np.ndarray, **kwargs: Any) -> DetectionResult:
    """Adapter used by the experiment framework baseline registry."""
    window = int(kwargs.get("window", 50))
    ablation = ABLATION_FROM_KEY.get(key, "full")
    metric = kwargs.get("metric", "sliced_wasserstein")
    distance_type = _metric_to_distance(str(metric))

    local_th = kwargs.get("alert_threshold")
    if local_th is None:
        local_th = kwargs.get("threshold")
    posterior_th = kwargs.get("shift_threshold")

    persistence = int(kwargs.get("min_persistence", 3 if key == "proposed_full" else 2))
    if key in {"proposed_local_only"}:
        persistence = 1
    if key == "proposed_local_persistence_proxy":
        persistence = max(persistence, 2)

    candidate_quantile = float(kwargs.get("candidate_quantile", 0.5))

    detector = LocalGlobalWassersteinDetector(
        window_size=window,
        refinement_horizon=int(kwargs.get("horizon", max(2 * window, 80))),
        n_prototypes=int(kwargs.get("n_prototypes", 3)),
        distance_type=distance_type,
        local_threshold=float(local_th) if local_th is not None else None,
        posterior_threshold=float(posterior_th) if posterior_th is not None else None,
        boundary_penalty=float(kwargs.get("penalty", 1.0)),
        short_segment_penalty=float(kwargs.get("short_segment_penalty", 1.0)),
        min_segment_length=int(kwargs.get("min_seg_len", max(10, window // 2))),
        temperature=float(kwargs.get("temperature", 1.0)),
        persistence=persistence,
        merge_tolerance=int(kwargs.get("min_distance", window // 2)),
        prototype_init=str(kwargs.get("prototype_init", "kmeans_windows")),
        update_prototypes=bool(kwargs.get("update_prototypes", False)),
        max_candidates=kwargs.get("max_candidates"),
        random_state=int(kwargs.get("seed", kwargs.get("random_state", 42))),
        ablation=ablation,
        n_projections=int(kwargs.get("n_projections", 64)),
        candidate_quantile=candidate_quantile,
    )

    out = detector.detect(x)
    metadata = {
        "resource": resource_table([key])[0],
        "placeholder": False,
        "variant": key,
        "ablation": ablation,
        "metric": metric,
        "distance_type": distance_type,
        "window": window,
        "horizon": detector.refinement_horizon,
        "candidate_boundaries": out["candidate_boundaries"],
        "global_retained_boundaries": out["global_retained_boundaries"],
        "regime_posteriors": out["regime_posteriors"],
        "regime_labels": out["regime_labels"],
        "posterior_shift_scores": out["posterior_shift_scores"],
        "confirmed_labels": out["confirmed_labels"],
        "retention_counts": out.get("retention_counts", {}),
        "use_global": detector.use_global,
        "use_prototypes": detector.use_prototypes,
    }
    return DetectionResult(
        key,
        out["confirmed_boundaries"],
        out["alert_scores"],
        out["config"]["local_threshold"],
        metadata,
    )


def regime_labels_from_prototypes(
    x: np.ndarray,
    window: int,
    n_prototypes: int,
    metric: str = "sliced_wasserstein",
    temperature: float = 1.0,
    n_em_rounds: int = 5,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], np.ndarray]:
    """A_regime labeling via the prototype posterior layer."""
    del n_em_rounds
    distance_type = _metric_to_distance(metric)
    detector = LocalGlobalWassersteinDetector(
        window_size=window,
        n_prototypes=n_prototypes,
        distance_type=distance_type,
        temperature=temperature,
        ablation="local_prototype_no_global",
        random_state=0,
    )
    detector.fit(x)
    data = as_2d(x)
    centers = np.arange(window, len(data) + 1)
    labels = np.zeros(len(centers), dtype=int)
    entropy = np.zeros(len(centers))
    assert detector._distance_fn is not None
    for i, end in enumerate(centers):
        cur = data[end - window : end]
        pi = prototype_posterior(
            cur,
            detector.prototypes,
            detector._distance_fn,
            temperature,
            distance_type=distance_type,
        )
        labels[i] = int(np.argmax(pi))
        entropy[i] = float(-np.sum(pi * np.log(pi + 1e-12)))
    return centers, labels, detector.prototypes, entropy


PROPOSED_DISPATCH = {
    key: (lambda k: lambda x, **kw: run_proposed(k, x, **kw))(key)
    for key in PROPOSED_ABLATIONS
}
PROPOSED_DISPATCH["proposed_local_global"] = lambda x, **kw: run_proposed(
    "proposed_local_persistence_proxy", x, **kw
)
