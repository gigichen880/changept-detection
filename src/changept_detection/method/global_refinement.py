"""Greedy global boundary refinement and persistence (docs/proposed_method.md §9–10)."""

from __future__ import annotations

from typing import Callable

import numpy as np

from changept_detection.baselines.core import as_2d


def segment_bounds(lo: int, hi: int, boundaries: list[int]) -> list[tuple[int, int]]:
    bounds = [lo] + sorted(boundaries) + [hi]
    return [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]


def short_segment_cost(segments: list[tuple[int, int]], min_segment_length: int) -> float:
    if min_segment_length <= 0:
        return 0.0
    cost = 0.0
    for a, b in segments:
        length = b - a
        if length < min_segment_length:
            cost += (min_segment_length - length) / min_segment_length
    return cost


def global_segmentation_score(
    x: np.ndarray,
    lo: int,
    hi: int,
    boundaries: list[int],
    window: int,
    distance_fn: Callable[[np.ndarray, np.ndarray], float],
    boundary_penalty: float,
    short_segment_penalty: float,
    min_segment_length: int,
) -> float:
    data = as_2d(x)
    segments = segment_bounds(lo, hi, boundaries)
    if any(b - a < 1 for a, b in segments):
        return -np.inf
    effective_min = min(min_segment_length, max(5, (hi - lo) // 4))
    if any(b - a < effective_min for a, b in segments):
        return -np.inf

    separation = 0.0
    for (a, b), (c, d) in zip(segments[:-1], segments[1:]):
        left = data[max(a, b - window) : b]
        right = data[c : min(c + window, d)]
        if len(left) == 0 or len(right) == 0:
            continue
        separation += distance_fn(left, right)

    score = separation
    score -= boundary_penalty * len(boundaries)
    score -= short_segment_penalty * short_segment_cost(segments, effective_min)
    return float(score)


def greedy_global_refine(
    x: np.ndarray,
    candidates: list[int],
    horizon_end: int,
    horizon: int,
    window: int,
    distance_fn: Callable[[np.ndarray, np.ndarray], float],
    alert_scores: np.ndarray,
    boundary_penalty: float = 1.0,
    short_segment_penalty: float = 1.0,
    min_segment_length: int = 10,
    merge_tolerance: int | None = None,
    max_candidates: int | None = None,
) -> list[int]:
    lo = max(window, horizon_end - horizon)
    hi = horizon_end
    merge_tolerance = merge_tolerance or max(window // 2, 1)
    span = max(hi - lo, 1)
    min_segment_length = min(min_segment_length, max(5, span // 4))

    cand = sorted({c for c in candidates if lo <= c <= hi})
    if not cand:
        return []

    if max_candidates is not None and len(cand) > max_candidates:
        cand = sorted(
            cand,
            key=lambda c: alert_scores[c] if np.isfinite(alert_scores[c]) else -np.inf,
            reverse=True,
        )[:max_candidates]
        cand = sorted(cand)

    cand = sorted(cand, key=lambda c: alert_scores[c] if np.isfinite(alert_scores[c]) else -np.inf, reverse=True)
    retained: list[int] = []
    for c in cand:
        trial = sorted(retained + [c])
        if global_segmentation_score(
            x, lo, hi, trial, window, distance_fn, boundary_penalty, short_segment_penalty, min_segment_length
        ) <= global_segmentation_score(
            x, lo, hi, retained, window, distance_fn, boundary_penalty, short_segment_penalty, min_segment_length
        ):
            continue
        retained.append(c)
        retained = dedupe_by_evidence(retained, alert_scores, merge_tolerance)

    return sorted(retained)


def dedupe_by_evidence(boundaries: list[int], alert_scores: np.ndarray, merge_tolerance: int) -> list[int]:
    kept: list[int] = []
    for c in sorted(
        boundaries,
        key=lambda i: alert_scores[i] if np.isfinite(alert_scores[i]) else -np.inf,
        reverse=True,
    ):
        if all(abs(c - p) > merge_tolerance for p in kept):
            kept.append(c)
    return sorted(kept)


def filter_by_persistence(
    candidates: list[int],
    alert_scores: np.ndarray,
    threshold: float,
    persistence: int,
) -> list[int]:
    if persistence <= 1:
        return sorted(candidates)
    kept: list[int] = []
    for c in sorted(set(candidates)):
        lo = max(0, c - persistence + 1)
        hi = min(len(alert_scores), c + persistence)
        nb = alert_scores[lo:hi]
        above = np.sum(np.isfinite(nb) & (nb >= threshold))
        if above >= persistence:
            kept.append(c)
    return sorted(kept)


def merge_nearby(boundaries: list[int], alert_scores: np.ndarray, merge_tolerance: int) -> list[int]:
    return dedupe_by_evidence(boundaries, alert_scores, merge_tolerance)


def increment_retention_counts(
    counts: dict[int, int],
    retained: list[int],
    merge_tolerance: int,
    alert_scores: np.ndarray,
) -> dict[int, int]:
    """Rolling persistence counters (docs/proposed_method.md §10)."""
    for r in retained:
        canonical = r
        for key in list(counts):
            if abs(key - r) <= merge_tolerance:
                canonical = key if alert_scores[key] >= alert_scores[r] else r
                if canonical != key:
                    counts[canonical] = counts.pop(key, 0)
                break
        counts[canonical] = counts.get(canonical, 0) + 1
    return counts


def confirmed_from_retention_counts(counts: dict[int, int], persistence: int) -> list[int]:
    return sorted([t for t, n in counts.items() if n >= persistence])
