"""Experiment metrics helpers aligned with docs/experiment_plan.md §4."""

from __future__ import annotations

from typing import Any

import numpy as np

from changept_detection.baselines.core import DetectionResult, duplicate_rate


def duplicate_rate_conditional_on_hit(truth, detected, event_window: int) -> float:
    truth_list = list(truth)
    detected_list = list(detected)
    extras = 0
    n_hit_events = 0
    for tau in truth_list:
        hits = [d for d in detected_list if abs(d - tau) <= event_window]
        if hits:
            n_hit_events += 1
            extras += max(0, len(hits) - 1)
    if n_hit_events == 0:
        return float("nan")
    return extras / n_hit_events


def score_diagnostics(result: DetectionResult) -> dict[str, float]:
    scores = result.scores
    finite = scores[np.isfinite(scores)] if len(scores) else np.array([])
    max_score = float(np.max(finite)) if len(finite) else 0.0
    threshold = float(result.threshold) if result.threshold is not None else float("nan")
    return {
        "max_score": max_score,
        "threshold": threshold,
        "num_detected": float(len(result.changepoints)),
        "above_threshold": float(max_score >= threshold) if np.isfinite(threshold) else float("nan"),
    }


def s6_metrics(truth, detected, event_window: int, detection: dict[str, float]) -> dict[str, float]:
    return {
        "duplicate_rate": duplicate_rate(truth, detected, event_window),
        "duplicate_rate_conditional_on_hit": duplicate_rate_conditional_on_hit(
            truth, detected, event_window
        ),
        "mean_num_detections": float(len(list(detected))),
        "event_recall": float(detection.get("recall", 0.0)),
    }


def with_localization_alias(metrics: dict[str, float]) -> dict[str, float]:
    if "mean_abs_error" in metrics:
        metrics = {**metrics, "localization_error": metrics["mean_abs_error"]}
    return metrics


def build_score_audit_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        if float(row.get("unavailable", 0.0)) == 1.0:
            continue
        grouped.setdefault((row["experiment"], row["method"]), []).append(row)

    audit = []
    for (experiment, method), method_rows in sorted(grouped.items()):

        def _mean(key: str) -> float:
            vals = []
            for r in method_rows:
                if key not in r or r[key] == "":
                    continue
                try:
                    v = float(r[key])
                    if np.isfinite(v):
                        vals.append(v)
                except (TypeError, ValueError):
                    pass
            return float(np.mean(vals)) if vals else float("nan")

        audit.append(
            {
                "experiment": experiment,
                "method": method,
                "n_runs": len(method_rows),
                "mean_threshold": _mean("threshold"),
                "mean_max_score": _mean("max_score"),
                "mean_num_detected": _mean("num_detected"),
                "frac_above_threshold": _mean("above_threshold"),
                "mean_f1": _mean("f1"),
                "mean_recall": _mean("recall"),
            }
        )
    return audit


def print_score_audit_table(rows: list[dict[str, Any]]) -> None:
    audit = build_score_audit_table(rows)
    print("\n--- Score / threshold audit (available methods) ---")
    print(f"{'exp':<4} {'method':<34} {'th':>7} {'max':>7} {'ndet':>5} {'frac>=th':>8} {'f1':>6}")
    for row in audit:
        print(
            f"{row['experiment']:<4} {row['method']:<34} "
            f"{row['mean_threshold']:7.3f} {row['mean_max_score']:7.3f} "
            f"{row['mean_num_detected']:5.2f} {row['frac_above_threshold']:8.3f} "
            f"{row['mean_f1']:6.3f}"
        )
