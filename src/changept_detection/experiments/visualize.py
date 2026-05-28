"""Plots comparing the proposed method against plan-aligned baselines."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from changept_detection.experiments.metrics import print_score_audit_table
from changept_detection.experiments.spec import (
    A_REGIME_LABEL_METRIC,
    EXPERIMENT_ORDER,
    EXPERIMENT_TITLES,
    PRIMARY_METRIC,
    PROPOSED_PRIMARY,
)

PROPOSED_METHOD = PROPOSED_PRIMARY


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Plotting requires matplotlib. Install with: pip install matplotlib") from exc
    return plt


def _available_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if float(row.get("unavailable", 0.0)) == 0.0]


def aggregate_by_method(
    rows: list[dict[str, Any]],
    experiment: str,
    metric: str,
) -> dict[str, float]:
    out: dict[str, list[float]] = {}
    for row in _available_rows(rows):
        if row.get("experiment") != experiment:
            continue
        if metric not in row or row[metric] == "":
            continue
        try:
            value = float(row[metric])
        except (TypeError, ValueError):
            continue
        out.setdefault(row["method"], []).append(value)
    return {method: float(np.mean(values)) for method, values in out.items()}


def print_results_audit(rows: list[dict[str, Any]], summary: list[dict[str, Any]]) -> None:
    print("\n--- Results audit ---")
    unavailable = sorted({row["method"] for row in rows if float(row.get("unavailable", 0.0)) == 1.0})
    if unavailable:
        print(f"Unavailable baselines (install optional deps): {', '.join(unavailable)}")

    def proposed_metric(exp: str, metric: str) -> float:
        return aggregate_by_method(rows, exp, metric).get(PROPOSED_METHOD, float("nan"))

    def best_other(exp: str, metric: str, higher_better: bool = True) -> tuple[str, float]:
        scores = aggregate_by_method(rows, exp, metric)
        others = {k: v for k, v in scores.items() if k != PROPOSED_METHOD}
        if not others:
            return ("—", float("nan"))
        key_fn = max if higher_better else min
        best_method = key_fn(others, key=others.get)
        return best_method, others[best_method]

    p = proposed_metric("A1", "f1")
    b, bv = best_other("A1", "f1")
    print(f"A1 F1 — proposed: {p:.3f}, best baseline ({b}): {bv:.3f}")

    coord = aggregate_by_method(rows, "A4", "f1").get("coordinate_w2t", float("nan"))
    prop = proposed_metric("A4", "f1")
    b, bv = best_other("A4", "f1")
    print(
        f"A4 F1 — coordinate_w2t: {coord:.3f}, proposed: {prop:.3f}, "
        f"best baseline ({b}): {bv:.3f}"
    )

    prop_rec = proposed_metric("A7", "event_recall")
    prop_ndet = proposed_metric("A7", "mean_num_detections")
    prop_dup = proposed_metric("A7", "duplicate_rate_conditional_on_hit")
    print(
        f"A7 — proposed: event_recall={prop_rec:.3f}, num_det={prop_ndet:.2f}, "
        f"dup_rate|hit={prop_dup:.3f}"
    )

    prop_f1 = proposed_metric("A_regime", "f1")
    b, bv = best_other("A_regime", "f1")
    print(f"A_regime boundary F1 — proposed: {prop_f1:.3f}, best baseline ({b}): {bv:.3f}")
    prop_ari = proposed_metric("A_regime", "ari")
    ari_scores = aggregate_by_method(rows, "A_regime", "ari")
    if ari_scores:
        best_ari = max(ari_scores, key=ari_scores.get)
        print(f"A_regime ARI — proposed: {prop_ari:.3f}, best: {best_ari}={ari_scores[best_ari]:.3f}")

    proposed_rows = [r for r in summary if r.get("method") == PROPOSED_METHOD and r.get("available_runs", 0)]
    if not proposed_rows:
        print(f"WARNING: no available runs for {PROPOSED_METHOD}.")

    print_score_audit_table(rows)


def _style_bar(ax, methods: list[str], values: list[float], metric_label: str, title: str) -> None:
    plt = _require_matplotlib()
    colors = []
    for method in methods:
        if method == PROPOSED_METHOD:
            colors.append("#d62728")
        elif method.startswith("proposed_"):
            colors.append("#ff9896")
        else:
            colors.append("#4c72b0")
    bars = ax.bar(range(len(methods)), values, color=colors, edgecolor="white", linewidth=0.6)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(metric_label)
    ax.set_title(title, fontsize=10)
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        if np.isfinite(value):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )


def plot_single_experiment(
    rows: list[dict[str, Any]],
    experiment: str,
    out_path: Path,
    grid: str,
) -> Path:
    plt = _require_matplotlib()
    metric, metric_label = PRIMARY_METRIC[experiment]
    scores = aggregate_by_method(rows, experiment, metric)
    if not scores:
        raise ValueError(f"No plottable scores for {experiment}")

    higher_better = metric != "duplicate_rate"
    methods = sorted(scores, key=lambda m: scores[m], reverse=higher_better)
    values = [scores[m] for m in methods]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    title = f"{EXPERIMENT_TITLES[experiment]} ({grid} grid)"
    _style_bar(ax, methods, values, metric_label, title)
    if experiment == "A7":
        ax.invert_yaxis()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_overview_grid(rows: list[dict[str, Any]], out_path: Path, grid: str) -> Path:
    plt = _require_matplotlib()
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    axes_flat = axes.ravel()

    for ax, experiment in zip(axes_flat, EXPERIMENT_ORDER):
        metric, metric_label = PRIMARY_METRIC[experiment]
        scores = aggregate_by_method(rows, experiment, metric)
        if not scores:
            ax.set_title(f"{experiment} (no data)")
            ax.axis("off")
            continue
        higher_better = metric != "duplicate_rate"
        methods = sorted(scores, key=lambda m: scores[m], reverse=higher_better)
        values = [scores[m] for m in methods]
        _style_bar(ax, methods, values, metric_label.split("(")[0].strip(), experiment)
        if experiment == "A7":
            ax.invert_yaxis()

    fig.suptitle(f"Proposed vs baselines — Set A ({grid} grid)", fontsize=13, y=1.02)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_proposed_vs_best(rows: list[dict[str, Any]], out_path: Path, grid: str) -> Path:
    plt = _require_matplotlib()
    experiments = []
    deltas = []
    best_names = []

    for experiment in EXPERIMENT_ORDER:
        metric, _ = PRIMARY_METRIC[experiment]
        scores = aggregate_by_method(rows, experiment, metric)
        if PROPOSED_METHOD not in scores:
            continue
        others = {k: v for k, v in scores.items() if k != PROPOSED_METHOD}
        if not others:
            continue
        if metric == "duplicate_rate":
            best_val = min(others.values())
            best_name = min(others, key=others.get)
            delta = best_val - scores[PROPOSED_METHOD]
        else:
            best_val = max(others.values())
            best_name = max(others, key=others.get)
            delta = scores[PROPOSED_METHOD] - best_val
        experiments.append(experiment)
        deltas.append(delta)
        best_names.append(best_name)

    colors = ["#2ca02c" if d >= 0 else "#ff7f0e" for d in deltas]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(experiments, deltas, color=colors, edgecolor="white")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Proposed − best baseline (primary metric)")
    ax.set_title(f"Proposed vs strongest baseline ({grid} grid)")
    for i, (exp, delta, best) in enumerate(zip(experiments, deltas, best_names)):
        ax.text(
            i,
            delta,
            f"vs {best}\n{delta:+.2f}",
            ha="center",
            va="bottom" if delta >= 0 else "top",
            fontsize=8,
        )
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_recall_precision_scatter(rows: list[dict[str, Any]], out_path: Path) -> Path:
    plt = _require_matplotlib()
    fig, ax = plt.subplots(figsize=(7, 6))
    for row in _available_rows(rows):
        if row.get("experiment") == "A_regime":
            continue
        if "recall" not in row or "precision" not in row:
            continue
        try:
            recall = float(row["recall"])
            precision = float(row["precision"])
        except (TypeError, ValueError):
            continue
        method = row["method"]
        if method == PROPOSED_METHOD:
            ax.scatter(recall, precision, c="#d62728", s=60, alpha=0.85, label=PROPOSED_METHOD, zorder=3)
        else:
            ax.scatter(recall, precision, c="#4c72b0", s=18, alpha=0.25, zorder=1)
    handles, labels = ax.get_legend_handles_labels()
    if labels:
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc="lower left")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Detection trade-off (A1–A7)")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_a_regime_dual(rows: list[dict[str, Any]], out_path: Path, grid: str) -> Path:
    plt = _require_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))

    f1_scores = aggregate_by_method(rows, "A_regime", "f1")
    if f1_scores:
        methods = sorted(f1_scores, key=lambda m: f1_scores[m], reverse=True)
        _style_bar(axes[0], methods, [f1_scores[m] for m in methods], "Mean CP-F1", "A_regime boundary detection")

    ari_label = A_REGIME_LABEL_METRIC[1]
    ari_scores = aggregate_by_method(rows, "A_regime", "ari")
    if ari_scores:
        methods = sorted(ari_scores, key=lambda m: ari_scores[m], reverse=True)
        _style_bar(axes[1], methods, [ari_scores[m] for m in methods], ari_label, "A_regime regime labeling")

    fig.suptitle(f"A_regime: recurring regimes ({grid} grid)", fontsize=11)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def generate_all_plots(
    rows: list[dict[str, Any]],
    summary: list[dict[str, Any]],
    plot_dir: Path,
    grid: str = "quick",
) -> list[Path]:
    del summary
    paths: list[Path] = []
    for experiment in EXPERIMENT_ORDER:
        exp_rows = [r for r in rows if r.get("experiment") == experiment]
        if not exp_rows:
            continue
        if experiment == "A_regime":
            paths.append(plot_a_regime_dual(exp_rows, plot_dir / "A_regime.png", grid))
        else:
            paths.append(plot_single_experiment(exp_rows, experiment, plot_dir / f"{experiment}.png", grid))
    paths.append(plot_overview_grid(rows, plot_dir / "overview_set_a.png", grid))
    paths.append(plot_proposed_vs_best(rows, plot_dir / "proposed_vs_best.png", grid))
    paths.append(plot_recall_precision_scatter(rows, plot_dir / "recall_precision_scatter.png"))
    return paths
