"""Plots comparing proposed_local_global against baselines on synthetic results."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

PROPOSED_METHOD = "proposed_full"

EXPERIMENT_ORDER = ["S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7"]

EXPERIMENT_TITLES = {
    "S0": "S0: mean / variance shifts",
    "S1": "S1: variance-matched tail shift",
    "S2": "S2: mixture weight shift",
    "S3": "S3: correlation crisis (fixed marginals)",
    "S4": "S4: low-rank factor shock",
    "S5": "S5: transient vs persistent",
    "S6": "S6: duplicate peak suppression",
    "S7": "S7: recurring regime labels",
}

# Primary metric per experiment (matches experiment_plan.md emphasis).
PRIMARY_METRIC = {
    "S0": ("f1", "Mean F1"),
    "S1": ("f1", "Mean F1"),
    "S2": ("f1", "Mean F1"),
    "S3": ("f1", "Mean F1"),
    "S4": ("f1", "Mean F1"),
    "S5": ("f1", "Mean F1"),
    "S6": ("duplicate_rate", "Mean duplicate rate (lower is better)"),
    "S7": ("f1", "Mean boundary F1"),
}

S7_REGIME_METRIC = ("ari", "Mean ARI (regime baselines)")


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
    """Mean metric per method for one experiment."""
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
    """Print a short sanity check against experiment-plan expectations."""
    print("\n--- Results audit ---")
    unavailable = sorted({row["method"] for row in rows if float(row.get("unavailable", 0.0)) == 1.0})
    if unavailable:
        print(f"Unavailable baselines (install optional deps): {', '.join(unavailable)}")

    def proposed_metric(exp: str, metric: str) -> float:
        scores = aggregate_by_method(rows, exp, metric)
        return scores.get(PROPOSED_METHOD, float("nan"))

    def best_other(exp: str, metric: str, higher_better: bool = True) -> tuple[str, float]:
        scores = aggregate_by_method(rows, exp, metric)
        others = {k: v for k, v in scores.items() if k != PROPOSED_METHOD}
        if not others:
            return ("—", float("nan"))
        key = max if higher_better else min
        best_method = key(others, key=others.get)
        return best_method, others[best_method]

    # S0: competitive on easy Gaussian shifts
    p = proposed_metric("S0", "f1")
    b, bv = best_other("S0", "f1")
    print(f"S0 F1 — proposed: {p:.3f}, best baseline ({b}): {bv:.3f} (classical methods often strong here)")

    # S3: coordinate-wise should be weak; structure-aware should beat it
    coord = aggregate_by_method(rows, "S3", "f1").get("coordinate_w2t", float("nan"))
    prop = proposed_metric("S3", "f1")
    b, bv = best_other("S3", "f1")
    print(
        f"S3 F1 — coordinate_w2t: {coord:.3f}, proposed: {prop:.3f}, best baseline ({b}): {bv:.3f} "
        "(expect coordinate_w2t weak vs multivariate methods)"
    )

    # S6: lower duplicate rate is better
    prop_dup = proposed_metric("S6", "duplicate_rate")
    local_dup = aggregate_by_method(rows, "S6", "duplicate_rate").get(
        "coordinate_w2_matched_filter", float("nan")
    )
    proxy_dup = aggregate_by_method(rows, "S6", "duplicate_rate").get(
        "proposed_local_persistence_proxy", float("nan")
    )
    print(
        f"S6 duplicate rate — matched_filter: {local_dup:.3f}, "
        f"persistence_proxy: {proxy_dup:.3f}, proposed_full: {prop_dup:.3f}"
    )

    # S7: boundary F1 vs regime ARI (different tasks until full prototype labels exist)
    prop_f1 = proposed_metric("S7", "f1")
    b, bv = best_other("S7", "f1")
    print(f"S7 boundary F1 — proposed: {prop_f1:.3f}, best baseline ({b}): {bv:.3f}")
    prop_ari = proposed_metric("S7", "ari")
    ari_scores = aggregate_by_method(rows, "S7", "ari")
    if ari_scores:
        best_ari = max(ari_scores, key=ari_scores.get)
        print(f"S7 regime ARI — proposed_full: {prop_ari:.3f}, best: {best_ari}={ari_scores[best_ari]:.3f}")

    proposed_rows = [r for r in summary if r.get("method") == PROPOSED_METHOD and r.get("available_runs", 0)]
    if not proposed_rows:
        print(f"WARNING: no available runs for {PROPOSED_METHOD} — check thresholds/window.")


def _style_bar(ax, methods: list[str], values: list[float], metric_label: str, title: str) -> None:
    plt = _require_matplotlib()
    colors = []
    for method in methods:
        if method == PROPOSED_METHOD:
            colors.append("#d62728")
        elif method.endswith("_proxy") or "unavailable" in method:
            colors.append("#bdbdbd")
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
    if PROPOSED_METHOD in methods:
        ax.axhline(
            values[methods.index(PROPOSED_METHOD)],
            color="#d62728",
            linestyle="--",
            alpha=0.35,
            linewidth=1,
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
    if experiment == "S6":
        ax.invert_yaxis()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_overview_grid(
    rows: list[dict[str, Any]],
    out_path: Path,
    grid: str,
) -> Path:
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
        short_title = experiment
        _style_bar(ax, methods, values, metric_label.split("(")[0].strip(), short_title)
        if experiment == "S6":
            ax.invert_yaxis()

    fig.suptitle(f"Proposed vs baselines across synthetic suite ({grid} grid)", fontsize=13, y=1.02)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_proposed_vs_best(
    rows: list[dict[str, Any]],
    out_path: Path,
    grid: str,
) -> Path:
    """Delta of proposed minus best baseline per experiment (primary metric)."""
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
            delta = best_val - scores[PROPOSED_METHOD]  # positive = proposed better (lower dup)
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
    ax.set_title(f"Proposed local-global vs strongest baseline ({grid} grid)")
    for i, (exp, delta, best) in enumerate(zip(experiments, deltas, best_names)):
        ax.text(i, delta, f"vs {best}\n{delta:+.2f}", ha="center", va="bottom" if delta >= 0 else "top", fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_recall_precision_scatter(
    rows: list[dict[str, Any]],
    out_path: Path,
) -> Path:
    """Per-run recall vs precision; highlights proposed method."""
    plt = _require_matplotlib()
    fig, ax = plt.subplots(figsize=(7, 6))
    for row in _available_rows(rows):
        if row.get("experiment") == "S7":
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
    ax.set_title("Detection trade-off (S0–S6, all baselines)")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_s7_dual(rows: list[dict[str, Any]], out_path: Path, grid: str) -> Path:
    """S7: boundary F1 for CPD methods; ARI for regime-clustering baselines."""
    plt = _require_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))

    f1_scores = aggregate_by_method(rows, "S7", "f1")
    if f1_scores:
        methods = sorted(f1_scores, key=lambda m: f1_scores[m], reverse=True)
        _style_bar(axes[0], methods, [f1_scores[m] for m in methods], "Mean F1", "S7 boundary detection")

    ari_scores = aggregate_by_method(rows, "S7", "ari")
    if ari_scores:
        methods = sorted(ari_scores, key=lambda m: ari_scores[m], reverse=True)
        _style_bar(axes[1], methods, [ari_scores[m] for m in methods], "Mean ARI", "S7 regime labeling")

    fig.suptitle(f"S7: recurring regimes ({grid} grid)", fontsize=11)
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
    del summary  # reserved for future summary-only plots
    paths: list[Path] = []
    for experiment in EXPERIMENT_ORDER:
        exp_rows = [r for r in rows if r.get("experiment") == experiment]
        if not exp_rows:
            continue
        if experiment == "S7":
            paths.append(plot_s7_dual(exp_rows, plot_dir / "S7.png", grid))
        else:
            paths.append(plot_single_experiment(exp_rows, experiment, plot_dir / f"{experiment}.png", grid))
    paths.append(plot_overview_grid(rows, plot_dir / "overview_S0-S7.png", grid))
    paths.append(plot_proposed_vs_best(rows, plot_dir / "proposed_vs_best.png", grid))
    paths.append(plot_recall_precision_scatter(rows, plot_dir / "recall_precision_scatter.png"))
    return paths
