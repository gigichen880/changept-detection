"""Detection diagnostics: true vs predicted changepoint timelines (Set A)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from changept_detection.baselines.core import DetectionResult, run_baseline
from changept_detection.experiments.calibration import CalibratedThresholds, calibrate_for_case, calibration_config_key
from changept_detection.experiments.spec import (
    BASELINE_SETS,
    EXPERIMENT_ORDER,
    PROPOSED_PRIMARY,
    detection_tolerance,
)
from changept_detection.experiments.synthetic import (
    SyntheticCase,
    baseline_kwargs,
    generate_null_series,
    make_cases,
    resolve_window,
)
from changept_detection.method.local_global_wasserstein import (
    LocalGlobalWassersteinDetector,
    _metric_to_distance,
)


@dataclass
class MethodDetectionSnapshot:
    method: str
    result: DetectionResult
    proposed_detail: dict[str, Any] | None = None


@dataclass
class CaseDetectionSnapshot:
    experiment: str
    case: SyntheticCase
    window: int
    tolerance: int
    methods: list[MethodDetectionSnapshot] = field(default_factory=list)


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Plotting requires matplotlib. Install with: pip install matplotlib") from exc
    return plt


def _series_for_plot(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        return x
    return np.mean(x, axis=1)


def _proposed_pipeline_detail(case: SyntheticCase, window: int, kwargs: dict[str, Any]) -> dict[str, Any]:
    metric = str(kwargs.get("metric", "sliced_wasserstein"))
    distance_type = _metric_to_distance(metric)
    local_th = kwargs.get("alert_threshold") or kwargs.get("threshold")
    posterior_th = kwargs.get("shift_threshold") or kwargs.get("threshold")
    det = LocalGlobalWassersteinDetector(
        window_size=window,
        refinement_horizon=int(kwargs.get("horizon", max(2 * window, 80))),
        n_prototypes=int(kwargs.get("n_prototypes", 3)),
        distance_type=distance_type,
        local_threshold=float(local_th) if local_th is not None else None,
        posterior_threshold=float(posterior_th) if posterior_th is not None else None,
        persistence=int(kwargs.get("min_persistence", 3)),
        merge_tolerance=int(kwargs.get("min_distance", window // 2)),
        min_segment_length=int(kwargs.get("min_seg_len", window)),
        ablation="full",
        random_state=int(kwargs.get("seed", kwargs.get("random_state", 42))),
        n_projections=int(kwargs.get("n_projections", 64)),
    )
    out = det.detect(case.x)
    return {
        "candidates": out["candidate_boundaries"],
        "global_retained": out["global_retained_boundaries"],
        "confirmed": out["confirmed_boundaries"],
        "local_threshold": out["config"]["local_threshold"],
        "posterior_threshold": out["config"]["posterior_threshold"],
        "alert_scores": out["alert_scores"],
        "posterior_shift_scores": out["posterior_shift_scores"],
        "retention_counts": out.get("retention_counts", {}),
    }


def run_representative_case(
    experiment: str,
    grid: str = "quick",
    seed: int = 0,
    baselines: list[str] | None = None,
    window: int | None = None,
    calibrate: bool = True,
    null_seeds: int = 8,
    false_alarm_quantile: float = 0.95,
) -> CaseDetectionSnapshot:
    """Run all methods on the first grid config for one experiment (diagnostic default)."""
    cases = make_cases(experiment, grid=grid, seeds=[seed])
    if not cases:
        raise ValueError(f"No cases for experiment {experiment}")
    case = cases[0]
    methods = baselines or BASELINE_SETS[experiment]
    w = resolve_window(case, window or max(20, min(80, len(case.x) // 8)))
    tol = detection_tolerance(w)

    calibration: CalibratedThresholds | None = None
    kwargs_map = {m: baseline_kwargs(m, case, window=w, n_bkps=len(case.changepoints)) for m in methods}
    if calibrate:
        null_series = generate_null_series(case, n_null=null_seeds)
        calibration = calibrate_for_case(
            case, methods, null_series, kwargs_map, false_alarm_quantile=false_alarm_quantile
        )
        cfg = calibration_config_key(case, w)
    else:
        cfg = None

    snapshots: list[MethodDetectionSnapshot] = []
    for method in methods:
        kwargs = dict(kwargs_map[method])
        if calibration is not None and cfg is not None:
            th = calibration.get(cfg, method)
            if th is not None and np.isfinite(th):
                kwargs["threshold"] = th
                if method.startswith("proposed"):
                    kwargs["alert_threshold"] = th
                    shift_th = calibration.get(cfg, f"{method}_shift")
                    if shift_th is not None and np.isfinite(shift_th):
                        kwargs["shift_threshold"] = shift_th
        result = run_baseline(method, case.x, **kwargs)
        detail = None
        if method == PROPOSED_PRIMARY and not result.metadata.get("unavailable"):
            detail = _proposed_pipeline_detail(case, w, kwargs)
        snapshots.append(MethodDetectionSnapshot(method, result, detail))

    return CaseDetectionSnapshot(experiment, case, w, tol, snapshots)


def _match_symbol(detected: list[int], truth: list[int], tolerance: int) -> str:
    if not detected:
        return "none"
    hits = []
    for d in detected:
        if any(abs(d - t) <= tolerance for t in truth):
            hits.append(d)
    if hits:
        return "hit"
    return "miss"


def print_diagnosis_report(snapshots: list[CaseDetectionSnapshot]) -> None:
    print("\n--- Detection diagnosis (representative case per experiment) ---")
    for snap in snapshots:
        truth = snap.case.changepoints
        print(f"\n{snap.experiment} | case={snap.case.name} | truth={truth} | w={snap.window} | tol={snap.tolerance}")
        for ms in snap.methods:
            if ms.result.metadata.get("unavailable"):
                print(f"  {ms.method:<36} UNAVAILABLE")
                continue
            det = ms.result.changepoints
            tag = _match_symbol(det, truth, snap.tolerance)
            finite = ms.result.scores[np.isfinite(ms.result.scores)]
            mx = float(np.max(finite)) if len(finite) else float("nan")
            th = ms.result.threshold
            th_str = f"{th:.3f}" if th is not None and np.isfinite(th) else "—"
            print(
                f"  {ms.method:<36} detected={det} [{tag}] "
                f"max_score={mx:.3f} threshold={th_str}"
            )
            if ms.proposed_detail:
                d = ms.proposed_detail
                print(
                    f"    proposed pipeline: candidates={d['candidates']} "
                    f"retained={d['global_retained']} confirmed={d['confirmed']}"
                )
                if truth and not d["confirmed"]:
                    t0 = truth[0]
                    alerts = d["alert_scores"]
                    if t0 < len(alerts) and np.isfinite(alerts[t0]):
                        print(
                            f"    alert at true CP t={t0}: {alerts[t0]:.3f} "
                            f"(local_th={d['local_threshold']:.3f}); "
                            f"non-overlapping windows peak often after t+w"
                        )


def plot_experiment_detections(snapshot: CaseDetectionSnapshot, out_path: Path) -> Path:
    """Timeline: series + true CPs + per-method detected CPs."""
    plt = _require_matplotlib()
    case = snapshot.case
    y = _series_for_plot(case.x)
    truth = case.changepoints
    methods = [ms for ms in snapshot.methods if not ms.result.metadata.get("unavailable")]
    if not methods:
        raise ValueError(f"No available methods to plot for {snapshot.experiment}")

    n_methods = len(methods)
    fig_h = 2.2 + 0.55 * n_methods
    fig, axes = plt.subplots(
        n_methods + 1,
        1,
        figsize=(14, fig_h),
        sharex=True,
        gridspec_kw={"height_ratios": [2.0] + [0.55] * n_methods},
    )
    if n_methods == 0:
        axes = [axes]

    ax0 = axes[0]
    ax0.plot(y, color="#333333", linewidth=0.7, alpha=0.9)
    for tau in truth:
        ax0.axvline(tau, color="#2ca02c", linewidth=2.0, linestyle="--", label="true CP" if tau == truth[0] else "")
        ax0.axvspan(tau - snapshot.tolerance, tau + snapshot.tolerance, color="#2ca02c", alpha=0.08)
    ax0.set_ylabel("series (mean dim)")
    ax0.set_title(
        f"{snapshot.experiment}: {case.name} | truth={truth} | w={snapshot.window} | tol=±{snapshot.tolerance}",
        fontsize=10,
    )
    ax0.grid(alpha=0.2)
    ax0.legend(loc="upper right", fontsize=8)

    for ax, ms in zip(axes[1:], methods):
        det = ms.result.changepoints
        is_proposed = ms.method == PROPOSED_PRIMARY or ms.method.startswith("proposed_")
        color = "#d62728" if is_proposed else "#1f77b4"
        ax.set_ylim(-0.5, 1.5)
        ax.set_yticks([])
        for tau in truth:
            ax.axvline(tau, color="#2ca02c", linewidth=1.5, linestyle="--", alpha=0.8)
            ax.axvspan(tau - snapshot.tolerance, tau + snapshot.tolerance, color="#2ca02c", alpha=0.06)
        for d in det:
            hit = any(abs(d - t) <= snapshot.tolerance for t in truth)
            ax.scatter(
                [d],
                [0.5],
                marker="x" if hit else "o",
                s=80,
                c="#2ca02c" if hit else color,
                linewidths=2,
                zorder=3,
            )
        tag = _match_symbol(det, truth, snapshot.tolerance)
        th = ms.result.threshold
        th_s = f"{th:.3f}" if th is not None and np.isfinite(th) else "—"
        ax.set_ylabel(ms.method, fontsize=7, rotation=0, labelpad=70, ha="right")
        ax.text(
            0.01,
            0.5,
            f"det={det} [{tag}] th={th_s}",
            transform=ax.transAxes,
            fontsize=7,
            va="center",
            ha="left",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7, edgecolor="none"),
        )
        ax.grid(axis="x", alpha=0.2)

    axes[-1].set_xlabel("time")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_proposed_pipeline(snapshot: CaseDetectionSnapshot, out_path: Path) -> Path | None:
    """Alert scores + candidate / retained / confirmed stages for proposed_full."""
    plt = _require_matplotlib()
    ms = next((m for m in snapshot.methods if m.method == PROPOSED_PRIMARY), None)
    if ms is None or ms.proposed_detail is None:
        return None

    d = ms.proposed_detail
    truth = snapshot.case.changepoints
    t = np.arange(len(snapshot.case.x))
    alerts = d["alert_scores"]
    shifts = d["posterior_shift_scores"]

    fig, axes = plt.subplots(3, 1, figsize=(14, 7), sharex=True)

    axes[0].plot(t, alerts, color="#333", linewidth=0.8, label="alert A_t")
    axes[0].axhline(d["local_threshold"], color="#ff7f0e", linestyle=":", label="local threshold")
    for tau in truth:
        axes[0].axvline(tau, color="#2ca02c", linestyle="--", alpha=0.8)
    for c in d["candidates"]:
        axes[0].axvline(c, color="#9467bd", alpha=0.35, linewidth=1)
    axes[0].set_ylabel("alert score")
    axes[0].legend(fontsize=7, loc="upper right")
    axes[0].set_title(f"{snapshot.experiment}: proposed_full pipeline (purple=candidates)")

    axes[1].plot(t, shifts, color="#8c564b", linewidth=0.8, label="posterior shift B_t")
    axes[1].axhline(d["posterior_threshold"], color="#ff7f0e", linestyle=":", label="posterior threshold")
    for tau in truth:
        axes[1].axvline(tau, color="#2ca02c", linestyle="--", alpha=0.8)
    axes[1].set_ylabel("posterior shift")
    axes[1].legend(fontsize=7, loc="upper right")

    axes[2].set_ylim(-0.5, 2.5)
    axes[2].set_yticks([0, 1, 2])
    axes[2].set_yticklabels(["confirmed", "retained", "candidates"])
    for tau in truth:
        axes[2].axvline(tau, color="#2ca02c", linestyle="--", alpha=0.9, linewidth=2)
    for c in d["candidates"]:
        axes[2].scatter([c], [2], marker="|", s=200, c="#9467bd")
    for c in d["global_retained"]:
        axes[2].scatter([c], [1], marker="|", s=200, c="#1f77b4")
    for c in d["confirmed"]:
        axes[2].scatter([c], [0], marker="|", s=200, c="#d62728")
    axes[2].set_xlabel("time")
    axes[2].grid(axis="x", alpha=0.2)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def generate_detection_diagnostic_plots(
    experiments: list[str] | None = None,
    grid: str = "quick",
    seed: int = 0,
    plot_dir: Path | None = None,
    calibrate: bool = True,
    null_seeds: int = 8,
    false_alarm_quantile: float = 0.95,
) -> tuple[list[CaseDetectionSnapshot], list[Path]]:
    experiments = experiments or list(EXPERIMENT_ORDER)
    plot_dir = plot_dir or Path("results/plots/diagnostics")
    snapshots: list[CaseDetectionSnapshot] = []
    paths: list[Path] = []

    for experiment in experiments:
        snap = run_representative_case(
            experiment,
            grid=grid,
            seed=seed,
            calibrate=calibrate,
            null_seeds=null_seeds,
            false_alarm_quantile=false_alarm_quantile,
        )
        snapshots.append(snap)
        det_path = plot_experiment_detections(snap, plot_dir / f"{experiment}_detections.png")
        paths.append(det_path)
        pipe_path = plot_proposed_pipeline(snap, plot_dir / f"{experiment}_proposed_pipeline.png")
        if pipe_path is not None:
            paths.append(pipe_path)

    print_diagnosis_report(snapshots)
    return snapshots, paths
