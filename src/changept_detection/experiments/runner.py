"""Run synthetic CPD experiments and write tabular outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from changept_detection.baselines.core import resource_table
from changept_detection.experiments.metrics import build_score_audit_table
from changept_detection.experiments.spec import BASELINE_SETS, EXPERIMENT_DESCRIPTIONS
from changept_detection.experiments.synthetic import flatten_result, run_synthetic_suite


def json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return None
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def sanitize_nans(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: sanitize_nans(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_nans(v) for v in obj]
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    if isinstance(obj, np.generic):
        item = obj.item()
        if isinstance(item, float) and (np.isnan(item) or np.isinf(item)):
            return None
        return item
    return obj


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Set A synthetic changepoint experiments (docs/experiment_plan.md)."
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=list(BASELINE_SETS),
        choices=list(BASELINE_SETS),
        help="Set A experiments: A1–A7 plus A_regime (default: all).",
    )
    parser.add_argument(
        "--grid",
        choices=["quick", "full"],
        default="quick",
        help="Quick smoke grids or full difficulty sweeps from the plan.",
    )
    parser.add_argument("--seeds", type=int, default=1, help="Random seeds per parameter setting.")
    parser.add_argument("--baselines", nargs="+", default=None, help="Override default method keys.")
    parser.add_argument("--window", type=int, default=None, help="Rolling window length override.")
    parser.add_argument("--output-dir", default="results", help="Directory for CSV/JSON/plots.")
    parser.add_argument("--write-resources", action="store_true", help="Write baseline_resources.json.")
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate metric bar charts and detection timeline plots (requires matplotlib).",
    )
    parser.add_argument(
        "--plot-only",
        metavar="STEM",
        default=None,
        help="Skip experiments; replot from existing results/<STEM>/metrics/results.csv (legacy flat CSV also supported).",
    )
    parser.add_argument(
        "--no-calibrate",
        action="store_true",
        help="Use in-sample quantile thresholds (legacy; not plan §3.1).",
    )
    parser.add_argument(
        "--null-seeds",
        type=int,
        default=8,
        help="Null sequences per case config for threshold calibration (plan §3.1).",
    )
    parser.add_argument(
        "--false-alarm-quantile",
        type=float,
        default=0.95,
        help="Quantile of null max-scores for threshold (~5%% null FP rate at 0.95).",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the tqdm progress bar during experiment runs.",
    )
    parser.add_argument(
        "--plot-detections",
        action="store_true",
        help="Also generate detection timeline plots (included automatically with --plot).",
    )
    parser.add_argument(
        "--diagnostics-only",
        action="store_true",
        help="Skip experiment suite; only run detection timeline diagnostics.",
    )
    return parser.parse_args(argv)


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sanitize_nans(payload), indent=2, default=json_default) + "\n")


def _mean_metric(rows: list[dict[str, Any]], key: str) -> float:
    values = []
    for row in rows:
        if key not in row or row[key] == "":
            continue
        try:
            values.append(float(row[key]))
        except (TypeError, ValueError):
            continue
    return float(np.nanmean(values)) if values else float("nan")


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["experiment"], row["method"]), []).append(row)

    summary = []
    for (experiment, method), method_rows in sorted(grouped.items()):
        available = [row for row in method_rows if float(row.get("unavailable", 0.0)) == 0.0]
        if not available:
            summary.append(
                {
                    "experiment": experiment,
                    "method": method,
                    "available_runs": 0,
                    "mean_f1": np.nan,
                    "mean_recall": np.nan,
                    "mean_precision": np.nan,
                    "mean_ari": np.nan,
                    "mean_duplicate_rate": np.nan,
                }
            )
            continue
        summary.append(
            {
                "experiment": experiment,
                "method": method,
                "available_runs": len(available),
                "mean_f1": _mean_metric(available, "f1"),
                "mean_recall": _mean_metric(available, "recall"),
                "mean_precision": _mean_metric(available, "precision"),
                "mean_ari": _mean_metric(available, "ari"),
                "mean_duplicate_rate": _mean_metric(available, "duplicate_rate"),
                "mean_threshold": _mean_metric(available, "threshold"),
                "mean_max_score": _mean_metric(available, "max_score"),
                "mean_num_detected": _mean_metric(available, "num_detected"),
            }
        )
    return summary


def result_stem(grid: str, experiments: list[str]) -> str:
    return f"seta_{grid}_{'-'.join(experiments)}"


def run_dir(output_dir: Path, stem: str) -> Path:
    """Root folder for one experiment run."""
    return output_dir / stem


def metrics_dir(output_dir: Path, stem: str) -> Path:
    return run_dir(output_dir, stem) / "metrics"


def plots_metrics_dir(output_dir: Path, stem: str) -> Path:
    return run_dir(output_dir, stem) / "plots" / "metrics"


def plots_detections_dir(output_dir: Path, stem: str) -> Path:
    return run_dir(output_dir, stem) / "plots" / "detections"


def load_rows_from_output(output_dir: Path, stem: str) -> list[dict[str, Any]]:
    candidates = [
        metrics_dir(output_dir, stem) / "results.json",
        metrics_dir(output_dir, stem) / "results.csv",
        output_dir / f"{stem}.json",
        output_dir / f"{stem}.csv",
    ]
    for path in candidates:
        if not path.exists():
            continue
        if path.suffix == ".json":
            return json.loads(path.read_text())
        with path.open(newline="") as handle:
            return list(csv.DictReader(handle))
    raise FileNotFoundError(
        f"No results found for stem '{stem}' under {run_dir(output_dir, stem) / 'metrics'} "
        f"or legacy {output_dir / stem}.csv"
    )


def print_output_tree(output_dir: Path, stem: str) -> None:
    root = run_dir(output_dir, stem)
    print("\nOutput layout:")
    print(f"  {root}/")
    print("    metrics/")
    for name in ("results.csv", "results.json", "summary.json", "score_audit.csv", "score_audit.json"):
        path = root / "metrics" / name
        if path.exists():
            print(f"      {name}")
    print("    plots/")
    print("      metrics/          # bar charts (A1.png, overview_set_a.png, …)")
    metrics_plot_dir = plots_metrics_dir(output_dir, stem)
    if metrics_plot_dir.exists():
        for path in sorted(metrics_plot_dir.glob("*.png")):
            print(f"        {path.name}")
    print("      detections/       # true vs detected CP timelines + proposed pipeline")
    detections_plot_dir = plots_detections_dir(output_dir, stem)
    if detections_plot_dir.exists():
        for path in sorted(detections_plot_dir.glob("*.png")):
            print(f"        {path.name}")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    stem = result_stem(args.grid, args.experiments)

    if args.diagnostics_only:
        rows = []
        summary = []
    elif args.plot_only:
        stem = args.plot_only
        rows = load_rows_from_output(output_dir, stem)
        summary = summarize(rows)
    else:
        results = run_synthetic_suite(
            experiments=args.experiments,
            grid=args.grid,
            seeds=args.seeds,
            baselines=args.baselines,
            window=args.window,
            calibrate=not args.no_calibrate,
            null_seeds=args.null_seeds,
            false_alarm_quantile=args.false_alarm_quantile,
            show_progress=not args.no_progress,
        )
        rows = [flatten_result(result) for result in results]
        summary = summarize(rows)
        mdir = metrics_dir(output_dir, stem)
        write_csv(rows, mdir / "results.csv")
        write_json(rows, mdir / "results.json")
        write_json(summary, mdir / "summary.json")
        audit = build_score_audit_table(rows)
        write_json(audit, mdir / "score_audit.json")
        write_csv(audit, mdir / "score_audit.csv")
        if args.write_resources:
            write_json(resource_table(args.baselines), mdir / "baseline_resources.json")

        print("Set A synthetic CPD experiments complete")
        print(f"Experiments: {', '.join(args.experiments)}")
        for experiment in args.experiments:
            print(f"  {experiment}: {EXPERIMENT_DESCRIPTIONS[experiment]}")
        print(f"Rows: {len(rows)}")
        print(f"Metrics: {mdir / 'results.csv'}")
        print(f"Summary: {mdir / 'summary.json'}")
        print(f"Score audit: {mdir / 'score_audit.csv'}")

    if args.plot or args.plot_only:
        from changept_detection.experiments.visualize import generate_all_plots, print_results_audit

        plot_dir = plots_metrics_dir(output_dir, stem)
        paths = generate_all_plots(rows, summary, plot_dir, grid=args.grid)
        print_results_audit(rows, summary)
        print(f"Metric plots: {plot_dir}")
        for path in paths:
            print(f"  {path}")

    if args.plot or args.plot_detections or args.diagnostics_only:
        from changept_detection.experiments.diagnostics import generate_detection_diagnostic_plots

        diag_dir = plots_detections_dir(output_dir, stem)
        _, diag_paths = generate_detection_diagnostic_plots(
            experiments=args.experiments,
            grid=args.grid,
            seed=0,
            plot_dir=diag_dir,
            calibrate=not args.no_calibrate,
            null_seeds=args.null_seeds,
            false_alarm_quantile=args.false_alarm_quantile,
        )
        print(f"Detection plots: {diag_dir}")
        for path in diag_paths:
            print(f"  {path}")

    print_output_tree(output_dir, stem)


if __name__ == "__main__":
    main()
