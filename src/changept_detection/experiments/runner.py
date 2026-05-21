"""Run synthetic CPD experiments and write tabular outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from changept_detection.baselines.core import resource_table
from changept_detection.experiments.synthetic import (
    BASELINE_SETS,
    EXPERIMENT_DESCRIPTIONS,
    flatten_result,
    run_synthetic_suite,
)


def json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return None
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def sanitize_nans(obj: Any) -> Any:
    """Recursively replace NaN/Inf with null for strict JSON."""
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
        description="Run synthetic changepoint-detection experiments from docs/experiment_plan.md."
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=list(BASELINE_SETS),
        choices=list(BASELINE_SETS),
        help="Synthetic experiments to run (S0-S7).",
    )
    parser.add_argument(
        "--grid",
        choices=["quick", "full"],
        default="quick",
        help="Quick smoke grids or full difficulty sweeps.",
    )
    parser.add_argument("--seeds", type=int, default=1, help="Random seeds per parameter setting.")
    parser.add_argument("--baselines", nargs="+", default=None, help="Override default baseline keys.")
    parser.add_argument("--window", type=int, default=None, help="Rolling window length override.")
    parser.add_argument("--output-dir", default="results", help="Directory for CSV/JSON/plots.")
    parser.add_argument("--write-resources", action="store_true", help="Write baseline_resources.json.")
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate comparison plots after the run (requires matplotlib).",
    )
    parser.add_argument(
        "--plot-only",
        metavar="STEM",
        default=None,
        help="Skip experiments; plot from existing results/<STEM>.{csv,json} in --output-dir.",
    )
    parser.add_argument(
        "--no-calibrate",
        action="store_true",
        help="Use in-sample quantile thresholds (legacy; not plan-faithful).",
    )
    parser.add_argument("--null-seeds", type=int, default=15, help="Null sequences per experiment for calibration.")
    parser.add_argument(
        "--false-alarm-quantile",
        type=float,
        default=0.95,
        help="Quantile of null max-scores for threshold (0.95 ~ 5%% null FP rate).",
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
            }
        )
    return summary


def result_stem(grid: str, experiments: list[str]) -> str:
    return f"synthetic_{grid}_{'-'.join(experiments)}"


def load_rows_from_output(output_dir: Path, stem: str) -> list[dict[str, Any]]:
    json_path = output_dir / f"{stem}.json"
    if json_path.exists():
        return json.loads(json_path.read_text())
    csv_path = output_dir / f"{stem}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"No results found for stem '{stem}' in {output_dir}")
    with csv_path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    stem = result_stem(args.grid, args.experiments)

    if args.plot_only:
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
        )
        rows = [flatten_result(result) for result in results]
        summary = summarize(rows)
        write_csv(rows, output_dir / f"{stem}.csv")
        write_json(rows, output_dir / f"{stem}.json")
        write_json(summary, output_dir / f"{stem}_summary.json")
        if args.write_resources:
            write_json(resource_table(args.baselines), output_dir / "baseline_resources.json")

        print("Synthetic CPD experiments complete")
        print(f"Experiments: {', '.join(args.experiments)}")
        for experiment in args.experiments:
            print(f"  {experiment}: {EXPERIMENT_DESCRIPTIONS[experiment]}")
        print(f"Rows: {len(rows)}")
        print(f"CSV: {output_dir / f'{stem}.csv'}")
        print(f"JSON: {output_dir / f'{stem}.json'}")
        print(f"Summary: {output_dir / f'{stem}_summary.json'}")

    if args.plot or args.plot_only:
        from changept_detection.experiments.visualize import generate_all_plots, print_results_audit

        plot_dir = output_dir / "plots" / stem
        paths = generate_all_plots(rows, summary, plot_dir, grid=args.grid)
        print_results_audit(rows, summary)
        print(f"Plots: {plot_dir}")
        for path in paths:
            print(f"  {path}")
