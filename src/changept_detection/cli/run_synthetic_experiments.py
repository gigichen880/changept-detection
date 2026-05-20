"""Command-line runner for the synthetic CPD experiment suite."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from changept_detection.baselines import resource_table
from changept_detection.synthetic import (
    BASELINE_SETS,
    EXPERIMENT_DESCRIPTIONS,
    flatten_result,
    run_synthetic_suite,
)


def json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, float) and np.isnan(value):
        return None
    return str(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run synthetic changepoint-detection experiments from docs/experiment_plan.md."
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=list(BASELINE_SETS),
        choices=list(BASELINE_SETS),
        help="Synthetic experiments to run.",
    )
    parser.add_argument(
        "--grid",
        choices=["quick", "full"],
        default="quick",
        help="Use quick smoke-test grids or full difficulty sweeps.",
    )
    parser.add_argument("--seeds", type=int, default=1, help="Number of random seeds per parameter setting.")
    parser.add_argument(
        "--baselines",
        nargs="+",
        default=None,
        help="Override the default baseline set with explicit baseline keys.",
    )
    parser.add_argument("--window", type=int, default=None, help="Override rolling window length.")
    parser.add_argument("--output-dir", default="results", help="Directory for CSV/JSON outputs.")
    parser.add_argument(
        "--write-resources",
        action="store_true",
        help="Also write baseline resource metadata to JSON.",
    )
    return parser.parse_args()


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=json_default) + "\n")


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
                }
            )
            continue
        f1_values = [float(row["f1"]) for row in available if "f1" in row and row["f1"] != ""]
        recall_values = [float(row["recall"]) for row in available if "recall" in row and row["recall"] != ""]
        precision_values = [float(row["precision"]) for row in available if "precision" in row and row["precision"] != ""]
        summary.append(
            {
                "experiment": experiment,
                "method": method,
                "available_runs": len(available),
                "mean_f1": float(np.nanmean(f1_values)) if f1_values else np.nan,
                "mean_recall": float(np.nanmean(recall_values)) if recall_values else np.nan,
                "mean_precision": float(np.nanmean(precision_values)) if precision_values else np.nan,
            }
        )
    return summary


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    results = run_synthetic_suite(
        experiments=args.experiments,
        grid=args.grid,
        seeds=args.seeds,
        baselines=args.baselines,
        window=args.window,
    )
    rows = [flatten_result(result) for result in results]
    stem = f"synthetic_{args.grid}_{'-'.join(args.experiments)}"
    csv_path = output_dir / f"{stem}.csv"
    json_path = output_dir / f"{stem}.json"
    summary_path = output_dir / f"{stem}_summary.json"
    write_csv(rows, csv_path)
    write_json(rows, json_path)
    summary = summarize(rows)
    write_json(summary, summary_path)
    if args.write_resources:
        write_json(resource_table(args.baselines), output_dir / "baseline_resources.json")

    print("Synthetic CPD experiments complete")
    print(f"Experiments: {', '.join(args.experiments)}")
    for experiment in args.experiments:
        print(f"  {experiment}: {EXPERIMENT_DESCRIPTIONS[experiment]}")
    print(f"Rows: {len(rows)}")
    print(f"CSV: {csv_path}")
    print(f"JSON: {json_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
