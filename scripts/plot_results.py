#!/usr/bin/env python3
"""
Replot metric charts (and optional detection timelines) from saved experiment results.

Usage (from repo root, with changept env active and matplotlib installed):

  # Auto-detect the only run folder under results/
  PYTHONPATH=src python scripts/plot_results.py

  # Explicit run stem (folder name under results/)
  PYTHONPATH=src python scripts/plot_results.py \\
    --stem seta_quick_A1-A2-A3-A4-A5-A6-A7-A_regime

  # Also regenerate detection timeline plots (re-runs detectors on representative cases)
  PYTHONPATH=src python scripts/plot_results.py --detections

Equivalent built-in CLI:
  PYTHONPATH=src python -m changept_detection \\
    --plot-only seta_quick_A1-A2-A3-A4-A5-A6-A7-A_regime \\
    --output-dir results --plot --plot-detections
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from changept_detection.experiments.runner import (  # noqa: E402
    load_rows_from_output,
    plots_detections_dir,
    plots_metrics_dir,
    print_output_tree,
    summarize,
)


def discover_stems(output_dir: Path) -> list[str]:
    """Return run folder names that contain metrics/results.csv or results.json."""
    stems: list[str] = []
    if not output_dir.is_dir():
        return stems
    for path in sorted(output_dir.iterdir()):
        if not path.is_dir():
            continue
        metrics = path / "metrics"
        if (metrics / "results.csv").exists() or (metrics / "results.json").exists():
            stems.append(path.name)
    return stems


def parse_stem(stem: str) -> tuple[str, list[str]]:
    """
    Parse seta_{grid}_{A1-A2-...} into (grid, experiments).

    Falls back to grid='quick' and experiments inferred from rows later.
    """
    match = re.match(r"seta_(quick|full)_(.+)$", stem)
    if not match:
        return "quick", []
    grid = match.group(1)
    experiments = match.group(2).split("-")
    return grid, experiments


def experiments_from_rows(rows: list[dict]) -> list[str]:
    seen: list[str] = []
    for row in rows:
        exp = row.get("experiment")
        if exp and exp not in seen:
            seen.append(exp)
    return seen


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot saved Set A experiment results from results/<stem>/metrics/."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results",
        help="Root directory containing run folders (default: results/).",
    )
    parser.add_argument(
        "--stem",
        default=None,
        help="Run folder name, e.g. seta_quick_A1-A2-A3-A4-A5-A6-A7-A_regime. "
        "Auto-detected when exactly one run exists.",
    )
    parser.add_argument(
        "--grid",
        choices=["quick", "full"],
        default=None,
        help="Grid label for plot titles (default: parsed from --stem, else quick).",
    )
    parser.add_argument(
        "--detections",
        action="store_true",
        help="Also generate detection timeline plots under plots/detections/ "
        "(re-runs detectors on one representative case per experiment).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = args.output_dir.resolve()

    stem = args.stem
    if stem is None:
        stems = discover_stems(output_dir)
        if not stems:
            raise SystemExit(f"No saved runs found under {output_dir}/")
        if len(stems) > 1:
            raise SystemExit(
                "Multiple runs found; pass --stem explicitly:\n  "
                + "\n  ".join(stems)
            )
        stem = stems[0]
        print(f"Auto-selected run: {stem}")

    rows = load_rows_from_output(output_dir, stem)
    summary = summarize(rows)

    grid_from_stem, experiments_from_stem = parse_stem(stem)
    grid = args.grid or grid_from_stem
    experiments = experiments_from_stem or experiments_from_rows(rows)

    from changept_detection.experiments.visualize import generate_all_plots, print_results_audit

    metrics_plot_dir = plots_metrics_dir(output_dir, stem)
    metric_paths = generate_all_plots(rows, summary, metrics_plot_dir, grid=grid)
    print_results_audit(rows, summary)
    print(f"\nMetric plots written to {metrics_plot_dir}/")
    for path in metric_paths:
        print(f"  {path.name}")

    if args.detections:
        from changept_detection.experiments.diagnostics import generate_detection_diagnostic_plots

        detections_plot_dir = plots_detections_dir(output_dir, stem)
        _, detection_paths = generate_detection_diagnostic_plots(
            experiments=experiments,
            grid=grid,
            seed=0,
            plot_dir=detections_plot_dir,
        )
        print(f"\nDetection plots written to {detections_plot_dir}/")
        for path in detection_paths:
            print(f"  {path.name}")

    print_output_tree(output_dir, stem)


if __name__ == "__main__":
    main()
