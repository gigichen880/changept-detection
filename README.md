# Change-Point Detection Experiment Framework

Plan-aligned scaffold for **Set A synthetic changepoint detection**
([`docs/experiment_plan.md`](docs/experiment_plan.md)). Baselines, null calibration,
metrics, and plots are wired; the proposed local–global Wasserstein method is a
**placeholder** until you plug in the real detector.

## Repository layout

```text
changept-detection/
├── docs/
│   ├── experiment_plan.md       # canonical experiment design (Set A/B/C)
│   ├── baseline_resources.md    # baseline citations
│   └── deep-research-report.md  # background literature notes (not used by runner)
├── requirements.txt             # numpy, scipy, sklearn, matplotlib
├── requirements-optional.txt    # ruptures, POT, hmmlearn
└── src/changept_detection/
    ├── __main__.py              # python -m changept_detection
    ├── baselines/
    │   └── core.py              # detector registry + metrics (plan §2.3)
    ├── experiments/
    │   ├── spec.py              # S0–S7 ↔ plan A1–A7, method lists, primary metrics
    │   ├── synthetic.py         # DGP generators + run_case
    │   ├── calibration.py       # null-sequence thresholds (plan §3.1)
    │   ├── metrics.py           # CP-F1, S6 duplicate metrics, score audit
    │   ├── runner.py            # CLI entry
    │   └── visualize.py         # bar charts, overview, audit prints
    └── method/
        ├── placeholder.py       # ← implement run_proposed() here
        └── proposed.py          # stable re-export of placeholder
```

Generated artifacts go to `results/` (gitignored). Do not commit CSV/JSON/plots.

## Quick start

From the repo root:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-optional.txt   # PELT, Sinkhorn, HMM baselines

PYTHONPATH=src python -m changept_detection \
  --grid quick --seeds 2 --output-dir results --write-resources --plot
```

### Common CLI flags

| Flag | Purpose |
|------|---------|
| `--experiments S0 S3 S7` | Subset of Set A (default: all S0–S7) |
| `--grid quick\|full` | Smoke grid vs full difficulty sweep |
| `--seeds N` | Random seeds per parameter setting |
| `--null-seeds N` | Null sequences for calibration (default 8) |
| `--no-calibrate` | Legacy in-sample thresholds (not plan §3.1) |
| `--plot-only STEM` | Replot from existing `results/STEM.{csv,json}` |

### Outputs

| File | Contents |
|------|----------|
| `results/synthetic_quick_S0-…-S7.csv` | Per-run metrics + calibration fields |
| `results/synthetic_quick_…_summary.json` | Aggregated means per experiment × method |
| `results/synthetic_quick_…_score_audit.csv` | Threshold vs max-score diagnostic |
| `results/plots/synthetic_quick_…/` | Per-experiment bars, overview, proposed vs best |

## Plug in the proposed method

1. Edit `src/changept_detection/method/placeholder.py`.
2. Replace `run_proposed()` (and `regime_labels_from_prototypes()` for S7 regime metrics).
3. Return `DetectionResult(changepoints, scores, threshold, metadata)` from
   `changept_detection.baselines.core`.
4. Re-run the CLI — runner, calibration, and plots stay unchanged.

Registered keys (plan §2.2): `proposed_local_only`, `proposed_local_persistence_proxy`,
`proposed_local_global_no_proto`, `proposed_local_proto_no_global`, `proposed_full`.

## Set A experiment map

| Id | Plan | Purpose |
|----|------|---------|
| S0 | A1 | Mean/variance sanity check |
| S1 | A2 | Variance-matched tail shift |
| S2 | A3 | Scenario-mixture weight shift |
| S3 | A4 | Fixed-marginal correlation crisis |
| S4 | A5 | Low-rank factor shock |
| S5 | A6 | Transient vs persistent regime |
| S6 | A7 | Duplicate local peak suppression |
| S7 | — | Recurring-regime labeling (ARI/NMI) |

Per-experiment baseline lists and primary metrics: `src/changept_detection/experiments/spec.py`.

## Protocol notes

- **Calibration** (plan §3.1): thresholds from matched no-change nulls, frozen before
  evaluation. Tune via `--null-seeds` and `--false-alarm-quantile` (default 0.95).
- **Detection tolerance** (plan §3.2): `|τ̂ − τ*| ≤ w/2` via `spec.detection_tolerance()`.
- **Optional deps**: without `ruptures` / `POT` / `hmmlearn`, affected baselines appear as
  `unavailable=1` rows; core window-scan methods still run.

## Documentation

- Experiment design: [`docs/experiment_plan.md`](docs/experiment_plan.md)
- Baseline citations: [`docs/baseline_resources.md`](docs/baseline_resources.md)
