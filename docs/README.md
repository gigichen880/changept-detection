# Documentation

| File | Purpose |
|------|---------|
| [`experiment_plan.md`](experiment_plan.md) | Canonical experiment design (Sets A/B/C), methods, metrics, calibration |
| [`baseline_resources.md`](baseline_resources.md) | Paper/repo citations for each registered baseline |
| [`deep-research-report.md`](deep-research-report.md) | Background CPD literature survey (reference only) |

Implementation mapping for Set A synthetic runs:

- Experiment ids **S0–S7** → plan sections **A1–A7** (+ S7 regime labeling)
- Method lists → `src/changept_detection/experiments/spec.py`
- Detectors → `src/changept_detection/baselines/core.py`
- Proposed method slot → `src/changept_detection/method/placeholder.py`
