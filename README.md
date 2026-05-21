# Change-Point Detection Experiments

Prototype and experiment scaffold for online local–global Wasserstein regime
filtering in financial changepoint detection. Design: `docs/experiment_plan.md`.

## Repository layout

```text
.
├── docs/                          # experiment plan, baseline citations
├── examples/
│   └── wpcg_demo.py               # standalone WPCG / Eq. (13) demo
├── requirements.txt
├── requirements-optional.txt
└── src/changept_detection/
    ├── __main__.py                # python -m changept_detection
    ├── method/                    # our method
    │   ├── wpcg.py                # offline WPCG coordinate sweep
    │   └── proposed.py            # proposed_full + ablations
    ├── baselines/
    │   └── core.py                # baseline registry + detectors
    └── experiments/
        ├── synthetic.py           # S0–S7 generators and sweeps
        ├── calibration.py         # null-sequence threshold calibration
        ├── runner.py              # CLI
        └── visualize.py           # result plots
```

## Run experiments

```bash
pip install -r requirements.txt
pip install -r requirements-optional.txt   # recommended

PYTHONPATH=src python -m changept_detection \
  --grid quick --seeds 5 --output-dir results --write-resources --plot
```

Options: `--experiments S0 S3`, `--no-calibrate`, `--null-seeds 20`, `--plot-only <stem>`.

Outputs under `results/` (gitignored): CSV, JSON, summary, `results/plots/<stem>/`.

## Methods

| Module | Role |
|--------|------|
| `method/wpcg.py` | Offline Eq. (13) segmentation (fixed # segments). |
| `method/proposed.py` | **`proposed_full`**: local alert + prototypes + global refinement. |
| `baselines/core.py` | OT, classical, kernel, CUSUM, BOCPD, HMM registry. |
| `experiments/synthetic.py` | Diagnostic suite S0–S7. |

`examples/wpcg_demo.py` is **not** the S0–S7 benchmark detector; use `proposed_full` there.

## Calibration

Default: thresholds from **null (no-change)** sequences (plan §5.1). Legacy: `--no-calibrate`.
