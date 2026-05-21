# Change-Point Detection Experiments

This repository contains a prototype and experiment scaffold for online
local-global Wasserstein regime filtering in financial changepoint detection.
The experiment design is driven by `docs/experiment_plan.md`.

## Repository Layout

```text
.
├── docs/
│   ├── experiment_plan.md          # Research questions, experiments, baselines
│   └── baseline_resources.md       # Paper/repo references for each baseline
├── scripts/
│   └── run_synthetic_experiments.py
├── src/changept_detection/
│   ├── baselines.py                # Baseline registry, detectors, metrics
│   ├── synthetic.py                # S0-S7 generators and sweep orchestration
│   └── cli/
│       └── run_synthetic_experiments.py
└── wpcg_cpd.py                     # Original standalone WPCG/1D-Wasserstein prototype
```

## Baselines

The implemented baseline registry is in `src/changept_detection/baselines.py`.
Each baseline has a citation/resource entry pointing to the original paper,
canonical documentation, or reference repository. The resource metadata is also
written into experiment outputs so result rows remain traceable.

Core baselines include:

- OT scans: local W2T, coordinate-wise W2T, sliced Wasserstein, Bures-Wasserstein,
  Sinkhorn when POT is installed, and a WATCH-style monitor proxy.
- Classical CPD: PELT, binary segmentation, and bottom-up segmentation through
  optional `ruptures`.
- Distributional tests: MMD, energy distance, KS, Cramer-von Mises, and a
  density-ratio proxy.
- Online/finance baselines: CUSUM, EWMA volatility, Gaussian BOCPD, optional
  Gaussian HMM, and registered Markov-switching references.

See `docs/baseline_resources.md` for the source list.

## Synthetic Experiments

`src/changept_detection/synthetic.py` implements the synthetic suite from
`docs/experiment_plan.md`:

- `S0`: Gaussian mean/variance sanity checks
- `S1`: variance-matched tail shifts
- `S2`: scenario-mixture weight shifts
- `S3`: fixed-marginal correlation crises
- `S4`: low-rank factor shocks
- `S5`: transient shock versus persistent regime
- `S6`: duplicate local peak suppression
- `S7`: recurring-regime interpretability

Each experiment has a small `quick` grid for smoke tests and a broader `full`
grid for difficulty sweeps.

## Running Experiments

From a source checkout, use the wrapper:

```bash
python scripts/run_synthetic_experiments.py --grid quick --seeds 1 --output-dir results --write-resources
```

Or run the package module with `PYTHONPATH`:

```bash
PYTHONPATH=src python -m changept_detection.cli.run_synthetic_experiments --grid quick --seeds 1 --output-dir results --write-resources
```

Run selected experiments:

```bash
python scripts/run_synthetic_experiments.py --experiments S1 S3 S4 --grid quick --seeds 3 --output-dir results
```

Run a fuller sweep:

```bash
python scripts/run_synthetic_experiments.py --grid full --seeds 10 --output-dir results/full --write-resources
```

Outputs are written as CSV, JSON, and summary JSON. `results/` is ignored by git.

## Dependencies

Install core dependencies:

```bash
pip install -r requirements.txt
```

| Package          | Role                                                |
| ---------------- | --------------------------------------------------- |
| `numpy`, `scipy` | Data generation, distances, stats tests, BOCPD-lite |
| `scikit-learn`   | Clustering metrics, density-ratio proxy             |
| `ruptures`       | PELT, binary segmentation, bottom-up                |
| `POT`            | Sinkhorn window scan                                |
| `hmmlearn`       | Gaussian HMM regime baseline                        |

If an optional dependency is missing, that baseline returns an unavailable row
with the relevant resource citation instead of failing the whole run.

## Development Notes

The current `proposed_local_global` entry is a compact experiment implementation
of local Wasserstein alerts with persistence and duplicate suppression. It is
intended to exercise the synthetic claims while the full prototype/posterior
layer is developed further.
