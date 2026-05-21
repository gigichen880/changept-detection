# Change-Point Detection Experiments

Prototype and experiment scaffold for online local–global Wasserstein regime
filtering in financial changepoint detection. The design follows
`docs/experiment_plan.md`.

## Why there is only one CLI entry point

Earlier versions had both `scripts/run_synthetic_experiments.py` (a thin wrapper
that edited `sys.path`) and `src/changept_detection/cli/run_synthetic_experiments.py`
(the real logic). That duplicated the same command in two places.

Now everything runs through the package:

```bash
PYTHONPATH=src python -m changept_detection [options]
```

Implementation lives in `src/changept_detection/runner.py`; plots in
`src/changept_detection/visualize.py`.

## Repository layout

```text
.
├── docs/
│   ├── experiment_plan.md
│   └── baseline_resources.md
├── requirements.txt
├── requirements-optional.txt
├── src/changept_detection/
│   ├── __main__.py          # python -m changept_detection
│   ├── baselines.py
│   ├── synthetic.py
│   ├── runner.py
│   └── visualize.py
└── wpcg_cpd.py              # standalone 1D-Wasserstein / WPCG prototype
```

## Run experiments and plots

```bash
pip install -r requirements.txt
# optional: pip install -r requirements-optional.txt

PYTHONPATH=src python -m changept_detection \
  --grid quick --seeds 1 --output-dir results --write-resources --plot
```

Selected experiments:

```bash
PYTHONPATH=src python -m changept_detection \
  --experiments S3 S4 --grid quick --seeds 3 --output-dir results --plot
```

Regenerate plots from saved CSV/JSON:

```bash
PYTHONPATH=src python -m changept_detection \
  --plot-only synthetic_quick_S0-S1-S2-S3-S4-S5-S6-S7 \
  --output-dir results --grid quick
```

## Outputs

| File | Contents |
|------|----------|
| `results/synthetic_<grid>_S0-....csv` | One row per (case, method) with metrics and citations |
| `results/synthetic_<grid>_S0-....json` | Same as CSV |
| `results/synthetic_<grid>_S0-...._summary.json` | Mean F1 / ARI / duplicate rate per (experiment, method) |
| `results/plots/<stem>/S0.png … S7.png` | Per-experiment bar charts (proposed in red) |
| `results/plots/<stem>/overview_S0-S7.png` | 2×4 panel overview |
| `results/plots/<stem>/proposed_vs_best.png` | Proposed minus best baseline per experiment |
| `results/plots/<stem>/recall_precision_scatter.png` | Trade-off across S0–S6 runs |

`results/` is gitignored.

## Interpreting results

The runner prints a short **results audit** when `--plot` is used. Expectations
from the experiment plan (quick grids are noisy with few seeds):

- **S0**: Classical / CUSUM / BOCPD often match or beat distributional methods on pure Gaussian shifts.
- **S1–S2**: Proposed and OT/kernel scans should stay competitive on tail and mixture shifts.
- **S3**: `coordinate_w2t` should be weak; `bures`, `sliced_wasserstein`, `mmd`, and proposed should do better.
- **S4**: Covariance/factor-aware distances should help in high dimension.
- **S5**: Transient shocks may trigger local methods; persistent shifts should be detected.
- **S6**: Lower **duplicate rate** is better; proposed global filter should reduce extras vs `local_w2t`.
- **S7**: Compare **ARI** / NMI (regime labels), not F1.

Rows with `unavailable=1` mean an optional package is missing (`ruptures`, `POT`, `hmmlearn`).

## Dependencies

```bash
pip install -r requirements.txt              # core + matplotlib
pip install -r requirements-optional.txt   # ruptures, POT, hmmlearn
```

## Development notes

`proposed_local_global` is a compact local Wasserstein + persistence + duplicate
suppression implementation for sweeps, not the full prototype posterior layer in
the paper plan.
