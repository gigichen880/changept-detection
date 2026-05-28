# Change-Point Detection Experiment Framework

Plan-aligned scaffold for **Set A synthetic changepoint detection**
([`docs/experiment_plan.md`](docs/experiment_plan.md)). Experiment ids **match the plan**
(`A1`–`A7`). The proposed local–global Wasserstein method is implemented in
`method/local_global_wasserstein.py`.

## Repository layout

```text
changept-detection/
├── docs/
│   ├── experiment_plan.md       # canonical design (Sets A/B/C)
│   ├── baseline_resources.md    # baseline citations
│   └── README.md                # doc index
├── requirements.txt
├── requirements-optional.txt
└── src/changept_detection/
    ├── __main__.py              # python -m changept_detection
    ├── baselines/core.py        # detector registry (plan §2.3)
    ├── experiments/
    │   ├── spec.py              # A1–A7, A_regime, baselines, grids, metrics
    │   ├── synthetic.py         # DGP generators + run_case
    │   ├── calibration.py       # null-sequence thresholds (§3.1)
    │   ├── metrics.py           # CP-F1, A7 duplicate metrics, score audit
    │   ├── runner.py            # CLI
    │   └── visualize.py
    └── method/
        ├── local_global_wasserstein.py  # detector + run_proposed adapter
        ├── proposed.py                  # stable public import path
        ├── wasserstein_distances.py
        ├── prototype_layer.py
        └── global_refinement.py
```

## Quick start

```bash
pip install -r requirements.txt
pip install -r requirements-optional.txt

PYTHONPATH=src python -m changept_detection \
  --grid quick --seeds 2 --output-dir results --write-resources --plot
```

### CLI

| Flag | Example | Purpose |
|------|---------|---------|
| `--experiments` | `A1 A4 A_regime` | Subset (default: all Set A ids) |
| `--grid` | `quick` or `full` | Smoke vs full difficulty sweep |
| `--seeds` | `5` | Random seeds per DGP config |
| `--plot-only` | `seta_quick_A1-A7` | Replot existing results stem |
| `--plot-detections` | | True vs detected CP timelines per experiment |
| `--diagnostics-only` | | Skip suite; only run detection diagnostics |
| `--no-progress` | | Disable tqdm progress bar |

Output stem format: `seta_{grid}_{A1-A2-…}.csv` under `results/` (gitignored).

## Experiments (Set A)

| Id | Plan section | Purpose |
|----|--------------|---------|
| **A1** | A1 | Mean/variance sanity check |
| **A2** | A2 | Variance-matched tail shift |
| **A3** | A3 | Scenario-mixture weight shift |
| **A4** | A4 | Fixed-marginal correlation crisis |
| **A5** | A5 | Low-rank factor shock |
| **A6** | A6 | Transient vs persistent regime |
| **A7** | A7 | Duplicate local peak suppression |
| **A_regime** | §4.3 extension | Recurring-regime labeling (ARI/NMI) |

`A_regime` is the only id that is **not** a numbered plan section — it tests prototype/regime
metrics from §4.3. Sets **B** (B1–B3) and **C** (C1–C3) are in the plan but not implemented yet.

## Grids

Both grids vary DGP difficulty knobs; total work ≈ `(# configs) × seeds × methods`.

**`--grid quick`** — smoke / dev (~29 configs across all experiments):

| Id | # configs | Notes |
|----|-----------|-------|
| A1–A6 | 2 each | easy + harder corner |
| A7 | 16 | shift family × strength × window × noise |
| A_regime | 1 | fixed recurring-regime setting |

**`--grid full`** — plan difficulty tables (`synthetic.full_grid()`):

A1 **400**, A2 **48**, A3 **160**, A4 **72**, A5 **120**, A6 **90**, A7 **288**, A_regime **9**.

Start with subsets: `--experiments A2 A3 --grid full --seeds 3`.

## Proposed method

Three-layer local–global Wasserstein filter (design:
[`docs/proposed_method.md`](docs/proposed_method.md)).

**Main entry points**

| Use case | Import / call |
|----------|----------------|
| Experiments & baselines | `run_baseline("proposed_full", x, **kwargs)` → `method.proposed.run_proposed` |
| Direct detector API | `LocalGlobalWassersteinDetector(...).detect(x)` |
| Stable imports | `from changept_detection.method import run_proposed, LocalGlobalWassersteinDetector` |

Implementation lives in `method/local_global_wasserstein.py`; `method/proposed.py` is the
stable re-export layer used by the baseline registry.

Registry keys: `proposed_full` (primary), `proposed_local_only`,
`proposed_local_global_no_proto`, `proposed_local_proto_no_global`,
`proposed_local_persistence_proxy`.

## Protocol

- **Calibration** (§3.1): null-sequence thresholds, frozen before evaluation (`--null-seeds`, default 8).
- **Tolerance** (§3.2): `|τ̂ − τ*| ≤ w/2` via `spec.detection_tolerance()`.
