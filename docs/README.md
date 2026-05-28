# Documentation

| File | Purpose |
|------|---------|
| [`experiment_plan.md`](experiment_plan.md) | Canonical design: Sets A/B/C, methods, metrics |
| [`baseline_resources.md`](baseline_resources.md) | Citations for registered baselines |
| [`deep-research-report.md`](deep-research-report.md) | Background CPD literature (reference only) |

## Code mapping

| Plan | Runner id | Module |
|------|-----------|--------|
| Set A §A1–A7 | `A1` … `A7` | `experiments/synthetic.py` generators |
| §4.3 regime labels | `A_regime` | same + `method.proposed.regime_labels_from_prototypes` |
| Proposed method | `proposed_full` (primary) | `method/local_global_wasserstein.py` |
| Set B B1–B3 | — | not implemented |
| Set C C1–C3 | — | not implemented |

Spec (baselines, grids, metrics): `src/changept_detection/experiments/spec.py`
