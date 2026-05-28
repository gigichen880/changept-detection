# Baseline Resources

Citations for detectors registered in `changept_detection.baselines.core.BASELINE_RESOURCES`.
Experiment design and per-suite method lists: `docs/experiment_plan.md` and
`changept_detection.experiments.spec`.

## Plan §2.3 compact pool

| Key | Category |
|-----|----------|
| `cusum_mean`, `cusum_vol` | Classical CPD |
| `pelt_rbf`, `binseg` | Classical offline CPD |
| `mmd`, `energy` | Distributional CPD |
| `coordinate_w2_matched_filter` | OT (local W2 + matched filter) |
| `sliced_wasserstein`, `bures` | OT multivariate / covariance |
| `bocpd_gaussian` | Online Bayesian CPD |
| `gaussian_hmm` | Online / regime model |

Additional keys used in specific Set A experiments (e.g. `ks`, `cvm`, `ewma_vol`,
`sinkhorn`, `covariance_frobenius`) are documented inline in `BASELINE_RESOURCES`.

## Proposed method

All `proposed_*` keys dispatch through `changept_detection.method.proposed.run_proposed`
(full implementation in `method/local_global_wasserstein.py`). Primary comparison key:
`proposed_full`. Design: `docs/proposed_method.md`.

## Optional dependencies

Install `requirements-optional.txt` for:

- `ruptures` → `pelt_*`, `binseg`, `bottomup`
- `POT` → `sinkhorn` (fallback implementation exists)
- `hmmlearn` → `gaussian_hmm`

See `docs/experiment_plan.md` §6 for the minimal first milestone scope.
