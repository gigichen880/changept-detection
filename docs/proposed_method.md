# Implementation Plan: Online Local-Global Wasserstein Regime Filtering

## 0. Goal

Implement the proposed change-point detection method as a modular detector that can plug into the existing experiment framework and be compared against existing baselines.

The method should implement the full three-layer pipeline:

1. **Local Wasserstein alert layer**
2. **Regime prototype posterior layer**
3. **Rolling global boundary refinement layer**

The main method used in baseline comparisons should be the **full pipeline**. The individual layers should also be exposed through ablation flags.

---

## 1. Scope

### In scope

Implement a reusable detector class that can:

- run on a full sequence in offline/backtest mode,
- simulate online updates step by step,
- compute local Wasserstein alert scores,
- compute posterior probabilities over regime prototypes,
- generate candidate boundaries from local alerts and posterior shifts,
- refine recent candidates into confirmed persistent boundaries,
- log all intermediate quantities for debugging and ablation studies.

### Out of scope for first version

Do **not** overbuild the first implementation.

Leave these as optional or placeholder features:

- advanced Wasserstein barycenter optimization,
- WPCG prototype updates,
- exact global dynamic programming if greedy refinement is easier first,
- fully adaptive online creation of new regimes,
- complex trading/risk-control downstream tasks.

The first working version should prioritize correctness, logging, and compatibility with the experiment framework.

---

## 2. Repository layout (implemented)

```text
src/changept_detection/method/
  local_global_wasserstein.py   # LocalGlobalWassersteinDetector + run_proposed adapter
  proposed.py                   # stable public import path (re-exports)
  wasserstein_distances.py      # D_W interface (§6)
  prototype_layer.py            # prototypes + posteriors (§7)
  global_refinement.py          # greedy refinement + persistence (§9–10)
tests/
  test_local_global_wasserstein.py
```

### Main entry points

| Layer | Symbol | Module |
|-------|--------|--------|
| Detector class | `LocalGlobalWassersteinDetector` | `local_global_wasserstein.py` |
| Experiment adapter | `run_proposed(key, x, **kwargs)` | `proposed.py` (re-export) |
| Baseline registry | `run_baseline("proposed_full", x, …)` | `baselines/core.py` → `PROPOSED_DISPATCH` |
| Package import | `from changept_detection.method import run_proposed` | `method/__init__.py` |

Primary comparison key: **`proposed_full`**. Ablation keys:
`proposed_local_only`, `proposed_local_global_no_proto`, `proposed_local_proto_no_global`,
`proposed_local_persistence_proxy`.

Optional registry aliases (`lgw`, `local_global_wasserstein`, `ours_full`) are not registered;
use `proposed_full` in the experiment framework.

## 3. Main Public API

Implement one main detector class.

```python
class LocalGlobalWassersteinDetector:
    def __init__(
        self,
        window_size: int,
        refinement_horizon: int,
        n_prototypes: int,
        distance_type: str = "sliced",
        local_threshold: float | None = None,
        posterior_threshold: float | None = None,
        boundary_penalty: float = 1.0,
        short_segment_penalty: float = 1.0,
        min_segment_length: int | None = None,
        temperature: float = 1.0,
        persistence: int = 3,
        prototype_init: str = "kmeans_windows",
        update_prototypes: bool = False,
        max_candidates: int | None = None,
        random_state: int = 42,
        ablation: str = "full",
    ):
        ...
```

### Required methods

```python
fit(X)
```

Initialize thresholds, prototypes, and any calibration state using a warm-up or training sequence.

```python
detect(X)
```

Run the detector over a full sequence and return all scores, candidates, confirmed boundaries, and regime posteriors.

```python
partial_fit(x_t)
```

Optional but recommended. Update the detector online with one new observation and return the current state.

```python
get_results()
```

Return all logged outputs as a dictionary or dataclass.

---

## 4. Input and Output Contract

### Input shape

The detector should accept:

```python
X: np.ndarray
```

Supported shapes:

```text
[T]       for univariate time series
[T, d]    for multivariate time series
```

Internally, convert univariate input to shape:

```text
[T, 1]
```

### Output dictionary

`detect(X)` should return:

```python
{
    "alert_scores": np.ndarray,              # shape [T]
    "posterior_shift_scores": np.ndarray,    # shape [T]
    "regime_posteriors": np.ndarray,         # shape [T, K]
    "regime_labels": np.ndarray,             # shape [T]
    "candidate_boundaries": list[int],
    "global_retained_boundaries": list[int],
    "confirmed_boundaries": list[int],
    "confirmed_labels": dict[int, int],
    "prototype_info": dict,
    "config": dict,
}
```

For times where there is not enough history, fill scores with `np.nan` and labels with `-1`.

---

## 5. Layer 1: Local Wasserstein Alert

### Window construction

At each valid time `t`, construct:

```text
reference window: X[t - 2w : t - w]
current window:   X[t - w  : t]
```

Using Python slicing, if `t` is the right endpoint, the first valid `t` is:

```text
t = 2 * window_size
```

### Local alert score

Compute:

```text
A_t = D_W(reference_window, current_window)
```

where `D_W` is selected by `distance_type`.

A local candidate is generated when:

```text
A_t > local_threshold
```

### Required implementation behavior

- If `local_threshold` is provided, use it directly.
- If `local_threshold` is `None`, support a simple calibration option:
  - historical quantile of alert scores from warm-up data, or
  - no-change bootstrap if the existing calibration module already supports this.
- Store every `A_t` in `alert_scores`.

---

## 6. Wasserstein Distance Interface

Implement a modular distance interface.

```python
def compute_distance(
    X_left: np.ndarray,
    X_right: np.ndarray,
    distance_type: str,
    **kwargs,
) -> float:
    ...
```

### Required distance types for first version

| `distance_type`    | First-version behavior                                                        |
| ------------------ | ----------------------------------------------------------------------------- |
| `"wasserstein_1d"` | Use scipy 1D Wasserstein per feature; average across features if multivariate |
| `"sliced"`         | Random projections + 1D Wasserstein; average over projections                 |
| `"bures"`          | Compare covariance matrices with Bures-Wasserstein distance                   |
| `"sinkhorn"`       | Optional; implement if POT is already available, otherwise placeholder        |
| `"projected"`      | Optional; PCA/factor projection then sliced or 1D Wasserstein                 |

### Practical first-version defaults

Use:

```text
distance_type = "sliced"
```

for general multivariate data.

Use:

```text
distance_type = "bures"
```

for covariance/factor-shock experiments.

### Notes

For `"bures"`:

1. Compute covariance matrices from each window.
2. Add ridge regularization for numerical stability:

```text
Sigma = sample_cov + ridge * I
```

3. Compute the Bures covariance distance.

If matrix square roots are numerically unstable, use eigenvalue clipping.

---

## 7. Layer 2: Regime Prototype Posterior

### Prototype representation

For first version, store each prototype as one of:

```text
Option A: a representative rolling window
Option B: a covariance matrix
Option C: a feature vector summarizing a window
```

Recommended first version:

- For `"wasserstein_1d"` and `"sliced"`, store representative windows.
- For `"bures"`, store covariance matrices.
- For `"projected"`, store projected/covariance representations.

### Prototype initialization

Support at least:

```python
prototype_init = "random_windows"
prototype_init = "kmeans_windows"
```

Recommended default:

```text
"kmeans_windows"
```

Implementation:

1. Build rolling windows from the warm-up/training period.
2. Convert each window into a feature vector.
3. Cluster feature vectors into `K` clusters.
4. Select one representative window per cluster as the prototype.

Simple window feature vector:

```text
mean per feature
std per feature
skew/kurtosis if easy
flattened upper triangle of correlation/covariance for small d
```

If this is too expensive, start with mean/std/correlation summary only.

### Posterior computation

For each current window, compute distance to each prototype:

```text
d_k = D_W(current_window, prototype_k)
```

Convert distances into soft probabilities:

```text
score_k = exp(-d_k / temperature)
pi_t(k) = score_k / sum_j score_j
```

Numerical stability:

```python
scores = np.exp(-(distances - distances.min()) / temperature)
posterior = scores / scores.sum()
```

### Posterior shift score

Compute:

```text
B_t = L1 distance between pi_t and pi_{t - w}
```

If `pi_{t-w}` is unavailable, use `np.nan`.

A posterior-shift candidate is generated when:

```text
B_t > posterior_threshold
```

If `posterior_threshold` is `None`, either disable this trigger or calibrate it from warm-up data.

---

## 8. Candidate Generation

At each valid time `t`, generate candidate flags.

### Full method

Add `t` to the candidate set if:

```text
local alert is high
OR
posterior shift is high
```

In code:

```python
is_local_candidate = A_t > local_threshold
is_posterior_candidate = B_t > posterior_threshold
is_candidate = is_local_candidate or is_posterior_candidate
```

### Ablation modes

Support these ablations:

| `ablation`                    | Candidate logic                                        |
| ----------------------------- | ------------------------------------------------------ |
| `"full"`                      | local alert OR posterior shift, then global refinement |
| `"local_only"`                | local alert only; no global refinement                 |
| `"local_global_no_prototype"` | local alert only, then global refinement               |
| `"local_prototype_no_global"` | local alert OR posterior shift; no global refinement   |
| `"coordinate_w"`              | full pipeline but force coordinate-wise/1D Wasserstein |
| `"bures"`                     | full pipeline but force Bures distance                 |
| `"sliced"`                    | full pipeline but force sliced Wasserstein             |

---

## 9. Layer 3: Rolling Global Boundary Refinement

The global refinement layer should suppress duplicate peaks and reject short transient shocks.

### Candidate horizon

At time `t`, keep only candidates inside:

```text
[t - refinement_horizon, t]
```

Optional: if there are too many candidates, keep only the top `max_candidates` by local score.

### Segment score

Given a sorted candidate subset:

```text
tau = [c_1, c_2, ..., c_m]
```

form induced segments over the recent horizon. Score the segmentation by:

```text
global_score =
    adjacent_segment_separation
    - boundary_penalty * number_of_boundaries
    - short_segment_penalty * short_segment_cost
```

Where:

```text
adjacent_segment_separation =
    sum of D_W(segment_i, segment_{i+1}) over neighboring segments
```

and:

```text
short_segment_cost =
    sum over segments of max(0, min_segment_length - segment_length) / min_segment_length
```

### First-version implementation

Use a greedy refinement first.

Greedy algorithm:

```text
Input: recent candidates sorted by descending local evidence

Start with empty retained set.

For each candidate:
  1. Tentatively add the candidate.
  2. Reject if it creates a segment shorter than min_segment_length.
  3. Compute new global score.
  4. Keep the candidate only if score improves.
  5. If two retained candidates are closer than min_segment_length or window_size / 2,
     keep the one with stronger local evidence.

Return retained candidates.
```

This is enough for the first version.

### Optional second version

Replace greedy with exact dynamic programming over candidate subsets if needed.

---

## 10. Persistence Confirmation

A candidate should not become a confirmed boundary immediately unless desired.

Maintain a counter:

```python
retained_counter[candidate_time] += 1
```

A candidate is confirmed when:

```text
it has been retained for at least `persistence` consecutive updates
```

Because rolling refinement may shift a boundary by a few timestamps, use a tolerance window:

```text
same boundary if abs(new_candidate - existing_candidate) <= merge_tolerance
```

Recommended default:

```text
merge_tolerance = window_size // 2
```

If several retained candidates fall into the same tolerance window, keep the one with the highest local alert score.

---

## 11. Regime Label Assignment

For each time `t`, assign:

```text
regime_label_t = argmax_k pi_t(k)
```

For each confirmed boundary `tau`, assign the post-boundary regime using the posterior shortly after the boundary:

```text
confirmed_labels[tau] = argmax_k average pi_s(k) for s in [tau, tau + w]
```

If future data is unavailable in online mode, assign using the current posterior when the boundary is confirmed.

---

## 12. Prototype Updates

Prototype updates should be optional in the first version.

### Default

```python
update_prototypes = False
```

Use fixed prototypes initialized during `fit`.

### Optional simple update

If `update_prototypes = True`, periodically update prototypes using assigned recent windows.

First-version update can be simple:

```text
For each prototype k:
  collect windows assigned to k
  choose medoid window as the new prototype
```

Medoid means the assigned window with the smallest average distance to other assigned windows.

Do not implement full Wasserstein barycenters or WPCG in the first version unless everything else is done.

---

## 13. Threshold Calibration

If the experiment framework already has calibration, reuse it.

Otherwise implement a simple method:

### Local threshold

From warm-up or no-change calibration scores:

```text
local_threshold = quantile(alert_scores, 1 - alpha)
```

Example:

```text
alpha = 0.01
```

### Posterior threshold

From warm-up posterior shift scores:

```text
posterior_threshold = quantile(posterior_shift_scores, 1 - alpha)
```

### Important

Calibration should be fit only on training/no-change data and then frozen during evaluation.

---

## 14. Logging Requirements

Log enough information to debug every layer.

At each valid time `t`, store:

```python
{
    "t": int,
    "alert_score": float,
    "posterior_shift_score": float,
    "regime_posterior": np.ndarray,
    "regime_label": int,
    "is_local_candidate": bool,
    "is_posterior_candidate": bool,
    "is_candidate": bool,
    "is_retained_by_global": bool,
    "is_confirmed": bool,
}
```

This is necessary for:

- plotting local scores,
- checking posterior behavior,
- debugging duplicate suppression,
- evaluating ablations,
- explaining detections in real financial experiments.

---

## 15. Compatibility with Existing Experiment Framework

The detector should expose the same kind of output as existing baselines.

At minimum, the framework should be able to read:

```python
confirmed_boundaries
alert_scores
```

If the framework expects a standard object, add an adapter.

Example adapter output:

```python
DetectionResult(
    change_points=confirmed_boundaries,
    scores=alert_scores,
    metadata={
        "candidate_boundaries": candidate_boundaries,
        "regime_posteriors": regime_posteriors,
        "regime_labels": regime_labels,
        "posterior_shift_scores": posterior_shift_scores,
    },
)
```

The headline comparison against baselines should use:

```text
confirmed_boundaries
```

not raw local candidates.

Raw candidates should only be used for ablation or diagnostic plots.

---

## 16. Tests to Add

### Unit tests

1. **Window construction**
   - Verify reference/current windows are correctly sliced.

2. **Distance functions**
   - Distance is nonnegative.
   - Distance between identical windows is near zero.
   - Bures distance is stable with ridge regularization.

3. **Posterior computation**
   - Posterior sums to one.
   - Nearest prototype gets highest probability.

4. **Candidate generation**
   - Local threshold trigger works.
   - Posterior-shift trigger works.

5. **Global refinement**
   - Rejects candidates that create too-short segments.
   - Suppresses duplicate nearby candidates.
   - Keeps stronger candidate among nearby duplicates.

6. **Persistence**
   - Boundary is not confirmed before `persistence` updates.
   - Boundary is confirmed after enough repeated retention.

### Smoke tests

Run the full detector on:

1. Simple mean-shift synthetic data.
2. Tail-shift synthetic data.
3. Correlation-shift synthetic data.
4. Transient shock sequence.

Expected smoke-test behavior:

- mean shift: detects at least one boundary near the true point,
- tail shift: local score rises near the true point,
- correlation shift: Bures or sliced distance should produce a signal,
- transient shock: full method should be less likely than local-only to confirm it.

---

## 17. First Implementation Milestone

Build this first:

1. Main detector class.
2. Window builder.
3. Distance interface with:
   - 1D Wasserstein,
   - sliced Wasserstein,
   - Bures-Wasserstein.
4. Prototype initialization using random windows and k-means windows.
5. Posterior computation.
6. Candidate generation.
7. Greedy global refinement.
8. Persistence confirmation.
9. Complete logging.
10. Basic tests.

Do **not** implement WPCG or exact barycenter updates in milestone 1.

---

## 18. Suggested Default Configuration

Use this as the first default config:

```yaml
method: local_global_wasserstein

window_size: 50
refinement_horizon: 250
n_prototypes: 4

distance_type: sliced
n_projections: 100

local_threshold: null
posterior_threshold: null
threshold_alpha: 0.01

boundary_penalty: 1.0
short_segment_penalty: 1.0
min_segment_length: 50

temperature: 1.0
persistence: 3
merge_tolerance: 25

prototype_init: kmeans_windows
update_prototypes: false

ablation: full
random_state: 42
```

For covariance/factor-shock experiments, use:

```yaml
distance_type: bures
```

---

## 19. Acceptance Criteria

The implementation is complete when:

- the method can run end-to-end on a synthetic sequence,
- it returns confirmed boundaries and scores in the same format as baselines,
- it supports the main ablations,
- it logs local scores, posterior scores, posteriors, candidates, retained boundaries, and confirmed boundaries,
- it can be used in the existing experiment framework without modifying baseline code,
- it passes unit tests and smoke tests.

---

## 20. Notes for the Coding Agent

Please implement the proposed method only. The experiment framework and baselines are already done.

Prioritize:

1. simple, correct implementation,
2. clean interfaces,
3. full logging,
4. ablation support,
5. compatibility with the existing experiment runner.

Avoid overengineering the first version. The global refinement can be greedy first. Prototype updates can be fixed or medoid-based first. WPCG/barycenter updates can be added later.
