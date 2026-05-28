# Proposed Method Summary: Online Local–Global Wasserstein Regime Filtering

## 1. One-Sentence Summary

We propose an **online local–global Wasserstein regime filtering** method for financial change-point detection: local Wasserstein scores generate real-time candidate alerts, a regime-prototype layer assigns each window to interpretable market regimes, and a rolling global refinement layer confirms only persistent, coherent boundaries.

---

## 2. Motivation

Financial regime shifts are often not simple mean or variance changes. They may appear as:

- heavier tails,
- scenario-probability reweighting,
- correlation breakdowns,
- low-rank factor shocks,
- volatility or liquidity regime changes.

Classical change-point detection methods can work well for clean mean or variance shifts, but they may miss distributional or joint changes. Prior optimal-transport CPD methods use Wasserstein distances mainly as local two-sample statistics between adjacent windows. Our method keeps this useful local signal but adds **regime interpretation** and **global persistence filtering**.

---

## 3. Core Idea

At each time step, the method asks three questions:

1. **Did the current distribution change from the recent past?**  
   Answered by a local Wasserstein alert score.

2. **What regime does the current market window resemble?**  
   Answered by a posterior over Wasserstein regime prototypes.

3. **Is this change persistent enough to confirm as a regime boundary?**  
   Answered by rolling global refinement over recent candidate boundaries.

The output is richer than a binary alert. It returns:

- local alert score,
- candidate change points,
- confirmed persistent boundaries,
- posterior probabilities over regimes,
- historical regime prototypes or analogs.

---

## 4. Method Architecture

The method has three layers.

```text
Market time series
        |
        v
Rolling empirical windows
        |
        v
[1] Local Wasserstein alert layer
        |
        +--> candidate alerts
        |
        v
[2] Regime prototype posterior layer
        |
        +--> regime probabilities and posterior shifts
        |
        v
[3] Rolling global refinement layer
        |
        v
Confirmed persistent change points + regime labels
```

---

## 5. Layer 1: Local Wasserstein Alert

For each time `t`, define two rolling empirical distributions:

```text
reference window: observations from t - 2w + 1 to t - w
current window:   observations from t - w + 1 to t
```

The local alert score is:

```text
A_t = WassersteinDistance(reference_window, current_window)
```

A candidate change point is proposed when:

```text
A_t > local_threshold
```

The Wasserstein distance can be chosen according to the financial object being monitored:

| Data Type                            | Recommended Distance                    |
| ------------------------------------ | --------------------------------------- |
| Scalar returns or tail-risk features | 1D Wasserstein                          |
| Multivariate empirical windows       | Sliced Wasserstein or Sinkhorn          |
| Covariance or factor regimes         | Bures-Wasserstein                       |
| High-dimensional low-rank regimes    | Projected or factor-aligned Wasserstein |

This layer is designed to be high-recall: it should catch many possible changes, even if some are noisy.

---

## 6. Layer 2: Regime Prototype Posterior

Assume there are `K` latent regime prototypes:

```text
nu_1, nu_2, ..., nu_K
```

Each prototype represents a recurring market state, such as:

- calm expansion,
- equity stress,
- liquidity shock,
- high-correlation crisis,
- rates/inflation shock,
- rebound regime.

For the current window, compute its distance to each prototype:

```text
distance_to_regime_k = WassersteinDistance(current_window, nu_k)
```

Convert distances into soft regime probabilities:

```text
pi_t(k) proportional to exp(-distance_to_regime_k / temperature)
```

Interpretation:

- `pi_t(k)` is the probability that the current market window belongs to regime `k`.
- A sharp posterior means confident regime assignment.
- A large posterior shift means the market may be transitioning between regimes.

A posterior-shift candidate can be generated when:

```text
distance(pi_t, pi_{t-w}) > posterior_shift_threshold
```

This layer answers not only **whether a change happened**, but also **what kind of regime the market is entering**.

---

## 7. Layer 3: Rolling Global Boundary Refinement

Local rolling-window scans often create two problems:

1. **Duplicate peaks:** one true change can create many nearby local alerts.
2. **Transient shocks:** a one-day or few-day event can create a large local spike, even if it is not a persistent regime.

To fix this, we keep a recent candidate set over a rolling horizon:

```text
candidate_set_t = local and posterior-shift candidates inside [t - H, t]
```

Then we solve a regularized selection problem:

```text
choose a subset of recent candidates that:
  - maximizes distributional separation between neighboring segments,
  - penalizes too many boundaries,
  - penalizes very short segments.
```

Conceptually:

```text
global_score =
  segment_separation
  - boundary_penalty
  - short_segment_penalty
```

A candidate is confirmed only if it is retained by the global refinement step and remains stable for several update steps.

This layer turns noisy local alerts into coherent regime boundaries.

---

## 8. Prototype Updates

After confirmed boundaries split the history into segments, each segment can be assigned to the nearest prototype:

```text
segment_label = nearest Wasserstein prototype
```

Prototypes can then be updated using a Wasserstein barycenter-style step:

```text
new prototype = Wasserstein average of assigned segment distributions
```

This allows the regime dictionary to evolve over time while still preserving interpretable regime identities.

---

## 9. Full Online Algorithm

```text
Input:
  time series X_1, ..., X_T
  rolling window length w
  refinement horizon H
  number of prototypes K
  local threshold
  posterior-shift threshold
  global penalties

Initialize:
  regime prototypes using historical windows or warm-up clustering
  empty candidate set

For each time t:
  1. Build reference and current rolling windows.
  2. Compute local Wasserstein alert score A_t.
  3. Compute posterior over regime prototypes pi_t.
  4. Compute posterior-shift score.
  5. Add t to candidate set if either:
       A_t is large, or
       posterior shift is large.
  6. Restrict candidates to the recent horizon [t - H, t].
  7. Run rolling global refinement over recent candidates.
  8. Confirm a change only if it is retained and persistent.
  9. Periodically update regime prototypes.

Output:
  local alert scores
  candidate boundaries
  confirmed boundaries
  regime posteriors
  regime prototype assignments
```

---

## 10. Why Wasserstein?

Wasserstein distance is useful because it compares full distributions and accounts for how far probability mass moves.

This is especially relevant for finance:

| Regime Change Type     | Why Wasserstein Helps                                                            |
| ---------------------- | -------------------------------------------------------------------------------- |
| Tail shift             | Captures quantile displacement across the distribution                           |
| Scenario mixture shift | Measures both how much probability mass moves and how far it moves               |
| Correlation crisis     | Full or Bures-Wasserstein can detect joint dependence changes                    |
| Low-rank factor shock  | Bures/projected Wasserstein can focus on covariance or factor structure          |
| Regime recurrence      | Wasserstein distances provide a geometry for clustering and prototype assignment |

The claim is not that Wasserstein always wins. The claim is that Wasserstein provides a natural geometry for distributional financial regimes.

---

## 11. Relationship to Prior OT-CPD

| Component             | Prior Local OT-CPD                 | Our Proposed Method                                          |
| --------------------- | ---------------------------------- | ------------------------------------------------------------ |
| Detection statistic   | Local Wasserstein two-sample score | Local Wasserstein alert plus posterior shift                 |
| Boundary selection    | Local peaks or matched filtering   | Rolling global regularized refinement                        |
| Regime information    | Cluster segments after detection   | Online posterior over regime prototypes                      |
| Multivariate handling | Often coordinate-wise              | Bures, Sinkhorn, sliced, or projected variants               |
| Finance output        | Change-point locations             | Change points, regime labels, confidence, historical analogs |

The key difference is that we use local Wasserstein scores as **candidate generators**, not final decisions. The final decision is made through global persistence and regime-prototype consistency.

---

## 12. What the Method Is Designed to Improve

The method is designed to improve:

- detection of distributional regime changes,
- detection of dependence and factor-regime shifts,
- suppression of duplicate local peaks,
- rejection of transient shocks as persistent regimes,
- interpretability through regime posteriors,
- downstream financial use cases such as covariance forecasting, volatility monitoring, and portfolio risk control.

---

## 13. Main Ablations

To show which parts matter, compare:

| Ablation                     | Question Answered                                               |
| ---------------------------- | --------------------------------------------------------------- |
| Local-only Wasserstein       | Is the global layer necessary?                                  |
| Local + global, no prototype | Does global refinement improve boundary quality?                |
| Local + prototype, no global | Does the posterior help but still need denoising?               |
| Full method                  | Do all layers together improve robustness and interpretability? |
| Coordinate-wise Wasserstein  | Does joint geometry matter?                                     |
| Bures vs sliced vs Sinkhorn  | Which Wasserstein variant works best for each regime type?      |

---

## 14. Expected Strengths and Limitations

### Strengths

- Captures distributional changes beyond mean and variance.
- Handles joint dependence and covariance/factor shifts.
- Produces interpretable regime assignments.
- Reduces noisy local detections through global refinement.
- Naturally supports online monitoring.

### Limitations

- Requires choices of window length, horizon, thresholds, and penalties.
- Wasserstein computations can be expensive in high dimensions.
- No single Wasserstein variant is best for all settings.
- Real financial changepoints are not objectively labeled, so real-data evaluation must use event windows and downstream utility.
- Prototype count `K` may require tuning or stability analysis.

---

## 15. Final Positioning

The proposed method should be positioned as a **regime-filtering framework**, not just a new distance-based detector.

A defensible claim is:

> The method is not intended to replace classical CPD on simple mean or variance shifts. Instead, it targets finance-relevant distributional regime changes, including tail shifts, scenario-probability shifts, dependence breakdowns, and low-rank factor shocks. Its contribution is the combination of Wasserstein local alerts, global persistence filtering, and interpretable regime posteriors.

This framing avoids overclaiming and makes the method’s purpose clear.
