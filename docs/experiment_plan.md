# Experiment Plan: Online Local–Global Wasserstein Regime Filtering for Financial Change-Point Detection

## 0. Goal

This experiment suite evaluates an **online local–global Wasserstein regime filtering** method for financial change-point detection.

We do **not** claim that Wasserstein methods always outperform classical CPD. In simple mean or variance shifts, classical methods may be equally strong or better.

Our target claim is narrower:

> The proposed method is useful when financial regime changes are distributional, joint, tail-driven, covariance/factor-driven, or persistent rather than transient.

The experiments test three questions:

1. **Detection:** Can the method detect finance-relevant distributional regime changes?
2. **Filtering:** Does global refinement reduce duplicate local peaks and reject transient shocks?
3. **Usefulness:** Do detected regimes improve financial monitoring or downstream risk tasks?

---

## 1. Experiment Overview

We use three experiment sets.

| Set | Data Type | Ground Truth | Purpose |
|---|---|---:|---|
| A | Controlled synthetic data | Exact | Diagnose when the method works or fails |
| B | Synthetic overlays on real financial data | Exact injected labels | Test realistic dependence with known changepoints |
| C | Real financial data | Event-window labels | Test practical market relevance |

The main paper should focus on these three sets. General CPD benchmarks can be left for the appendix.

---

## 2. Methods Compared

### 2.1 Proposed Method

The full method contains three layers:

1. **Local Wasserstein alert:** compares recent reference and current windows.
2. **Regime prototype posterior:** assigns the current market window to Wasserstein regime prototypes.
3. **Rolling global refinement:** selects persistent boundaries from local candidates and suppresses transient or duplicate alerts.

### 2.2 Ablations

| Variant | Purpose |
|---|---|
| Local-only Wasserstein | Tests whether global refinement is needed |
| Local + global, no prototype | Tests boundary refinement alone |
| Local + prototype, no global | Tests regime posterior without denoising |
| Full method | Main proposed model |
| Coordinate-wise Wasserstein | Tests whether joint geometry matters |
| Bures-Wasserstein | Tests covariance/factor-regime detection |
| Sliced/Sinkhorn Wasserstein | Tests multivariate empirical distribution detection |

### 2.3 Baselines

Use a compact but defensible baseline set.

| Category | Baselines |
|---|---|
| Classical CPD | CUSUM mean/volatility, PELT-RBF, Binary Segmentation |
| Distributional CPD | MMD window scan, Energy distance |
| OT CPD | Local W2T + matched filter, sliced Wasserstein, Bures-Wasserstein |
| Online/regime models | BOCPD, Gaussian HMM or Markov-switching model |

---

## 3. Fair Comparison Protocol

### 3.1 Threshold Calibration

To avoid rewarding methods that simply fire more often, calibrate thresholds under a no-change null.

Protocol:

1. Simulate or bootstrap no-change sequences.
2. Tune each method to the same false-alarm budget, such as:
   - 1 false alarm per 1,000 observations, or
   - 5% sequence-level false-positive probability.
3. Freeze thresholds before evaluating changed sequences.

### 3.2 Detection Tolerance

For synthetic and semi-synthetic experiments, a detected changepoint is correct if it is close to the true changepoint:

$$
|\hat{\tau} - \tau^*| \leq \delta.
$$

Use:

$$
\delta = w/2,
$$

where $w$ is the rolling window length. Also report localization error directly.

---

# Set A: Controlled Synthetic Experiments

Synthetic experiments should be difficulty sweeps, not single easy examples.

## A1. Mean/Variance Shift Sanity Check

**Purpose:** Verify the method works on standard CPD problems.

Data:

$$
X_t \sim N(\mu_1, \Sigma_1), \quad t \leq \tau,
$$

$$
X_t \sim N(\mu_2, \Sigma_2), \quad t > \tau.
$$

Difficulty knobs:

| Knob | Values |
|---|---|
| Mean shift | 0.05, 0.10, 0.20, 0.50 |
| Volatility ratio | 1.05, 1.10, 1.25, 1.50 |
| Dimension | 1, 5, 20, 100 |
| Regime length | 50, 100, 250 |

Expected outcome:

Classical methods should perform strongly. This experiment is mainly a sanity check, not the main contribution.

---

## A2. Variance-Matched Tail Shift

**Purpose:** Test sensitivity to tail changes beyond mean and variance.

Data:

$$
P = N(0,1),
$$

$$
Q = t_\nu(0,\tilde{\sigma}_\nu^2),
\quad
\tilde{\sigma}_\nu = \sqrt{\frac{\nu - 2}{\nu}}.
$$

This matches the variance of $Q$ to $P$.

Difficulty knobs:

| Knob | Values |
|---|---|
| Student-t degrees of freedom $\nu$ | 4, 6, 8, 12, 20, 50 |
| Regime length | 50, 100, 250 |
| Background noise | none, mild GARCH, strong GARCH |

Expected outcome:

Variance-based alarms should weaken because variance is matched. Distributional methods should perform better.

---

## A3. Scenario-Mixture Weight Shift

**Purpose:** Test changes in probability mass between market scenarios.

Before changepoint:

$$
P = \frac{1}{2}N(-a,\sigma^2) + \frac{1}{2}N(a,\sigma^2).
$$

After changepoint:

$$
Q =
\left(\frac{1}{2}+\delta\right)N(-a,\sigma^2)
+
\left(\frac{1}{2}-\delta\right)N(a,\sigma^2).
$$

Difficulty knobs:

| Knob | Values |
|---|---|
| Mixture weight shift $\delta$ | 0.02, 0.05, 0.10, 0.20 |
| Mode separation $a/\sigma$ | 1, 2, 3, 5 |
| Regime length | 50, 100, 250 |

Expected outcome:

Wasserstein-type methods should be useful when probability mass moves between separated scenarios.

---

## A4. Fixed-Marginal Correlation Crisis

**Purpose:** Test joint dependence changes that coordinate-wise methods miss.

Data:

$$
X_t \sim N(0,\Sigma_{\rho_1}), \quad t \leq \tau,
$$

$$
X_t \sim N(0,\Sigma_{\rho_2}), \quad t > \tau,
$$

where

$$
\Sigma_\rho = (1-\rho)I_d + \rho \mathbf{1}\mathbf{1}^\top.
$$

The marginal variance of each coordinate stays fixed, but the correlation changes.

Difficulty knobs:

| Knob | Values |
|---|---|
| Correlation jump $\Delta \rho$ | 0.05, 0.10, 0.20, 0.40 |
| Starting correlation $\rho_1$ | 0.0, 0.2, 0.5, 0.8 |
| Dimension $d$ | 5, 20, 50, 100 |

Expected outcome:

Coordinate-wise methods should fail or degrade because the marginals are unchanged. Joint methods such as Bures-Wasserstein, Sinkhorn, MMD, or full multivariate methods should perform better.

---

## A5. Low-Rank Factor Shock

**Purpose:** Test high-dimensional factor-driven regime changes.

Data:

$$
P = N(0,I_d),
$$

$$
Q = N(0,I_d + \epsilon vv^\top),
\quad
\|v\|_2 = 1.
$$

Difficulty knobs:

| Knob | Values |
|---|---|
| Shock strength $\epsilon$ | 0.05, 0.10, 0.20, 0.50 |
| Dimension $d$ | 10, 50, 100, 500 |
| Factor direction | dense, sparse, sector-sparse |
| Regime length | 50, 100, 250 |

Expected outcome:

Bures-Wasserstein or projected Wasserstein should be more robust than naive high-dimensional empirical methods when the signal is low-rank.

---

## A6. Transient Shock vs Persistent Regime

**Purpose:** Test whether the global refinement layer rejects short-lived shocks.

Transient shock:

$$
P \rightarrow R \rightarrow P.
$$

Persistent regime shift:

$$
P \rightarrow Q.
$$

Difficulty knobs:

| Knob | Values |
|---|---|
| Shock length | 1, 2, 5, 10, 20 |
| Shock magnitude | small, medium, large |
| Background volatility | low, medium, high |

Main metric:

**Transient false-confirmation rate** =  
confirmed transient shocks / total transient shocks.

Expected outcome:

Local-only methods may fire on transient shocks. The full method should produce local alerts but avoid confirming short shocks as persistent regimes.

---

## A7. Duplicate Local Peak Suppression

**Purpose:** Test whether global refinement keeps one boundary instead of many nearby local peaks.

Data:

$$
P \rightarrow Q.
$$

Use tail, mixture, correlation, and factor shifts from A2-A5.

Difficulty knobs:

| Knob | Values |
|---|---|
| Window length $w$ | 20, 50, 100 |
| Signal strength | weak, medium, strong |
| Candidate threshold | liberal, medium, conservative |

Metrics:

**False duplicate rate**
=
Number of extra detections within one event window / Number of true events

Also report:

- number of detected boundaries per true event,
- CP-F1 before duplicate clustering,
- CP-F1 after duplicate clustering.

Expected outcome:

The global refinement layer should reduce duplicate peaks without sacrificing recall.

---

# Set B: Synthetic Overlays on Real Financial Data

These experiments preserve realistic financial dependence while keeping known changepoints.

## B1. Block-Bootstrap Regime Splice

**Purpose:** Test realistic market regime changes with exact splice labels.

Construct sequences from real ETF or factor blocks:

```text
calm blocks -> stress blocks -> calm blocks
```

Example source periods:

| Regime | Candidate Windows |
|---|---|
| Calm | 2017, 2019 |
| Stress | 2008 GFC, 2011 Euro crisis, 2020 COVID crash, 2022 rates shock |
| Rebound | April-June 2020 rebound |

Ground truth:

The splice dates are the true changepoints.

Metrics:

- CP-F1,
- localization error,
- detection delay,
- false alarms per 1,000 days,
- duplicate detections per event.

Expected outcome:

The method should detect realistic regime splices under heavy tails, volatility clustering, and cross-asset dependence.

---

## B2. Injected Covariance/Factor Shock

**Purpose:** Test controlled factor shocks on realistic financial residuals.

Procedure:

1. Fit a factor model to ETF or industry-portfolio returns.
2. Extract residuals.
3. Inject a low-rank shock during a known interval:

$$
r_t^{new} = r_t + z_t v,
$$

where $z_t$ has increased variance during the shock interval.

Difficulty knobs:

| Knob | Values |
|---|---|
| Shock strength | weak, medium, strong |
| Shock duration | 20, 50, 100 days |
| Factor direction | broad market, sector, random sparse |
| Background period | calm, volatile, mixed |

Metrics:

- detection power,
- localization error,
- factor-direction recovery,
- runtime and memory.

Expected outcome:

Bures/projected Wasserstein should be useful for covariance and factor-regime shifts.

---

## B3. Real-Block Scenario Reweighting

**Purpose:** Test scenario-probability shifts using real market blocks.

Create two block libraries:

- favorable blocks: positive equity returns, stable rates, tight credit;
- adverse blocks: equity drawdowns, credit weakness, high volatility.

Before changepoint:

$$
P(\text{adverse block}) = p_1.
$$

After changepoint:

$$
P(\text{adverse block}) = p_2,
\quad
p_2 > p_1.
$$

Difficulty knobs:

| Knob | Values |
|---|---|
| $p_2 - p_1$ | 0.05, 0.10, 0.20, 0.40 |
| Block length | 5, 10, 20 days |
| Asset universe size | 5, 10, 20+ |

Metrics:

- detection power versus $p_2-p_1$,
- posterior shift,
- prototype interpretability,
- false alarms under matched threshold.

Expected outcome:

The method should detect gradual reweighting toward adverse financial scenarios.

---

# Set C: Real Financial Data

Real financial data does not have exact changepoint labels. Use broad event windows instead of single-day labels.

## C1. Cross-Asset ETF Regime Detection

Universe:

```text
SPY, QQQ, IWM, TLT, GLD, HYG, LQD, XLF, XLK, XLE, UUP
```

Event windows:

| Event | Approximate Window | Regime Type |
|---|---:|---|
| Global Financial Crisis | 2007-07 to 2009-03 | credit/equity/liquidity stress |
| Euro debt crisis | 2011-07 to 2011-12 | sovereign/risk-off stress |
| Volmageddon | 2018-02 | volatility shock |
| Q4 2018 selloff | 2018-10 to 2018-12 | equity/rates/liquidity stress |
| COVID crash | 2020-02 to 2020-04 | crash/liquidity shock |
| COVID rebound | 2020-04 to 2020-08 | rebound regime |
| Inflation/rates shock | 2022-01 to 2022-12 | rates/equity correlation shift |
| Regional banking stress | 2023-03 to 2023-05 | financial-sector stress |

Main metric:

**Event-window recall**
=
event windows hit / event windows

Also report:

- false alarms per year outside event windows,
- first detection date,
- detection delay relative to event-window start,
- duplicate detections per event,
- assigned regime prototype.

Expected outcome:

The method should align with major market stress/rebound windows while producing fewer duplicate alerts and more interpretable regime labels.

---

## C2. Fama-French Factors and Industry Portfolios

Data:

- Fama-French 3-factor, 5-factor, and momentum factors;
- 10, 30, or 49 industry portfolios.

Feature groups:

| Feature | Purpose |
|---|---|
| Factor returns | Detect risk-premium regime changes |
| Industry returns | Detect sector rotation |
| Rolling factor covariance | Detect factor-risk regimes |
| PCA factors | Detect low-rank market regimes |
| Cross-sectional dispersion | Detect market breadth regimes |

Downstream task:

Compare covariance forecasts with and without regime conditioning.

| Model | Covariance Estimate |
|---|---|
| Baseline | rolling covariance or EWMA covariance |
| Regime-conditioned | prototype-specific covariance blended with EWMA |

Metrics:

- portfolio variance forecast error,
- covariance Frobenius error,
- realized volatility forecast error.

Expected outcome:

Regime-conditioned covariance forecasts should improve during periods when market covariance structure changes.

---

## C3. Volatility Surface or Volatility Proxy Features

Use this experiment only if data is available.

Possible data:

- VIX and VVIX,
- VIX term structure,
- OptionMetrics Ivy DB,
- SPX option surface features.

Feature groups:

| Feature | Meaning |
|---|---|
| ATM implied volatility | volatility level |
| put-call skew | crash/tail pricing |
| term slope | near-term versus long-term stress |
| surface curvature | smile regime |
| surface PCA factors | latent volatility-surface regimes |

Metrics:

- event-window recall around volatility events,
- false alarms per year,
- regime posterior stability,
- volatility forecasting improvement,
- VaR/ES calibration improvement.

Expected outcome:

This experiment connects the method to volatility-regime discovery and makes the project more finance-specific.

---

# 4. Main Metrics

## 4.1 Detection Metrics

| Metric | Meaning |
|---|---|
| CP-F1 | Precision/recall for detected changepoints |
| Localization error | Distance between detected and true changepoint |
| Detection delay | Delay for online detection |
| False alarms | Detections under no-change or outside event windows |
| Event-window recall | Fraction of real market events hit |

## 4.2 Local-Global Filtering Metrics

| Metric | Meaning |
|---|---|
| Duplicate detections per event | Measures repeated local peaks near one true event |
| Transient false-confirmation rate | Measures short shocks wrongly confirmed as regimes |
| Candidate-to-confirmed compression | Local candidates divided by confirmed boundaries |
| Confirmation stability | Fraction of confirmed boundaries that persist over updates |

## 4.3 Regime Interpretation Metrics

| Metric | Meaning |
|---|---|
| Posterior entropy | Confidence of regime assignment |
| Posterior shift | Size of transition between regime posteriors |
| ARI/NMI | Regime-label recovery in synthetic data |
| Prototype purity | Whether inferred prototypes match true regimes |
| Historical analog quality | Whether prototypes correspond to interpretable market periods |

## 4.4 Financial Utility Metrics

| Task | Metrics |
|---|---|
| Covariance forecasting | Portfolio variance forecast error, Frobenius error |
| Volatility forecasting | Realized volatility MAE/MSE |
| VaR/ES calibration | Exceedance rate, shortfall behavior |
| Portfolio risk control | Drawdown, realized volatility, missed rebound cost |

---

# 5. Robustness and Efficiency Tests

## 5.1 Robustness Tests

| Test | Purpose |
|---|---|
| Window length sensitivity | CPD often depends heavily on window size |
| Threshold sensitivity | Avoids overclaiming from lucky calibration |
| Dimension sensitivity | Important for cross-asset finance |
| GARCH/noise background | Tests robustness under realistic returns |
| Number of prototypes $K$ | Tests regime-layer stability |
| Penalties $\lambda,\eta$ | Tests global-refinement stability |
| Random seed stability | Checks whether results are cherry-picked |

## 5.2 Efficiency Tests

Report runtime and memory as $T$, $d$, and $w$ increase.

For the full method, report runtime separately for:

```text
local Wasserstein scoring
+ regime posterior computation
+ rolling global refinement
```

This makes the computational cost transparent.

---

# 6. Minimal First Milestone

Implement the following first:

1. Synthetic experiments A2-A7.
2. Baselines:
   - CUSUM/EWMA,
   - PELT-RBF,
   - MMD,
   - local W2T,
   - Bures or sliced Wasserstein,
   - full proposed method.
3. Null threshold calibration.
4. One ETF block-bootstrap experiment.
5. One real ETF event-window experiment using COVID 2020 and the 2022 rates shock.
6. Runtime and memory table.

This milestone is enough to determine whether the method is promising before expanding the full benchmark suite.

---

# 7. Suggested Paper Positioning

A defensible final claim is:

> The proposed method is not intended to replace classical CPD on simple mean or variance shifts. Instead, it targets finance-relevant distributional regime changes: tail shifts, scenario-probability shifts, dependence breakdowns, and low-rank factor shocks. Its main contribution is the combination of Wasserstein local alerts, global persistence filtering, and interpretable regime posteriors. Therefore, the experiments evaluate not only detection accuracy, but also false-alarm control, duplicate suppression, transient-shock rejection, robustness, runtime, and downstream financial utility.

This framing avoids overclaiming while giving the method a clear reason to exist.
