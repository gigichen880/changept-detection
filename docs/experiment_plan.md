# Experiment Plan: Online Local–Global Wasserstein Regime Filtering for Quant Finance Change-Point Detection

## 0. Goal of the experiment suite

This experiment suite is designed to evaluate an **online local–global Wasserstein regime filtering** method for financial change-point detection (CPD). The method is not positioned as “Wasserstein always wins.” Instead, the target claim is more specific:

> The proposed method is most useful when financial regime changes are distributional, joint, tail-driven, scenario-weight-driven, or factor-covariance-driven, and when online local alerts must be filtered into persistent, interpretable regime boundaries.

Therefore, the experiments should not be easy one-off changepoints where all baselines achieve nearly perfect detection. The synthetic experiments should be **diagnostic stress tests** with controlled difficulty sweeps. The semi-synthetic experiments should preserve realistic financial dependence while still providing ground-truth changepoints. The real-data experiments should emphasize event-window alignment, false-alarm control, and downstream financial utility.

---

## 1. Main research questions

### RQ1: Distributional sensitivity

Can the method detect changes beyond mean and variance shifts, such as tail thickening, mixture reweighting, dependence changes, and low-rank covariance shocks?

### RQ2: Joint-regime detection

Does full or structure-aware Wasserstein geometry detect joint financial regime changes that coordinate-wise methods miss?

### RQ3: Local-to-global filtering

Does the rolling global refinement layer reduce duplicate local peaks and reject transient shocks that do not become persistent regimes?

### RQ4: Regime interpretation

Does the Wasserstein prototype layer produce stable and interpretable regime posteriors, rather than only binary change/no-change alerts?

### RQ5: Downstream financial usefulness

Do detected regimes improve covariance forecasting, volatility forecasting, VaR/ES calibration, or portfolio risk-control decisions?

---

## 2. Data setup overview

We use four levels of data realism.

| Level   |                  Dataset type |                         Ground truth? | Purpose                                                                     |
| ------- | ----------------------------: | ------------------------------------: | --------------------------------------------------------------------------- |
| Level 1 |     Controlled synthetic data |                                 Exact | Isolate failure modes and produce phase-transition curves.                  |
| Level 2 | Semi-synthetic financial data |      Exact injected or spliced labels | Preserve realistic financial dependence/heavy tails while retaining labels. |
| Level 3 |           Real financial data | Event-window labels, not exact labels | Test market relevance and false-alarm behavior.                             |
| Level 4 |        General CPD benchmarks |                  Provided annotations | Sanity benchmark outside finance.                                           |

The main paper should emphasize Levels 1–3. Level 4 can be appendix/sanity-check material.

---

## 3. Proposed method variants

We should report the full method and several variants to demonstrate which component contributes to performance.

### 3.1 Full proposed method

**Full model:** local Wasserstein alert + regime posterior/prototype layer + rolling global refinement.

At time $t$:

1. Build reference and current windows.
2. Compute local Wasserstein alert score $A_t = D_W(\hat\mu_t^{ref}, \hat\mu_t^{cur})$.
3. Compute posterior over Wasserstein prototypes:

$$
\pi_t(k) \propto \exp\left(-D_W(\hat\mu_t^{cur}, \nu_k)/\tau_{temp}\right).
$$

4. Add candidate boundary if local alert or posterior shift is large.
5. On a recent horizon $[t-H,t]$, solve a regularized global refinement problem over candidates.
6. Confirm a boundary only if retained and persistent.

### 3.2 Distance choices inside the proposed method

| Variant                              | Use case                                                           |
| ------------------------------------ | ------------------------------------------------------------------ |
| 1D Wasserstein                       | Scalar returns, realized volatility, tail-risk features.           |
| Coordinate-wise Wasserstein          | Cheap baseline; useful but weak for dependence changes.            |
| Sliced Wasserstein                   | General multivariate empirical windows.                            |
| Sinkhorn divergence                  | Multivariate empirical distributions with entropic regularization. |
| Bures-Wasserstein                    | Gaussian/covariance/factor-regime monitoring.                      |
| Projected/factor-aligned Wasserstein | Low-rank factor regimes in high dimension.                         |

Recommended implementation source for OT computations: **POT: Python Optimal Transport**, which provides common OT routines including exact/regularized OT and related Wasserstein tools.

Source: https://pythonot.github.io/

---

## 4. Baseline methods

Use baselines from four categories: OT baselines, classical CPD, nonparametric distributional CPD, and finance/econometric regime models.

### 4.1 OT baselines

| Baseline                          | Description                                                                                                                 | Source / implementation                                                                                                                               |
| --------------------------------- | --------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| Local W2T + matched filter        | Prior OT-CPD baseline: compare adjacent windows with Wasserstein two-sample statistic, then matched-filter and peak-select. | Cheng et al., _Optimal Transport Based Change Point Detection and Time Series Segment Clustering_, arXiv:1911.01325. https://arxiv.org/abs/1911.01325 |
| Coordinate-wise W2T               | Apply W2T to each coordinate and average. This is especially important because it can miss pure dependence changes.         | Same as above; reimplement if needed.                                                                                                                 |
| Sliced Wasserstein window scan    | Compare adjacent multivariate windows using random 1D projections.                                                          | POT or custom implementation. https://pythonot.github.io/                                                                                             |
| Sinkhorn window scan              | Entropic OT/Sinkhorn divergence between adjacent empirical windows.                                                         | POT. https://pythonot.github.io/                                                                                                                      |
| Bures-Wasserstein covariance scan | Compare rolling covariance matrices using Bures-Wasserstein distance.                                                       | Implement directly from closed form; useful for covariance/factor regimes.                                                                            |
| WATCH                             | High-dimensional Wasserstein CPD method that models an initial distribution and monitors incoming points.                   | Faber et al., _WATCH: Wasserstein Change Point Detection for High-Dimensional Time Series Data_, arXiv:2201.07125. https://arxiv.org/abs/2201.07125   |

### 4.2 Classical offline CPD baselines

These are important because reviewers will expect standard segmentation baselines.

| Baseline                 | Description                                                               | Source / implementation                                                                                                                             |
| ------------------------ | ------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| PELT-L2                  | Penalized exact segmentation for mean shifts with squared-error cost.     | `ruptures`. PELT docs: https://ctruong.perso.math.cnrs.fr/ruptures-docs/build/html/detection/pelt.html                                              |
| PELT-RBF                 | Kernelized/RBF cost for more general distributional changes.              | `ruptures`. https://centre-borelli.github.io/ruptures-docs/                                                                                         |
| PELT-normal              | Parametric Gaussian mean/variance change cost, if available.              | `ruptures` or custom likelihood cost.                                                                                                               |
| Binary Segmentation      | Fast recursive segmentation baseline.                                     | `ruptures`.                                                                                                                                         |
| Wild Binary Segmentation | Strong multiple-change baseline with random intervals.                    | Fryzlewicz, _Wild Binary Segmentation for Multiple Change-Point Detection_, Annals of Statistics, 2014. Implementation may be custom or R packages. |
| Bottom-Up segmentation   | Agglomerative segmentation baseline.                                      | `ruptures`.                                                                                                                                         |
| Window-based CPD         | Sliding-window discrepancy baseline, separate from our global refinement. | `ruptures` window method or custom.                                                                                                                 |

Recommended library: **ruptures**.

Source: https://centre-borelli.github.io/ruptures-docs/

### 4.3 Nonparametric distributional CPD baselines

| Baseline                     | Description                                                     | Source / implementation                                                                                                                                             |
| ---------------------------- | --------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| M-statistic / kernel CPD     | Kernel two-sample CPD method used as a baseline in Cheng et al. | Li, Xie, Dai, Song, _M-Statistic for Kernel Change-Point Detection_, NeurIPS 2015. https://papers.nips.cc/paper/by-source-2015-1852                                 |
| MMD window scan              | Compare adjacent windows using maximum mean discrepancy.        | Implement from kernel two-sample test; Gretton et al. JMLR 2012.                                                                                                    |
| Energy distance / E-divisive | Nonparametric multivariate CPD based on energy distance.        | Matteson and James, _A Nonparametric Approach for Multiple Change Point Analysis of Multivariate Data_, JASA 2014. https://doi.org/10.1080/01621459.2013.849605     |
| KS window scan               | 1D Kolmogorov-Smirnov two-sample statistic.                     | `scipy.stats.ks_2samp`.                                                                                                                                             |
| Cramér-von Mises window scan | 1D distributional two-sample statistic.                         | `scipy.stats.cramervonmises_2samp`.                                                                                                                                 |
| Relative density-ratio CPD   | CPD by relative density-ratio estimation.                       | Liu, Yamada, Collier, Sugiyama, _Change-Point Detection in Time-Series Data by Relative Density-Ratio Estimation_, arXiv:1203.0453. https://arxiv.org/abs/1203.0453 |

### 4.4 Online/statistical baselines

| Baseline                                       | Description                                            | Source / implementation                                                                                                        |
| ---------------------------------------------- | ------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------ |
| CUSUM mean alarm                               | Classic online detector for mean shifts.               | Implement directly.                                                                                                            |
| CUSUM volatility alarm                         | Apply CUSUM to squared returns or realized volatility. | Implement directly.                                                                                                            |
| EWMA volatility alarm                          | Finance-standard volatility-monitoring baseline.       | Implement directly.                                                                                                            |
| Bayesian Online Change Point Detection (BOCPD) | Online Bayesian run-length posterior method.           | Adams and MacKay, 2007. arXiv:0710.3742. https://arxiv.org/abs/0710.3742; Python repo example: https://github.com/dtolpin/bocd |
| Bayesian Changepoint Detection Python package  | Another Python implementation family.                  | https://github.com/hildensia/bayesian_changepoint_detection                                                                    |

### 4.5 Finance/econometric regime baselines

| Baseline                                      | Description                                                                             | Source / implementation                                                                                                                                                 |
| --------------------------------------------- | --------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Gaussian HMM                                  | Hidden Markov model on returns/features; compare state transitions to detected regimes. | `hmmlearn`: https://github.com/hmmlearn/hmmlearn                                                                                                                        |
| Markov-switching regression/volatility        | Econometric regime-switching baseline.                                                  | Hamilton, _A New Approach to the Economic Analysis of Nonstationary Time Series and the Business Cycle_, Econometrica 1989. Python: `statsmodels.tsa.regime_switching`. |
| Markov-switching asset allocation baseline    | Finance motivation for regime-dependent correlations/volatility.                        | Ang and Bekaert, _International Asset Allocation With Regime Shifts_, RFS 2002. https://academic.oup.com/rfs/article/15/4/1137/1568247                                  |
| Bai-Perron structural breaks                  | Multiple structural breaks in linear regression parameters.                             | Bai and Perron structural-break literature; e.g. Econometrics Journal 2003 critical values. https://academic.oup.com/ectj/article/6/1/72/5074163                        |
| GARCH-break / MS-GARCH style volatility model | Volatility-regime baseline.                                                             | Implement a simple GARCH-vol residual break or use R/Python package if needed.                                                                                          |

---

## 5. Fair comparison protocol

### 5.1 Threshold calibration

To avoid giving an advantage to methods that simply fire more often, calibrate thresholds using a no-change null.

Recommended protocol:

1. For each data-generating process and dimension, simulate or bootstrap no-change sequences.
2. Tune each method’s threshold/penalty to a target false-alarm budget, such as:
   - 1 false alarm per 1,000 observations, or
   - 5% sequence-level false-positive probability.
3. Freeze thresholds before evaluating changed sequences.

This gives a fair comparison of detection power, delay, and localization at matched false-alarm rates.

### 5.2 Hyperparameter selection

| Method family          | Hyperparameters                                                                      | Selection rule                                                                           |
| ---------------------- | ------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------- |
| Rolling-window methods | window length $w$, threshold                                                         | Select from grid using null calibration and validation alternatives.                     |
| Proposed global layer  | horizon $H$, penalty $\lambda$, short-segment penalty $\eta$, persistence length $m$ | Choose via validation grid; report sensitivity.                                          |
| Prototype layer        | number of regimes $K$, temperature $\tau_{temp}$, update frequency                   | Choose via stability/predictive utility; report sensitivity to $K$.                      |
| PELT/BinSeg/WBS        | cost type, penalty, min segment length                                               | Match false-alarm budget; use same min segment length as proposed method where possible. |
| BOCPD                  | hazard rate, predictive distribution                                                 | Choose using validation likelihood/false-alarm budget.                                   |
| HMM/MS models          | number of states, covariance type                                                    | Select by BIC/AIC and compare with fixed K.                                              |

### 5.3 Detection tolerance

For exact-label synthetic/semi-synthetic data, define a detection as correct if it falls within a margin $\delta$ of the true changepoint:

$$
|\hat\tau - \tau^*| \leq \delta.
$$

Recommended settings:

- $\delta = w/2$ for rolling-window methods.
- Also report localization error directly to avoid hiding delayed detections.

---

## 6. Synthetic experiments

Synthetic experiments should be presented as **difficulty sweeps** and **phase-transition curves**, not single easy examples.

### 6.1 Synthetic Experiment S0: sanity-check mean/variance shifts

**Purpose:** Verify the method works on standard simple CPD settings, but do not overclaim.

#### Data-generating process

Univariate or multivariate Gaussian:

$$
X_t \sim N(\mu_1, \Sigma_1), \quad t \leq \tau,
$$

$$
X_t \sim N(\mu_2, \Sigma_2), \quad t > \tau.
$$

#### Difficulty knobs

| Knob                         | Values                       |
| ---------------------------- | ---------------------------- |
| Mean shift $\|\mu_2-\mu_1\|$ | 0.05, 0.10, 0.20, 0.50, 1.00 |
| Volatility ratio             | 1.05, 1.10, 1.25, 1.50, 2.00 |
| Segment length               | 50, 100, 250, 500            |
| Dimension $d$                | 1, 5, 20, 100                |

#### Baselines

- PELT-L2
- PELT-normal
- Binary Segmentation
- CUSUM mean/variance
- BOCPD
- Local W2T
- Proposed method

#### Expected outcome

All reasonable methods should do well at high signal. Classical methods may match or beat the proposed method on pure mean/variance shifts. That is acceptable and should be stated honestly.

---

### 6.2 Synthetic Experiment S1: variance-matched tail shift

**Purpose:** Test detection of tail thickening when mean and variance are not enough.

#### Data-generating process

$$
P = N(0,1),
$$

$$
Q = t_\nu(0, \tilde\sigma_\nu^2), \quad \tilde\sigma_\nu = \sqrt{\frac{\nu-2}{\nu}},
$$

so that $Q$ has matched variance.

#### Difficulty knobs

| Knob                               | Values              |
| ---------------------------------- | ------------------- |
| Student-t degrees of freedom $\nu$ | 4, 6, 8, 12, 20, 50 |
| Regime length                      | 50, 100, 250, 500   |
| Background GARCH noise             | none, mild, strong  |
| Number of changepoints             | 1, 3, 5             |

Large $\nu$ makes the Student-t closer to Gaussian, so the task becomes harder.

#### Baselines

- Rolling variance / EWMA volatility alarm
- CUSUM on squared returns
- KS window scan
- Cramér-von Mises window scan
- MMD window scan
- Energy distance
- PELT-RBF
- Local W2T
- Proposed method

#### Metrics

- CP-F1
- Detection power at matched false-alarm rate
- Localization error
- Detection delay
- Tail-risk downstream improvement: VaR/ES exceedance calibration before/after regime conditioning

#### Expected claim

The proposed method should remain competitive under subtle tail shifts where variance-based baselines degrade. This experiment supports the distributional-sensitivity claim.

---

### 6.3 Synthetic Experiment S2: scenario-mixture weight shift

**Purpose:** Test regime changes where probability mass moves between market scenarios.

#### Data-generating process

Before changepoint:

$$
P = \frac{1}{2}N(-a, \sigma^2) + \frac{1}{2}N(a, \sigma^2).
$$

After changepoint:

$$
Q = \left(\frac{1}{2}+\delta\right)N(-a, \sigma^2)
+ \left(\frac{1}{2}-\delta\right)N(a, \sigma^2).
$$

Interpretation: the market reweights from favorable to adverse macro scenario without changing the scenario locations.

#### Difficulty knobs

| Knob                          | Values                       |
| ----------------------------- | ---------------------------- |
| Mixture weight shift $\delta$ | 0.02, 0.05, 0.10, 0.20, 0.30 |
| Mode separation $a/\sigma$    | 1, 2, 3, 5                   |
| Regime length                 | 50, 100, 250, 500            |
| Serial dependence             | IID, AR(1), block dependence |

#### Baselines

- KS window scan
- MMD window scan
- Energy distance
- PELT-RBF
- Local W2T
- BOCPD with Gaussian predictive model
- Proposed method

#### Metrics

- CP-F1
- Detection power vs $\delta$
- Localization error
- Regime posterior stability
- Prototype purity: whether one prototype corresponds to each mixture/scenario regime

#### Expected claim

When modes are separated, Wasserstein-type methods should benefit because they account for both how much probability mass moves and how far it moves. Gaussian parametric baselines may be weak if the mean/variance change is subtle.

---

### 6.4 Synthetic Experiment S3: fixed-marginal correlation crisis

**Purpose:** Test joint dependence shifts where univariate marginals remain unchanged.

#### Data-generating process

Let:

$$
X_t \sim N(0, \Sigma_{\rho_1}), \quad t \leq \tau,
$$

$$
X_t \sim N(0, \Sigma_{\rho_2}), \quad t > \tau.
$$

Use equicorrelation matrices:

$$
\Sigma_\rho = (1-\rho)I_d + \rho \mathbf{1}\mathbf{1}^\top.
$$

All coordinates have variance 1 before and after the changepoint, so coordinate-wise marginal detectors have little or no signal.

#### Difficulty knobs

| Knob                                          | Values                 |
| --------------------------------------------- | ---------------------- |
| Correlation jump $\Delta\rho = \rho_2-\rho_1$ | 0.05, 0.10, 0.20, 0.40 |
| Starting correlation $\rho_1$                 | 0.0, 0.2, 0.5, 0.8     |
| Dimension $d$                                 | 5, 20, 50, 100         |
| Regime length                                 | 50, 100, 250, 500      |

#### Baselines

- Coordinate-wise W2T
- Coordinate-wise KS
- PELT-L2
- PELT-RBF
- MMD window scan
- Energy distance
- Bures-Wasserstein covariance scan
- Sliced Wasserstein
- Sinkhorn
- Proposed method with Bures or projected Wasserstein

#### Metrics

- Detection power vs $\Delta\rho$
- CP-F1
- Localization error
- False alarms per 1,000 observations
- Correlation-regime classification accuracy

#### Expected claim

Coordinate-wise methods should fail or degrade because marginals are unchanged. Structure-aware multivariate distances should detect the crisis. This is one of the strongest experiments for the joint-regime claim.

---

### 6.5 Synthetic Experiment S4: low-rank factor shock

**Purpose:** Test high-dimensional financial regime changes driven by a small number of latent factors.

#### Data-generating process

Before changepoint:

$$
X_t \sim N(0, I_d).
$$

After changepoint:

$$
X_t \sim N(0, I_d + \epsilon vv^\top), \quad \|v\|_2=1.
$$

This represents a shock to a latent factor such as rates, dollar, liquidity, or risk appetite.

#### Difficulty knobs

| Knob                      | Values                              |
| ------------------------- | ----------------------------------- |
| Shock strength $\epsilon$ | 0.05, 0.10, 0.20, 0.50, 1.00        |
| Dimension $d$             | 10, 50, 100, 500                    |
| Number of shocked factors | 1, 3, 5                             |
| Factor vector sparsity    | dense, sector-sparse, random sparse |
| Regime length             | 50, 100, 250, 500                   |

#### Baselines

- PELT-RBF
- MMD window scan
- Sliced Wasserstein
- Sinkhorn
- Covariance Frobenius-distance scan
- PCA subspace-distance scan
- Bures-Wasserstein covariance scan
- Proposed method with Bures/projected Wasserstein

#### Metrics

- Detection power vs $\epsilon$
- Detection power vs dimension $d$
- CP-F1
- Localization error
- Runtime and memory
- Estimated factor-regime interpretability: alignment between detected factor direction and true $v$

#### Expected claim

Naive high-dimensional distributional methods may suffer signal dilution. Bures-Wasserstein or factor-aligned Wasserstein should be more robust when the signal is covariance/factor-driven.

---

### 6.6 Synthetic Experiment S5: transient shock vs persistent regime

**Purpose:** Test whether the global refinement layer rejects one-day or few-day shocks that do not become persistent regimes.

#### Data-generating processes

Transient shock:

$$
P \rightarrow R \rightarrow P,
$$

where $R$ lasts $m$ observations.

Persistent regime shift:

$$
P \rightarrow Q.
$$

Use multiple shock types:

- large mean outlier,
- large volatility burst,
- tail shock,
- correlation shock,
- liquidity-style drawdown shock.

#### Difficulty knobs

| Knob                    | Values               |
| ----------------------- | -------------------- |
| Shock length $m$        | 1, 2, 5, 10, 20      |
| Shock magnitude         | small, medium, large |
| Minimum segment penalty | low, medium, high    |
| Background volatility   | low, medium, high    |

#### Baselines

- Local W2T
- MMD window scan
- PELT-RBF
- BOCPD
- CUSUM/EWMA alarms
- Proposed method without global refinement
- Proposed full method

#### Metrics

- Transient false-confirmation rate:

$$
\frac{\#\text{transient shocks confirmed as regimes}}{\#\text{transient shocks}}.
$$

- Persistent-regime detection rate
- Detection delay
- Posterior entropy during transient shock
- Regime-persistence score

#### Expected claim

The full method should fire local alerts during large shocks but avoid confirming them as persistent regimes unless they last long enough and improve global segmentation. This directly validates the local-global design.

---

### 6.7 Synthetic Experiment S6: duplicate local peak suppression

**Purpose:** Test whether overlapping rolling windows create multiple nearby detections and whether global refinement keeps only one boundary.

#### Data-generating process

One true changepoint:

$$
P \rightarrow Q.
$$

Use tail, mixture, correlation, and factor versions from S1–S4.

#### Difficulty knobs

| Knob                | Values                        |
| ------------------- | ----------------------------- |
| Window length $w$   | 20, 50, 100, 250              |
| Signal strength     | weak, medium, strong          |
| Noise level         | low, medium, high             |
| Candidate threshold | liberal, medium, conservative |

#### Baselines

- Local W2T
- Local W2T + matched filter
- MMD window scan
- Window-based ruptures
- Proposed method without global refinement
- Proposed full method

#### Metrics

- False duplicate rate:

$$
\frac{\#\text{extra detections within one event window}}{\#\text{true events}}.
$$

- Number of detected boundaries per true event
- Best localization error within event window
- CP-F1 after duplicate clustering
- CP-F1 before duplicate clustering

#### Expected claim

The global refinement layer should reduce duplicate peaks without sacrificing event-level recall.

---

### 6.8 Synthetic Experiment S7: regime-posterior interpretability

**Purpose:** Test whether the prototype layer does more than detect boundaries: it should assign windows to recurring regimes.

#### Data-generating process

Generate a sequence with recurring regimes:

$$
P_1 \rightarrow P_2 \rightarrow P_1 \rightarrow P_3 \rightarrow P_2 \rightarrow P_4.
$$

Example regimes:

| Regime | Distributional meaning  |
| ------ | ----------------------- |
| $P_1$  | calm Gaussian           |
| $P_2$  | heavy-tail stress       |
| $P_3$  | high-correlation stress |
| $P_4$  | low-rank factor shock   |

#### Difficulty knobs

| Knob                         | Values                                 |
| ---------------------------- | -------------------------------------- |
| Number of regimes $K_{true}$ | 2, 3, 4, 6                             |
| Fitted prototype count $K$   | $K_{true}-1$, $K_{true}$, $K_{true}+2$ |
| Regime duration              | 50, 100, 250                           |
| Regime similarity            | easy, medium, hard                     |

#### Baselines

- HMM/Gaussian HMM
- Markov-switching model
- Cheng et al. detect-then-cluster OT baseline
- K-means on rolling features
- Spectral clustering on segment distributions
- Proposed prototype layer

#### Metrics

- Regime classification accuracy after label matching
- Adjusted Rand Index (ARI)
- Normalized Mutual Information (NMI)
- Posterior entropy
- Transition matrix estimation error
- Boundary F1 and regime-label accuracy jointly

#### Expected claim

The proposed method should provide online regime posteriors and historical analogs, while detect-then-cluster baselines only assign labels after segmentation.

---

## 7. Semi-synthetic finance experiments

Semi-synthetic finance experiments are the bridge between clean synthetic data and noisy real markets. They preserve realistic return distributions, volatility clustering, serial dependence, and cross-asset relationships while retaining known changepoints.

### 7.1 Data universe

Use daily adjusted close returns for liquid ETFs:

| Asset       | Proxy                          |
| ----------- | ------------------------------ |
| SPY         | U.S. large-cap equity          |
| QQQ         | growth/technology equity       |
| IWM         | small-cap equity               |
| TLT         | long-duration Treasury         |
| GLD         | gold                           |
| HYG         | high-yield credit              |
| LQD         | investment-grade credit        |
| XLF         | financial sector               |
| XLK         | technology sector              |
| XLE         | energy sector                  |
| UUP         | U.S. dollar                    |
| VIX or VIXY | volatility proxy, if available |

Possible data sources:

- Stooq daily data: https://stooq.com/
- Yahoo Finance via `yfinance`: https://pypi.org/project/yfinance/
- Fama-French data library for factors and industry portfolios: https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
- HBS overview of Fama-French portfolios/factors: https://www.library.hbs.edu/databases-cases-and-more/databases/fama-french-portfolios-and-factors

### 7.2 Feature construction

Use rolling windows of features such as:

| Feature group            | Features                                                                        |
| ------------------------ | ------------------------------------------------------------------------------- |
| Return features          | daily log returns, cumulative return, drawdown                                  |
| Volatility features      | realized volatility, squared returns, EWMA volatility                           |
| Tail features            | rolling skewness, kurtosis, downside semivariance, VaR/ES estimates             |
| Correlation features     | rolling average correlation, bond-equity correlation, credit-equity correlation |
| Cross-sectional features | dispersion, first PCA variance share, factor returns                            |
| Macro/market proxies     | VIX level/change, yield proxy if available, dollar proxy, credit spread proxy   |

### 7.3 Semi-synthetic Experiment F1: block-bootstrap regime insertion

#### Setup

1. Choose calm and stress historical blocks from real ETF data.
2. Build sequences by concatenating block-bootstrap samples:

$$
\text{calm blocks} \rightarrow \text{stress blocks} \rightarrow \text{calm blocks}.
$$

3. Keep the splice date as the ground-truth changepoint.

#### Example source blocks

| Regime  | Candidate historical windows                                                       |
| ------- | ---------------------------------------------------------------------------------- |
| Calm    | 2017 low-volatility market; 2019 pre-COVID expansion                               |
| Stress  | 2008 GFC; 2011 Euro crisis; Feb 2018 vol shock; 2020 COVID crash; 2022 rates shock |
| Rebound | April–June 2020 rebound; post-crisis recovery windows                              |

#### Baselines

Use the full main baseline set.

#### Metrics

- CP-F1
- Localization error
- Detection delay
- False alarms per 1,000 days
- Regime classification accuracy if regime labels are defined by source block

#### Expected claim

The method should detect distributional market-regime splices under realistic heavy tails and dependence, not just Gaussian simulations.

---

### 7.4 Semi-synthetic Experiment F2: injected covariance/factor shocks into real residuals

#### Setup

1. Fit a factor model to ETF or industry-portfolio returns.
2. Extract residuals.
3. Inject a controlled low-rank covariance shock during a known interval:

$$
r_t^{new} = r_t + z_t v,
$$

where $z_t$ has increased variance during the shock window. 4. Keep the injected shock interval as ground truth.

#### Difficulty knobs

| Knob              | Values                              |
| ----------------- | ----------------------------------- |
| Shock strength    | weak, medium, strong                |
| Shock duration    | 20, 50, 100 days                    |
| Factor direction  | broad market, sector, random sparse |
| Background period | calm, volatile, mixed               |

#### Baselines

- Covariance Frobenius scan
- PCA subspace scan
- Bures-Wasserstein scan
- MMD
- Sliced Wasserstein
- PELT-RBF
- Proposed method

#### Metrics

- Detection power
- Localization error
- Factor-direction recovery
- Runtime in high-dimensional settings

---

### 7.5 Semi-synthetic Experiment F3: scenario mixture reweighting using real blocks

#### Setup

Construct two empirical block libraries:

- favorable blocks: positive equity returns, stable rates, tight credit;
- adverse blocks: equity drawdown, credit weakness, high VIX or proxy volatility.

Before the changepoint, sample blocks with probability:

$$
P(\text{adverse}) = p_1.
$$

After the changepoint:

$$
P(\text{adverse}) = p_2 > p_1.
$$

This creates a realistic semi-synthetic scenario-weight shift.

#### Difficulty knobs

| Knob                | Values                 |
| ------------------- | ---------------------- |
| $p_2-p_1$           | 0.05, 0.10, 0.20, 0.40 |
| Block length        | 5, 10, 20 days         |
| Asset universe size | 5, 10, 20+             |

#### Metrics

- Detection power vs scenario-probability shift
- Regime posterior shift score
- Prototype interpretability
- False alarms under matched threshold

---

## 8. Real financial experiments

Real financial experiments should avoid pretending exact changepoint labels are objective. Use broad event windows and downstream financial validation.

### 8.1 Real Experiment R1: cross-asset ETF regime detection

#### Dataset

Daily ETF returns and features from approximately 2005 to present, depending on data availability.

Core universe:

$$
\{\text{SPY, QQQ, IWM, TLT, GLD, HYG, LQD, XLF, XLK, XLE, UUP}\}.
$$

#### Event windows

Use broad windows, not single-day labels.

| Event                   | Approximate event window | Regime type                    |
| ----------------------- | -----------------------: | ------------------------------ |
| Global Financial Crisis |       2007-07 to 2009-03 | credit/equity/liquidity stress |
| Euro debt crisis        |       2011-07 to 2011-12 | sovereign/risk-off stress      |
| Volmageddon             |                  2018-02 | volatility shock               |
| Q4 2018 selloff         |       2018-10 to 2018-12 | equity/rates/liquidity stress  |
| COVID crash             |       2020-02 to 2020-04 | abrupt crash/liquidity shock   |
| COVID rebound           |       2020-04 to 2020-08 | rebound regime                 |
| Inflation/rates shock   |       2022-01 to 2022-12 | rates/equity correlation shift |
| Regional banking stress |       2023-03 to 2023-05 | financial-sector stress        |

#### Baselines

- EWMA volatility alarm
- CUSUM mean/volatility
- PELT-RBF
- MMD window scan
- Local W2T
- Bures-Wasserstein covariance scan
- BOCPD
- Gaussian HMM / Markov-switching model
- Proposed method

#### Metrics

- Event-window recall:

$$
\frac{\#\text{event windows hit}}{\#\text{event windows}}.
$$

- False alarms per year outside event windows
- Average detection delay relative to event-window start
- Duplicate detections per event
- Regime posterior entropy
- Prototype descriptions by feature profile

#### Expected claim

The method should align with major market regime windows while producing fewer duplicate alerts and better regime-type interpretation than local-only scans.

---

### 8.2 Real Experiment R2: Fama-French factors and industry portfolios

#### Dataset

Use daily or monthly Fama-French factors and industry portfolios.

Recommended data:

- Fama-French 3-factor, 5-factor, momentum factor.
- 10/12/17/30/49 industry portfolios.
- Optional: size/book-to-market portfolios.

Source: Kenneth French Data Library, https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html

#### Feature setup

| Feature                    | Purpose                               |
| -------------------------- | ------------------------------------- |
| Factor returns             | Regime changes in risk premia.        |
| Industry returns           | Cross-sectional sector-regime shifts. |
| Rolling factor covariance  | Factor risk regime.                   |
| PCA factors                | Low-rank latent regime shifts.        |
| Cross-sectional dispersion | Market breadth/dispersion regime.     |

#### Baselines

- PELT-RBF
- MMD
- Energy distance
- Bures-Wasserstein covariance scan
- HMM/Markov-switching model
- Proposed method

#### Metrics

- Event-window recall
- False alarms per year
- Regime persistence
- Factor covariance forecasting improvement
- One-step-ahead factor volatility forecast improvement

#### Downstream task

Train covariance forecasts with and without regime conditioning:

1. Baseline covariance: rolling sample covariance or EWMA covariance.
2. Regime-conditioned covariance: average covariance from windows assigned to the current prototype, possibly blended with EWMA.
3. Evaluate realized portfolio variance forecasting error.

---

### 8.3 Real Experiment R3: implied-volatility surface features

#### Dataset options

Depending on data access:

- VIX/VVIX/VIX term structure proxies.
- OptionMetrics Ivy DB if available.
- SPX options surface features: ATM vol, skew, term slope, curvature, surface PCA factors.

#### Feature setup

| Feature                 | Meaning                       |
| ----------------------- | ----------------------------- |
| ATM implied volatility  | volatility level              |
| 25-delta put-call skew  | crash/tail pricing            |
| term slope              | near-term vs long-term stress |
| surface curvature       | smile/convexity regime        |
| PCA factors             | latent surface regimes        |
| realized-implied spread | risk premium regime           |

#### Baselines

- EWMA vol alarm
- CUSUM on VIX/surface PC1
- PELT-RBF
- MMD
- Bures-Wasserstein on surface PCA covariance
- HMM/Markov-switching volatility
- Proposed method

#### Metrics

- Event-window recall around volatility events
- False alarms per year
- Regime posterior stability
- Downstream volatility forecasting improvement
- VaR/ES calibration improvement

#### Expected claim

This experiment makes the project more finance-specific and connects the method to volatility-surface regime discovery.

---

## 9. General benchmark experiments

These are not the main finance contribution, but they help show the method is not overfit to our hand-designed financial cases.

### 9.1 Turing Change Point Dataset (TCPD)

TCPD is a real-world CPD benchmark with annotations.

Source: https://github.com/alan-turing-institute/TCPD

Benchmark framework: https://github.com/alan-turing-institute/TCPDBench

#### Use

- Run univariate version of our local-global detector.
- Compare to PELT, BinSeg, BOCPD, and kernel baselines.
- Keep this as appendix or sanity benchmark.

### 9.2 Numenta Anomaly Benchmark (NAB)

NAB is technically an anomaly-detection benchmark, not a pure CPD benchmark, but it is useful for online detection behavior.

Source: https://github.com/numenta/NAB

Paper: Lavin and Ahmad, _Evaluating Real-time Anomaly Detection Algorithms — The Numenta Anomaly Benchmark_, arXiv:1510.03336.

#### Use

- Use only as auxiliary online detection experiment.
- Report NAB-style early-detection score or simpler event-window precision/recall.
- Be clear that anomaly detection and regime CPD are related but not identical.

---

## 10. Main metrics

### 10.1 Boundary detection metrics

| Metric                | Formula / description                                                |
| --------------------- | -------------------------------------------------------------------- | ------------------ | ------------------------- |
| Precision             | Fraction of detected changepoints matched to true changepoints.      |
| Recall                | Fraction of true changepoints detected.                              |
| CP-F1                 | Harmonic mean of precision and recall.                               |
| Localization error    | $                                                                    | \hat\tau - \tau^\* | $ for matched detections. |
| Detection delay       | $\hat\tau - \tau^*$ for online methods, where positive means late.   |
| Event-window recall   | Fraction of known real event windows hit by at least one detection.  |
| False alarms per year | Number of detections outside labeled event windows divided by years. |

### 10.2 Local-global filtering metrics

| Metric                             | Formula / description                                                     |
| ---------------------------------- | ------------------------------------------------------------------------- |
| False duplicate rate               | Extra detections within one event window per true event.                  |
| Transient false-confirmation rate  | Fraction of transient shocks incorrectly confirmed as persistent regimes. |
| Candidate-to-confirmed compression | Number of local candidates divided by number of confirmed boundaries.     |
| Confirmation stability             | Fraction of confirmed boundaries that persist for $m$ update steps.       |

### 10.3 Regime posterior metrics

| Metric                    | Formula / description                                                                                  |
| ------------------------- | ------------------------------------------------------------------------------------------------------ |
| Posterior entropy         | $H(\pi_t)=-\sum_k\pi_t(k)\log\pi_t(k)$.                                                                |
| Posterior shift           | $\|\pi_t-\pi_{t-w}\|_1$ or KL divergence.                                                              |
| Regime duration stability | Distribution of consecutive time spent in each prototype.                                              |
| Prototype purity          | In synthetic/semi-synthetic data, fraction of windows assigned to correct regime after label matching. |
| ARI/NMI                   | Clustering agreement between inferred and true regime labels.                                          |

### 10.4 Downstream finance metrics

| Task                   | Metrics                                                                            |
| ---------------------- | ---------------------------------------------------------------------------------- |
| Covariance forecasting | Frobenius error, portfolio variance forecast error, realized variance calibration. |
| Volatility forecasting | MSE/MAE for realized volatility, QLIKE loss if applicable.                         |
| VaR/ES calibration     | Exceedance rate, Kupiec-style coverage, ES shortfall behavior.                     |
| Portfolio risk control | Max drawdown, realized volatility, turnover, missed rebound cost.                  |
| Risk attribution       | Stability/interpretablity of factor/correlation regimes.                           |

---

## 11. Ablation studies

The ablation section should be explicit and central to the paper.

| Ablation                             | Purpose                                                          |
| ------------------------------------ | ---------------------------------------------------------------- |
| Local W only                         | Tests whether the global layer is necessary.                     |
| Local W + global, no prototype       | Isolates the boundary-refinement contribution.                   |
| Local W + prototype, no global       | Isolates regime-posterior contribution without global denoising. |
| Full method                          | Main proposed model.                                             |
| Full method with coordinate-wise W   | Tests whether joint Wasserstein geometry matters.                |
| Full method with sliced W            | Tests random-projection OT in multivariate data.                 |
| Full method with Sinkhorn            | Tests empirical multivariate OT.                                 |
| Full method with Bures W             | Tests covariance/factor-regime version.                          |
| Offline prototypes                   | Tests fixed historical regime dictionary.                        |
| Online prototype updates             | Tests adaptive regime learning.                                  |
| No posterior-shift candidate trigger | Tests whether posterior changes add useful candidate boundaries. |
| No entropy/persistence confirmation  | Tests whether posterior confidence helps reject unstable shocks. |
| Different $K$                        | Tests regime-count sensitivity.                                  |
| Different $w,H,\lambda,\eta$         | Tests robustness to window/global-layer parameters.              |

---

## 12. Recommended result tables and figures

### 12.1 Main synthetic phase-transition figure

Create one multi-panel figure:

| Panel             | X-axis                      | Y-axis                         |
| ----------------- | --------------------------- | ------------------------------ |
| Tail shift        | Student-t $\nu$             | CP-F1 / detection power        |
| Mixture shift     | $\delta$                    | CP-F1 / detection power        |
| Correlation shift | $\Delta\rho$                | CP-F1 / detection power        |
| Low-rank shock    | dimension $d$ or $\epsilon$ | CP-F1 / detection power        |
| Transient shock   | shock length                | false confirmation rate        |
| Duplicate peak    | window length               | duplicate detections per event |

### 12.2 Main baseline comparison table

Rows: experiments. Columns: major baselines.

Suggested columns:

- CUSUM/EWMA
- PELT-RBF
- MMD
- Energy
- BOCPD
- Local W2T
- Bures/Sinkhorn/SW variant
- Proposed full method

### 12.3 Real finance event-window table

Rows: known market events. Columns:

- whether each method detected the event,
- first detection date,
- delay relative to event-window start,
- number of duplicate alerts,
- posterior regime label for our method.

### 12.4 Regime-prototype interpretation table

For each learned prototype, report:

| Prototype | Label we assign | Return profile        | Vol profile | Correlation profile              | Tail profile | Representative historical windows |
| --------- | --------------- | --------------------- | ----------- | -------------------------------- | ------------ | --------------------------------- |
| 1         | Calm expansion  | low positive          | low         | low/moderate                     | thin         | 2017, 2019                        |
| 2         | Equity stress   | negative              | high        | high                             | heavy        | 2008, 2020                        |
| 3         | Rates shock     | equity/bond both weak | high        | bond-equity correlation positive | medium/heavy | 2022                              |
| 4         | Rebound         | high positive         | declining   | unstable                         | medium       | 2020 rebound                      |

This table helps make the method interpretable to finance readers.

---

## 13. Implementation checklist

### 13.1 Data pipeline

- [ ] Download/load ETF daily prices.
- [ ] Compute adjusted log returns.
- [ ] Align dates and handle missing values.
- [ ] Download/load Fama-French factors and industry portfolios.
- [ ] Build rolling feature matrices.
- [ ] Build synthetic data generators.
- [ ] Build semi-synthetic block bootstrap and shock injection tools.
- [ ] Define event-window metadata file for real financial experiments.

### 13.2 Baseline pipeline

- [ ] Implement CUSUM/EWMA alarms.
- [ ] Wrap `ruptures` PELT/BinSeg/BottomUp/Window methods.
- [ ] Implement or wrap MMD window scan.
- [ ] Implement energy-distance scan or use available implementation.
- [ ] Implement local W2T matched-filter baseline.
- [ ] Implement sliced Wasserstein and Sinkhorn scans using POT.
- [ ] Implement Bures-Wasserstein covariance scan.
- [ ] Implement BOCPD baseline.
- [ ] Implement HMM/Markov-switching baseline.

### 13.3 Proposed method pipeline

- [ ] Implement rolling local Wasserstein score.
- [ ] Implement prototype initialization from historical windows.
- [ ] Implement posterior computation.
- [ ] Implement posterior-shift candidate trigger.
- [ ] Implement rolling candidate set.
- [ ] Implement global subset-selection/refinement.
- [ ] Implement persistence confirmation.
- [ ] Implement online/offline prototype updates.
- [ ] Implement regime posterior logging.

### 13.4 Evaluation pipeline

- [ ] Implement threshold calibration under no-change null.
- [ ] Implement matching between detected and true changepoints.
- [ ] Implement CP-F1/localization/delay metrics.
- [ ] Implement duplicate and transient metrics.
- [ ] Implement event-window metrics.
- [ ] Implement posterior entropy and regime-label metrics.
- [ ] Implement downstream covariance/volatility/VaR tasks.
- [ ] Implement result plotting and LaTeX/Markdown table export.

---

## 14. Suggested minimal first milestone

The first implementation milestone should be small but compelling:

1. Synthetic S1–S6 with 3–5 baselines:
   - PELT-RBF,
   - MMD,
   - Local W2T,
   - Bures/SW/Sinkhorn depending on experiment,
   - proposed full method.
2. Matched false-alarm threshold calibration.
3. One ETF semi-synthetic block-bootstrap experiment.
4. One real ETF event-window experiment for 2020 COVID and 2022 rates shock.

This is enough to validate the story before adding every baseline.

---

## 15. Suggested paper positioning

A defensible final claim:

> In simple mean and variance shifts, classical CPD methods remain strong and often competitive. The proposed method is designed for a different and more finance-relevant regime: distributional market shifts, dependence changes, low-rank factor shocks, and online settings where local alerts must be filtered into persistent regime boundaries. Across controlled synthetic stress tests, semi-synthetic financial splices, and real cross-asset event windows, the local-global Wasserstein regime filter should be evaluated by its robustness envelope: high detection power under subtle distributional changes, fewer duplicate detections, lower transient false-confirmation rate, and more interpretable regime posteriors.

---

## 16. References and source list

### OT and Wasserstein CPD

1. Cheng, K. C., Aeron, S., Hughes, M. C., Hussey, E., Miller, E. L. _Optimal Transport Based Change Point Detection and Time Series Segment Clustering_. arXiv:1911.01325. https://arxiv.org/abs/1911.01325
2. Faber, K., Corizzo, R., Sniezynski, B., Baron, M., Japkowicz, N. _WATCH: Wasserstein Change Point Detection for High-Dimensional Time Series Data_. arXiv:2201.07125. https://arxiv.org/abs/2201.07125
3. POT: Python Optimal Transport. https://pythonot.github.io/
4. Cuturi, M. _Sinkhorn Distances: Lightspeed Computation of Optimal Transport_. NeurIPS 2013.

### Classical CPD

5. `ruptures` Python package documentation. https://centre-borelli.github.io/ruptures-docs/
6. PELT documentation in `ruptures`. https://ctruong.perso.math.cnrs.fr/ruptures-docs/build/html/detection/pelt.html
7. Killick, R., Fearnhead, P., Eckley, I. A. _Optimal Detection of Changepoints With a Linear Computational Cost_. JASA 2012.
8. Fryzlewicz, P. _Wild Binary Segmentation for Multiple Change-Point Detection_. Annals of Statistics 2014.

### Nonparametric distributional CPD

9. Li, S., Xie, Y., Dai, H., Song, L. _M-Statistic for Kernel Change-Point Detection_. NeurIPS 2015. https://papers.nips.cc/paper/by-source-2015-1852
10. Gretton, A., Borgwardt, K. M., Rasch, M. J., Schölkopf, B., Smola, A. _A Kernel Two-Sample Test_. JMLR 2012.
11. Matteson, D. S., James, N. A. _A Nonparametric Approach for Multiple Change Point Analysis of Multivariate Data_. JASA 2014. https://doi.org/10.1080/01621459.2013.849605
12. Liu, S., Yamada, M., Collier, N., Sugiyama, M. _Change-Point Detection in Time-Series Data by Relative Density-Ratio Estimation_. arXiv:1203.0453. https://arxiv.org/abs/1203.0453

### Online/statistical CPD

13. Adams, R. P., MacKay, D. J. C. _Bayesian Online Changepoint Detection_. arXiv:0710.3742. https://arxiv.org/abs/0710.3742
14. Python BOCPD repository example. https://github.com/dtolpin/bocd
15. Bayesian changepoint detection Python repository. https://github.com/hildensia/bayesian_changepoint_detection

### Finance regime and structural-break baselines

16. Hamilton, J. D. _A New Approach to the Economic Analysis of Nonstationary Time Series and the Business Cycle_. Econometrica 1989.
17. Ang, A., Bekaert, G. _International Asset Allocation With Regime Shifts_. Review of Financial Studies 2002. https://academic.oup.com/rfs/article/15/4/1137/1568247
18. Bai-Perron structural-break literature; example critical-values paper. https://academic.oup.com/ectj/article/6/1/72/5074163
19. `statsmodels` Markov switching models. https://www.statsmodels.org/stable/tsa.html
20. `hmmlearn` HMM package. https://github.com/hmmlearn/hmmlearn

### Financial and benchmark datasets

21. Kenneth French Data Library. https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
22. HBS Fama-French portfolio/factor overview. https://www.library.hbs.edu/databases-cases-and-more/databases/fama-french-portfolios-and-factors
23. Turing Change Point Dataset. https://github.com/alan-turing-institute/TCPD
24. Turing Change Point Benchmark. https://github.com/alan-turing-institute/TCPDBench
25. Numenta Anomaly Benchmark. https://github.com/numenta/NAB
26. Lavin, A., Ahmad, S. _Evaluating Real-time Anomaly Detection Algorithms — The Numenta Anomaly Benchmark_. arXiv:1510.03336. https://arxiv.org/abs/1510.03336
