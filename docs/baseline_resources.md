# Baseline Resources

This file records the source used for each implemented or registered baseline in
the synthetic experiment runner. The same metadata is embedded in every result
row through `changept_detection.baselines.core.BASELINE_RESOURCES`.

## Optimal Transport Baselines

- `local_w2t`: Cheng et al., _Optimal Transport Based Change Point Detection and
  Time Series Segment Clustering_, arXiv:1911.01325.
  https://arxiv.org/abs/1911.01325
- `coordinate_w2t`: same Cheng et al. W2T source, reimplemented coordinate-wise.
- `sliced_wasserstein`: Python Optimal Transport (POT) reference resource.
  https://pythonot.github.io/
- `sinkhorn`: POT Sinkhorn/regularized OT reference resource.
  https://pythonot.github.io/
- `bures`: closed-form Gaussian/covariance 2-Wasserstein geometry.
- `watch_proxy`: Faber et al., _WATCH: Wasserstein Change Point Detection for
  High-Dimensional Time Series Data_, arXiv:2201.07125.
  https://arxiv.org/abs/2201.07125

## Classical CPD Baselines

- `pelt_l2`: Killick, Fearnhead, Eckley (2012), implemented through `ruptures`
  when available. PELT docs:
  https://ctruong.perso.math.cnrs.fr/ruptures-docs/build/html/detection/pelt.html
- `pelt_rbf`, `pelt_normal`, `binseg`, `bottomup`: `ruptures`.
  https://centre-borelli.github.io/ruptures-docs/

## Nonparametric Distributional Baselines

- `mmd`, `m_statistic`: Gretton et al., _A Kernel Two-Sample Test_, JMLR 2012,
  and Li, Xie, Dai, Song, _M-Statistic for Kernel Change-Point Detection_,
  NeurIPS 2015.
  https://www.jmlr.org/papers/v13/gretton12a.html
  https://papers.nips.cc/paper/by-source-2015-1852
- `energy`: Matteson and James, _A Nonparametric Approach for Multiple Change
  Point Analysis of Multivariate Data_, JASA 2014.
  https://doi.org/10.1080/01621459.2013.849605
- `ks`: SciPy `ks_2samp`.
  https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ks_2samp.html
- `cvm`: SciPy `cramervonmises_2samp`.
  https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.cramervonmises_2samp.html
- `density_ratio_proxy`: Liu, Yamada, Collier, Sugiyama, _Change-Point Detection
  in Time-Series Data by Relative Density-Ratio Estimation_, arXiv:1203.0453.
  https://arxiv.org/abs/1203.0453

## Online And Finance Baselines

- `cusum_mean`, `cusum_vol`: classic cumulative-sum monitoring.
  https://en.wikipedia.org/wiki/CUSUM
- `ewma_vol`: RiskMetrics-style EWMA volatility monitoring.
  https://www.msci.com/research-and-insights/paper/1996-riskmetrics-technical-document
- `bocpd_gaussian`: Adams and MacKay, _Bayesian Online Changepoint Detection_,
  arXiv:0710.3742; reference repo https://github.com/dtolpin/bocd
- `gaussian_hmm`: `hmmlearn` Gaussian HMM.
  https://github.com/hmmlearn/hmmlearn
- `markov_switching`: Hamilton, _A New Approach to the Economic Analysis of
  Nonstationary Time Series and the Business Cycle_, Econometrica 1989; use
  `statsmodels.tsa.regime_switching` for full econometric fits.
- `covariance_frobenius`, `pca_subspace`: simple covariance/factor baselines
  specified in `docs/experiment_plan.md` for S4.

## Proposed Method Entry

- `proposed_local_global`: compact implementation of the local Wasserstein
  alert plus persistence and duplicate-suppression idea from
  `docs/experiment_plan.md`. It is included so the synthetic suite can exercise
  the local-global filtering claim while the full prototype/posterior layer is
  developed further.
