"""
Baseline detectors and metrics for the synthetic CPD experiment suite.

Every baseline exposed by this module has a resource entry pointing to the
original paper, canonical documentation, or reference repository listed in
docs/experiment_plan.md. Optional third-party packages are used when present,
but the core window-scan baselines run with NumPy/SciPy/scikit-learn only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import permutations
from typing import Callable, Iterable

import numpy as np
from scipy import linalg, stats
from scipy.spatial.distance import cdist
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


try:  # Optional dependency recommended by the experiment plan.
    import ruptures as rpt  # type: ignore
except Exception:  # pragma: no cover - depends on local environment.
    rpt = None

try:  # Optional dependency recommended by the experiment plan.
    import ot  # type: ignore
except Exception:  # pragma: no cover - depends on local environment.
    ot = None

try:  # Optional finance/regime baseline.
    from hmmlearn.hmm import GaussianHMM  # type: ignore
except Exception:  # pragma: no cover - depends on local environment.
    GaussianHMM = None


@dataclass(frozen=True)
class BaselineResource:
    """Citation and implementation resource for a baseline."""

    key: str
    name: str
    category: str
    source: str
    url: str
    implementation: str
    notes: str = ""


@dataclass
class DetectionResult:
    """Common return object for all detectors."""

    method: str
    changepoints: list[int]
    scores: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    threshold: float | None = None
    metadata: dict = field(default_factory=dict)


BASELINE_RESOURCES: dict[str, BaselineResource] = {
    "coordinate_w2_window_scan": BaselineResource(
        "coordinate_w2_window_scan",
        "Coordinate-wise W2 window scan",
        "OT",
        "Cheng et al., Optimal Transport Based Change Point Detection and Time Series Segment Clustering, arXiv:1911.01325",
        "https://arxiv.org/abs/1911.01325",
        "Adjacent-window coordinate-wise empirical W2; not the full Brownian-bridge W2T statistic.",
    ),
    "coordinate_w2_matched_filter": BaselineResource(
        "coordinate_w2_matched_filter",
        "Coordinate-wise W2 + matched filter",
        "OT",
        "Cheng et al., arXiv:1911.01325",
        "https://arxiv.org/abs/1911.01325",
        "W2 window scan with triangular matched-filter post-processing (Cheng-style proxy).",
    ),
    "local_w2t": BaselineResource(
        "local_w2t",
        "Alias for coordinate_w2_window_scan",
        "OT",
        "Cheng et al., arXiv:1911.01325",
        "https://arxiv.org/abs/1911.01325",
        "Deprecated alias; use coordinate_w2_window_scan or coordinate_w2_matched_filter.",
    ),
    "coordinate_w2t": BaselineResource(
        "coordinate_w2t",
        "Coordinate-wise W2T",
        "OT",
        "Cheng et al., Optimal Transport Based Change Point Detection and Time Series Segment Clustering, arXiv:1911.01325",
        "https://arxiv.org/abs/1911.01325",
        "Reimplemented by averaging one-dimensional W2 scores over coordinates.",
    ),
    "sliced_wasserstein": BaselineResource(
        "sliced_wasserstein",
        "Sliced Wasserstein window scan",
        "OT",
        "POT: Python Optimal Transport documentation",
        "https://pythonot.github.io/",
        "Custom random-projection implementation; POT can also provide OT utilities.",
    ),
    "sinkhorn": BaselineResource(
        "sinkhorn",
        "Sinkhorn window scan",
        "OT",
        "POT: Python Optimal Transport documentation",
        "https://pythonot.github.io/",
        "Uses POT entropic OT when installed.",
    ),
    "bures": BaselineResource(
        "bures",
        "Bures-Wasserstein covariance scan",
        "OT",
        "Closed-form Gaussian/covariance 2-Wasserstein geometry",
        "https://en.wikipedia.org/wiki/Wasserstein_metric#Normal_distributions",
        "Implemented directly from the covariance closed form.",
    ),
    "covariance_frobenius": BaselineResource(
        "covariance_frobenius",
        "Covariance Frobenius-distance scan",
        "Covariance/factor baseline",
        "docs/experiment_plan.md S4 baseline",
        "docs/experiment_plan.md",
        "Implemented as an adjacent-window Frobenius norm between sample covariance matrices.",
    ),
    "pca_subspace": BaselineResource(
        "pca_subspace",
        "PCA subspace-distance scan",
        "Covariance/factor baseline",
        "docs/experiment_plan.md S4 baseline",
        "docs/experiment_plan.md",
        "Implemented as an adjacent-window projector distance between leading PCA subspaces.",
    ),
    "watch_proxy": BaselineResource(
        "watch_proxy",
        "WATCH-style initial-distribution monitor",
        "OT",
        "Faber et al., WATCH: Wasserstein Change Point Detection for High-Dimensional Time Series Data, arXiv:2201.07125",
        "https://arxiv.org/abs/2201.07125",
        "Lightweight proxy: compare each rolling window to an initial reference distribution.",
        "Use the authors' code for exact WATCH replication when available.",
    ),
    "pelt_l2": BaselineResource(
        "pelt_l2",
        "PELT-L2",
        "Classical offline CPD",
        "Killick, Fearnhead, Eckley (2012); ruptures PELT documentation",
        "https://ctruong.perso.math.cnrs.fr/ruptures-docs/build/html/detection/pelt.html",
        "Uses ruptures Pelt(model='l2') when installed.",
    ),
    "pelt_rbf": BaselineResource(
        "pelt_rbf",
        "PELT-RBF",
        "Classical offline CPD",
        "ruptures documentation",
        "https://centre-borelli.github.io/ruptures-docs/",
        "Uses ruptures Pelt(model='rbf') when installed.",
    ),
    "pelt_normal": BaselineResource(
        "pelt_normal",
        "PELT-normal",
        "Classical offline CPD",
        "ruptures documentation",
        "https://centre-borelli.github.io/ruptures-docs/",
        "Uses ruptures Pelt(model='normal') when installed.",
    ),
    "binseg": BaselineResource(
        "binseg",
        "Binary Segmentation",
        "Classical offline CPD",
        "ruptures documentation",
        "https://centre-borelli.github.io/ruptures-docs/",
        "Uses ruptures Binseg when installed.",
    ),
    "bottomup": BaselineResource(
        "bottomup",
        "Bottom-Up segmentation",
        "Classical offline CPD",
        "ruptures documentation",
        "https://centre-borelli.github.io/ruptures-docs/",
        "Uses ruptures BottomUp when installed.",
    ),
    "window_rbf": BaselineResource(
        "window_rbf",
        "Window-based CPD",
        "Classical offline CPD",
        "ruptures window method",
        "https://centre-borelli.github.io/ruptures-docs/",
        "Implemented as a generic RBF-MMD window scan to avoid a hard ruptures dependency.",
    ),
    "mmd": BaselineResource(
        "mmd",
        "MMD window scan",
        "Nonparametric distributional CPD",
        "Gretton et al., A Kernel Two-Sample Test, JMLR 2012",
        "https://www.jmlr.org/papers/v13/gretton12a.html",
        "Implemented with an RBF kernel and median-bandwidth heuristic.",
    ),
    "m_statistic": BaselineResource(
        "m_statistic",
        "M-statistic / kernel CPD",
        "Nonparametric distributional CPD",
        "Li, Xie, Dai, Song, M-Statistic for Kernel Change-Point Detection, NeurIPS 2015",
        "https://papers.nips.cc/paper/by-source-2015-1852",
        "Implemented as a local MMD statistic over adjacent windows.",
    ),
    "energy": BaselineResource(
        "energy",
        "Energy distance scan",
        "Nonparametric distributional CPD",
        "Matteson and James, A Nonparametric Approach for Multiple Change Point Analysis of Multivariate Data, JASA 2014",
        "https://doi.org/10.1080/01621459.2013.849605",
        "Implemented as adjacent-window multivariate energy distance.",
    ),
    "ks": BaselineResource(
        "ks",
        "KS window scan",
        "Nonparametric distributional CPD",
        "scipy.stats.ks_2samp documentation",
        "https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ks_2samp.html",
        "Implemented coordinate-wise with average statistic.",
    ),
    "cvm": BaselineResource(
        "cvm",
        "Cramer-von Mises window scan",
        "Nonparametric distributional CPD",
        "scipy.stats.cramervonmises_2samp documentation",
        "https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.cramervonmises_2samp.html",
        "Implemented coordinate-wise with average statistic.",
    ),
    "density_ratio_proxy": BaselineResource(
        "density_ratio_proxy",
        "Relative density-ratio CPD proxy",
        "Nonparametric distributional CPD",
        "Liu, Yamada, Collier, Sugiyama, Change-Point Detection in Time-Series Data by Relative Density-Ratio Estimation, arXiv:1203.0453",
        "https://arxiv.org/abs/1203.0453",
        "Proxy implementation using logistic separability of adjacent windows.",
    ),
    "cusum_mean": BaselineResource(
        "cusum_mean",
        "CUSUM mean alarm",
        "Online/statistical",
        "Classic cumulative-sum sequential monitoring",
        "https://en.wikipedia.org/wiki/CUSUM",
        "Implemented directly as two-sided standardized CUSUM.",
    ),
    "cusum_vol": BaselineResource(
        "cusum_vol",
        "CUSUM volatility alarm",
        "Online/statistical",
        "Classic CUSUM applied to squared returns",
        "https://en.wikipedia.org/wiki/CUSUM",
        "Implemented directly on squared standardized observations.",
    ),
    "ewma_vol": BaselineResource(
        "ewma_vol",
        "EWMA volatility alarm",
        "Online/statistical",
        "RiskMetrics-style EWMA volatility monitoring",
        "https://www.msci.com/research-and-insights/paper/1996-riskmetrics-technical-document",
        "Implemented directly using EWMA variance innovations.",
    ),
    "bocpd_gaussian": BaselineResource(
        "bocpd_gaussian",
        "Bayesian Online Change Point Detection",
        "Online/statistical",
        "Adams and MacKay, Bayesian Online Changepoint Detection, arXiv:0710.3742; dtolpin/bocd",
        "https://arxiv.org/abs/0710.3742",
        "Lightweight Gaussian predictive BOCPD implementation for univariate streams.",
        "Reference repo: https://github.com/dtolpin/bocd",
    ),
    "gaussian_hmm": BaselineResource(
        "gaussian_hmm",
        "Gaussian HMM",
        "Finance/econometric regime model",
        "hmmlearn GaussianHMM repository",
        "https://github.com/hmmlearn/hmmlearn",
        "Uses hmmlearn when installed.",
    ),
    "markov_switching": BaselineResource(
        "markov_switching",
        "Markov-switching regression/volatility",
        "Finance/econometric regime model",
        "Hamilton, A New Approach to the Economic Analysis of Nonstationary Time Series and the Business Cycle, Econometrica 1989",
        "https://www.jstor.org/stable/1912559",
        "Registered for experiment reporting; use statsmodels.tsa.regime_switching for full model fits.",
    ),
    "proposed_local_only": BaselineResource(
        "proposed_local_only",
        "Proposed: local alert only",
        "Proposed method",
        "docs/experiment_plan.md §3.1",
        "docs/experiment_plan.md",
        "Ablation without prototypes or global refinement.",
    ),
    "proposed_local_persistence_proxy": BaselineResource(
        "proposed_local_persistence_proxy",
        "Proposed: local + persistence proxy",
        "Proposed method",
        "docs/experiment_plan.md §3.1",
        "docs/experiment_plan.md",
        "Former proposed_local_global stub (peak refine + persistence).",
    ),
    "proposed_local_global_no_proto": BaselineResource(
        "proposed_local_global_no_proto",
        "Proposed: local + global, no prototypes",
        "Proposed method",
        "docs/experiment_plan.md §3.1",
        "docs/experiment_plan.md",
        "Horizon subset refinement without prototype posterior.",
    ),
    "proposed_local_proto_no_global": BaselineResource(
        "proposed_local_proto_no_global",
        "Proposed: local + prototypes, no global",
        "Proposed method",
        "docs/experiment_plan.md §3.1",
        "docs/experiment_plan.md",
        "Prototype posterior shift without global refinement.",
    ),
    "proposed_full": BaselineResource(
        "proposed_full",
        "Proposed: local + prototypes + global refinement",
        "Proposed method",
        "docs/experiment_plan.md §3.1; WPCG refinement optional via wpcg.py",
        "docs/experiment_plan.md",
        "Full planned stack (compact implementation for sweeps).",
    ),
    "proposed_local_global": BaselineResource(
        "proposed_local_global",
        "Alias for proposed_local_persistence_proxy",
        "Proposed method",
        "docs/experiment_plan.md",
        "docs/experiment_plan.md",
        "Deprecated alias.",
    ),
}

PROPOSED_METHOD_KEYS = frozenset(
    {
        "proposed_local_only",
        "proposed_local_persistence_proxy",
        "proposed_local_global_no_proto",
        "proposed_local_proto_no_global",
        "proposed_full",
        "proposed_local_global",
    }
)


def resource_table(keys: Iterable[str] | None = None) -> list[dict[str, str]]:
    """Return citation metadata suitable for writing experiment reports."""

    selected = BASELINE_RESOURCES if keys is None else {k: BASELINE_RESOURCES[k] for k in keys}
    return [
        {
            "key": resource.key,
            "name": resource.name,
            "category": resource.category,
            "source": resource.source,
            "url": resource.url,
            "implementation": resource.implementation,
            "notes": resource.notes,
        }
        for resource in selected.values()
    ]


def as_2d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        return x[:, None]
    if x.ndim != 2:
        raise ValueError(f"Expected 1D or 2D data, got shape {x.shape}")
    return x


def w2_squared_1d(x: np.ndarray, y: np.ndarray) -> float:
    """Squared empirical 2-Wasserstein distance in one dimension."""

    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if len(x) == 0 or len(y) == 0:
        return 0.0
    xs = np.sort(x)
    ys = np.sort(y)
    if len(xs) == len(ys):
        return float(np.mean((xs - ys) ** 2))
    n_q = 4 * max(len(xs), len(ys))
    u = (np.arange(n_q) + 0.5) / n_q
    fx = np.interp(u, (np.arange(len(xs)) + 0.5) / len(xs), xs)
    fy = np.interp(u, (np.arange(len(ys)) + 0.5) / len(ys), ys)
    return float(np.mean((fx - fy) ** 2))


def coordinate_w2(left: np.ndarray, right: np.ndarray) -> float:
    left = as_2d(left)
    right = as_2d(right)
    return float(np.mean([w2_squared_1d(left[:, j], right[:, j]) for j in range(left.shape[1])]))


def sliced_wasserstein(
    left: np.ndarray,
    right: np.ndarray,
    n_projections: int = 64,
    seed: int = 0,
) -> float:
    left = as_2d(left)
    right = as_2d(right)
    rng = np.random.default_rng(seed)
    directions = rng.normal(size=(n_projections, left.shape[1]))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True) + 1e-12
    scores = [
        w2_squared_1d(left @ direction, right @ direction)
        for direction in directions
    ]
    return float(np.mean(scores))


def covariance_matrix(x: np.ndarray, ridge: float = 1e-6) -> np.ndarray:
    x = as_2d(x)
    if len(x) <= 1:
        return np.eye(x.shape[1]) * ridge
    cov = np.cov(x, rowvar=False)
    cov = np.atleast_2d(cov)
    return cov + ridge * np.eye(cov.shape[0])


def bures_wasserstein_cov(left: np.ndarray, right: np.ndarray) -> float:
    """Squared Bures-Wasserstein distance between empirical covariance matrices."""

    c1 = covariance_matrix(left)
    c2 = covariance_matrix(right)
    c1_sqrt = linalg.sqrtm(c1)
    middle = linalg.sqrtm(c1_sqrt @ c2 @ c1_sqrt)
    value = np.trace(c1 + c2 - 2.0 * middle)
    return float(np.real(value))


def covariance_frobenius(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.linalg.norm(covariance_matrix(left) - covariance_matrix(right), ord="fro"))


def pca_subspace_distance(left: np.ndarray, right: np.ndarray, n_components: int = 1) -> float:
    c1 = covariance_matrix(left)
    c2 = covariance_matrix(right)
    _, v1 = np.linalg.eigh(c1)
    _, v2 = np.linalg.eigh(c2)
    u1 = v1[:, -n_components:]
    u2 = v2[:, -n_components:]
    return float(np.linalg.norm(u1 @ u1.T - u2 @ u2.T, ord="fro"))


def median_bandwidth(z: np.ndarray) -> float:
    z = as_2d(z)
    if len(z) < 2:
        return 1.0
    sample = z
    if len(sample) > 400:
        rng = np.random.default_rng(0)
        sample = sample[rng.choice(len(sample), size=400, replace=False)]
    d2 = cdist(sample, sample, metric="sqeuclidean")
    values = d2[np.triu_indices_from(d2, k=1)]
    med = np.median(values[values > 0]) if np.any(values > 0) else 1.0
    return float(np.sqrt(0.5 * med) + 1e-12)


def mmd_rbf(left: np.ndarray, right: np.ndarray, gamma: float | None = None) -> float:
    left = as_2d(left)
    right = as_2d(right)
    if gamma is None:
        sigma = median_bandwidth(np.vstack([left, right]))
        gamma = 1.0 / (2.0 * sigma**2)
    kxx = np.exp(-gamma * cdist(left, left, metric="sqeuclidean")).mean()
    kyy = np.exp(-gamma * cdist(right, right, metric="sqeuclidean")).mean()
    kxy = np.exp(-gamma * cdist(left, right, metric="sqeuclidean")).mean()
    return float(max(kxx + kyy - 2.0 * kxy, 0.0))


def energy_distance(left: np.ndarray, right: np.ndarray) -> float:
    left = as_2d(left)
    right = as_2d(right)
    dxy = cdist(left, right).mean()
    dxx = cdist(left, left).mean()
    dyy = cdist(right, right).mean()
    return float(max(2.0 * dxy - dxx - dyy, 0.0))


def ks_stat(left: np.ndarray, right: np.ndarray) -> float:
    left = as_2d(left)
    right = as_2d(right)
    return float(np.mean([stats.ks_2samp(left[:, j], right[:, j]).statistic for j in range(left.shape[1])]))


def cvm_stat(left: np.ndarray, right: np.ndarray) -> float:
    left = as_2d(left)
    right = as_2d(right)
    return float(
        np.mean(
            [
                stats.cramervonmises_2samp(left[:, j], right[:, j]).statistic
                for j in range(left.shape[1])
            ]
        )
    )


def density_ratio_proxy(left: np.ndarray, right: np.ndarray) -> float:
    """Classify window membership; high separability proxies density-ratio change."""

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    left = as_2d(left)
    right = as_2d(right)
    x = np.vstack([left, right])
    y = np.r_[np.zeros(len(left)), np.ones(len(right))]
    if min(np.bincount(y.astype(int))) < 3:
        return 0.0
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
    clf = LogisticRegression(max_iter=500)
    return float(max(cross_val_score(clf, x, y, cv=cv).mean() - 0.5, 0.0))


def sinkhorn_divergence(left: np.ndarray, right: np.ndarray, reg: float = 0.1) -> float:
    if ot is None:
        raise ImportError("POT is required for the Sinkhorn baseline. Install with `pip install POT`.")
    left = as_2d(left)
    right = as_2d(right)
    a = np.ones(len(left)) / len(left)
    b = np.ones(len(right)) / len(right)
    cxy = cdist(left, right, metric="sqeuclidean")
    cxx = cdist(left, left, metric="sqeuclidean")
    cyy = cdist(right, right, metric="sqeuclidean")
    sxy = float(ot.sinkhorn2(a, b, cxy, reg))
    sxx = float(ot.sinkhorn2(a, a, cxx, reg))
    syy = float(ot.sinkhorn2(b, b, cyy, reg))
    return max(sxy - 0.5 * sxx - 0.5 * syy, 0.0)


def coordinate_w2_matched_scan(left: np.ndarray, right: np.ndarray, window: int = 50, **_) -> float:
    del window, _
    return coordinate_w2(left, right)


WINDOW_METRICS: dict[str, Callable[..., float]] = {
    "coordinate_w2_window_scan": coordinate_w2,
    "local_w2t": coordinate_w2,
    "coordinate_w2t": coordinate_w2,
    "sliced_wasserstein": sliced_wasserstein,
    "sinkhorn": sinkhorn_divergence,
    "bures": bures_wasserstein_cov,
    "covariance_frobenius": covariance_frobenius,
    "pca_subspace": pca_subspace_distance,
    "mmd": mmd_rbf,
    "m_statistic": mmd_rbf,
    "energy": energy_distance,
    "ks": ks_stat,
    "cvm": cvm_stat,
    "density_ratio_proxy": density_ratio_proxy,
    "window_rbf": mmd_rbf,
}


def adjacent_window_scores(
    x: np.ndarray,
    window: int,
    metric: Callable[..., float],
    step: int = 1,
    smooth: int = 1,
    **metric_kwargs,
) -> np.ndarray:
    """Score each candidate split by comparing [t-window, t) to [t, t+window)."""

    data = as_2d(x)
    n = len(data)
    scores = np.full(n, np.nan)
    for t in range(window, n - window + 1, step):
        scores[t] = metric(data[t - window : t], data[t : t + window], **metric_kwargs)
    if smooth > 1:
        valid = np.isfinite(scores)
        filled = np.where(valid, scores, 0.0)
        counts = np.convolve(valid.astype(float), np.ones(smooth), mode="same")
        smoothed = np.convolve(filled, np.ones(smooth), mode="same") / np.maximum(counts, 1.0)
        scores[valid] = smoothed[valid]
    return scores


def select_peaks(
    scores: np.ndarray,
    threshold: float | None = None,
    max_changes: int | None = None,
    min_distance: int = 1,
) -> list[int]:
    """Select local maxima above a threshold with non-maximum suppression."""

    finite_scores = np.where(np.isfinite(scores), scores, -np.inf)
    candidates: list[tuple[float, int]] = []
    for i in range(1, len(finite_scores) - 1):
        if finite_scores[i] >= finite_scores[i - 1] and finite_scores[i] >= finite_scores[i + 1]:
            if threshold is None or finite_scores[i] >= threshold:
                candidates.append((float(finite_scores[i]), i))
    candidates.sort(reverse=True)
    selected: list[int] = []
    for _, idx in candidates:
        if all(abs(idx - prev) >= min_distance for prev in selected):
            selected.append(idx)
            if max_changes is not None and len(selected) >= max_changes:
                break
    return sorted(selected)


def threshold_from_quantile(scores: np.ndarray, quantile: float) -> float:
    finite = scores[np.isfinite(scores)]
    if len(finite) == 0:
        return float("inf")
    return float(np.quantile(finite, quantile))


def run_window_baseline(
    key: str,
    x: np.ndarray,
    window: int = 50,
    threshold: float | None = None,
    threshold_quantile: float = 0.995,
    max_changes: int | None = None,
    min_distance: int | None = None,
    smooth: int = 1,
    **kwargs,
) -> DetectionResult:
    metric = WINDOW_METRICS[key]
    scores = adjacent_window_scores(x, window=window, metric=metric, smooth=smooth, **kwargs)
    chosen_threshold = threshold if threshold is not None else threshold_from_quantile(scores, threshold_quantile)
    changepoints = select_peaks(
        scores,
        threshold=chosen_threshold,
        max_changes=max_changes,
        min_distance=min_distance or window,
    )
    return DetectionResult(
        method=key,
        changepoints=changepoints,
        scores=scores,
        threshold=chosen_threshold,
        metadata={"resource": resource_table([key])[0]},
    )


def run_watch_proxy(
    x: np.ndarray,
    reference_window: int = 100,
    window: int = 50,
    threshold: float | None = None,
    threshold_quantile: float = 0.995,
    min_distance: int | None = None,
) -> DetectionResult:
    """WATCH-inspired monitor against an initial empirical reference."""

    data = as_2d(x)
    reference = data[:reference_window]
    scores = np.full(len(data), np.nan)
    for t in range(reference_window, len(data) - window + 1):
        scores[t] = sliced_wasserstein(reference, data[t : t + window], n_projections=64, seed=0)
    chosen_threshold = threshold if threshold is not None else threshold_from_quantile(scores, threshold_quantile)
    return DetectionResult(
        "watch_proxy",
        select_peaks(scores, chosen_threshold, min_distance=min_distance or window),
        scores,
        chosen_threshold,
        {"resource": resource_table(["watch_proxy"])[0]},
    )


def run_sinkhorn_baseline(x: np.ndarray, window: int = 50, **kwargs) -> DetectionResult:
    if ot is None:
        return unavailable_result("sinkhorn", "POT is not installed. Install with `pip install POT`.")
    return run_window_baseline("sinkhorn", x, window=window, **kwargs)


def run_ruptures_baseline(
    key: str,
    x: np.ndarray,
    penalty: float = 10.0,
    n_bkps: int | None = None,
    min_size: int = 10,
) -> DetectionResult:
    if rpt is None:
        return unavailable_result(key, "ruptures is not installed. Install with `pip install ruptures`.")

    data = as_2d(x)
    model_by_key = {
        "pelt_l2": "l2",
        "pelt_rbf": "rbf",
        "pelt_normal": "normal",
        "binseg": "l2",
        "bottomup": "l2",
    }
    model = model_by_key[key]
    if key.startswith("pelt"):
        algo = rpt.Pelt(model=model, min_size=min_size).fit(data)
        bkps = algo.predict(pen=penalty)
    elif key == "binseg":
        algo = rpt.Binseg(model=model, min_size=min_size).fit(data)
        bkps = algo.predict(n_bkps=n_bkps) if n_bkps is not None else algo.predict(pen=penalty)
    else:
        algo = rpt.BottomUp(model=model, min_size=min_size).fit(data)
        bkps = algo.predict(n_bkps=n_bkps) if n_bkps is not None else algo.predict(pen=penalty)
    changepoints = [int(b) for b in bkps if b < len(data)]
    return DetectionResult(key, changepoints, metadata={"resource": resource_table([key])[0]})


def unavailable_result(key: str, reason: str) -> DetectionResult:
    return DetectionResult(
        method=key,
        changepoints=[],
        metadata={"unavailable": True, "reason": reason, "resource": resource_table([key])[0]},
    )


def aggregate_series(x: np.ndarray, transform: str = "mean") -> np.ndarray:
    data = as_2d(x)
    if transform == "mean":
        return np.mean(data, axis=1)
    if transform == "norm":
        return np.linalg.norm(data, axis=1)
    if transform == "squared_norm":
        return np.sum(data**2, axis=1)
    raise ValueError(f"Unknown transform: {transform}")


def run_cusum(
    key: str,
    x: np.ndarray,
    drift: float = 0.25,
    threshold: float | None = None,
    threshold_quantile: float = 0.995,
    min_distance: int = 25,
    burn_in: int = 50,
) -> DetectionResult:
    y = aggregate_series(x, "mean")
    if key == "cusum_vol":
        y = y**2
    burn_in = max(10, min(burn_in, len(y) // 4))
    mu = float(np.mean(y[:burn_in]))
    sigma = float(np.std(y[:burn_in]) + 1e-12)
    y = (y - mu) / sigma
    pos = np.zeros_like(y)
    neg = np.zeros_like(y)
    score = np.zeros_like(y)
    for i in range(1, len(y)):
        pos[i] = max(0.0, pos[i - 1] + y[i] - drift)
        neg[i] = min(0.0, neg[i - 1] + y[i] + drift)
        score[i] = max(pos[i], -neg[i])
    chosen_threshold = threshold if threshold is not None else threshold_from_quantile(score, threshold_quantile)
    return DetectionResult(
        key,
        select_peaks(score, chosen_threshold, min_distance=min_distance),
        score,
        chosen_threshold,
        {"resource": resource_table([key])[0]},
    )


def run_ewma_vol(
    x: np.ndarray,
    lam: float = 0.94,
    threshold: float | None = None,
    threshold_quantile: float = 0.995,
    min_distance: int = 25,
) -> DetectionResult:
    y = aggregate_series(x, "mean")
    var = np.zeros_like(y)
    var[0] = np.var(y) + 1e-12
    for i in range(1, len(y)):
        var[i] = lam * var[i - 1] + (1.0 - lam) * y[i - 1] ** 2
    innovations = np.abs(y**2 / (var + 1e-12) - 1.0)
    chosen_threshold = threshold if threshold is not None else threshold_from_quantile(innovations, threshold_quantile)
    return DetectionResult(
        "ewma_vol",
        select_peaks(innovations, chosen_threshold, min_distance=min_distance),
        innovations,
        chosen_threshold,
        {"resource": resource_table(["ewma_vol"])[0]},
    )


def run_bocpd_gaussian(
    x: np.ndarray,
    hazard: float = 1 / 200,
    threshold: float = 0.5,
    max_run_length: int = 300,
    min_distance: int = 25,
    burn_in: int = 50,
) -> DetectionResult:
    """
    Lightweight Adams-MacKay BOCPD for a univariate Gaussian with known variance.

    Variance is estimated on a burn-in prefix only (no full-series leakage).
    """

    y = aggregate_series(x, "mean")
    burn_in = max(10, min(burn_in, len(y) // 4))
    sigma2 = float(np.var(y[:burn_in]) + 1e-6)
    run_probs = np.array([1.0])
    sums = np.array([0.0])
    counts = np.array([0.0])
    cp_prob = np.zeros(len(y))
    for t, obs in enumerate(y):
        means = sums / np.maximum(counts, 1.0)
        pred_var = sigma2 * (1.0 + 1.0 / np.maximum(counts, 1.0))
        pred = stats.norm.pdf(obs, loc=means, scale=np.sqrt(pred_var)) + 1e-300
        growth = run_probs * pred * (1.0 - hazard)
        changepoint = np.sum(run_probs * pred * hazard)
        new_probs = np.r_[changepoint, growth]
        new_probs /= np.sum(new_probs)
        cp_prob[t] = new_probs[0]
        sums = np.r_[0.0, sums + obs][: max_run_length + 1]
        counts = np.r_[0.0, counts + 1.0][: max_run_length + 1]
        run_probs = new_probs[: max_run_length + 1]
        run_probs /= np.sum(run_probs)
    return DetectionResult(
        "bocpd_gaussian",
        select_peaks(cp_prob, threshold=threshold, min_distance=min_distance),
        cp_prob,
        threshold,
        {"hazard": hazard, "resource": resource_table(["bocpd_gaussian"])[0]},
    )


def run_gaussian_hmm(
    x: np.ndarray,
    n_states: int = 2,
    min_distance: int = 25,
    random_state: int = 0,
) -> DetectionResult:
    if GaussianHMM is None:
        return unavailable_result("gaussian_hmm", "hmmlearn is not installed. Install with `pip install hmmlearn`.")
    data = as_2d(x)
    model = GaussianHMM(n_components=n_states, covariance_type="full", random_state=random_state, n_iter=200)
    states = model.fit(data).predict(data)
    cps = [i for i in range(1, len(states)) if states[i] != states[i - 1]]
    score = np.r_[0.0, (states[1:] != states[:-1]).astype(float)]
    return DetectionResult(
        "gaussian_hmm",
        select_peaks(score, threshold=0.5, min_distance=min_distance) or cps,
        score,
        0.5,
        {"states": states.tolist(), "resource": resource_table(["gaussian_hmm"])[0]},
    )


def run_coordinate_w2_matched_filter(x: np.ndarray, window: int = 50, **kwargs) -> DetectionResult:
    from changept_detection.method.proposed import matched_filter_1d

    metric = WINDOW_METRICS["coordinate_w2_window_scan"]
    skip = {"threshold", "threshold_quantile", "min_distance", "max_changes", "window", "smooth", "burn_in"}
    extra = {k: v for k, v in kwargs.items() if k not in skip}
    scores = adjacent_window_scores(x, window=window, metric=metric, smooth=1, **extra)
    scores = matched_filter_1d(scores, width=window)
    threshold = kwargs.get("threshold")
    if threshold is None:
        threshold = threshold_from_quantile(scores, kwargs.get("threshold_quantile", 0.995))
    changepoints = select_peaks(
        scores,
        threshold=threshold,
        min_distance=kwargs.get("min_distance") or window,
        max_changes=kwargs.get("max_changes"),
    )
    return DetectionResult(
        "coordinate_w2_matched_filter",
        changepoints,
        scores,
        threshold,
        {"resource": resource_table(["coordinate_w2_matched_filter"])[0]},
    )


def run_baseline(key: str, x: np.ndarray, **kwargs) -> DetectionResult:
    """Dispatch a detector by registry key."""

    if key == "sinkhorn":
        return run_sinkhorn_baseline(x, **kwargs)
    if key == "coordinate_w2_matched_filter":
        return run_coordinate_w2_matched_filter(x, **kwargs)
    if key in PROPOSED_METHOD_KEYS:
        from changept_detection.method.proposed import PROPOSED_DISPATCH

        dispatch = PROPOSED_DISPATCH.get(key)
        if dispatch is None:
            raise KeyError(key)
        return dispatch(x, **kwargs)
    if key in WINDOW_METRICS:
        resolved = "coordinate_w2_window_scan" if key == "local_w2t" else key
        return run_window_baseline(resolved, x, **kwargs)
    if key == "watch_proxy":
        return run_watch_proxy(x, **kwargs)
    if key in {"pelt_l2", "pelt_rbf", "pelt_normal", "binseg", "bottomup"}:
        return run_ruptures_baseline(key, x, **kwargs)
    if key in {"cusum_mean", "cusum_vol"}:
        return run_cusum(key, x, **kwargs)
    if key == "ewma_vol":
        return run_ewma_vol(x, **kwargs)
    if key == "bocpd_gaussian":
        return run_bocpd_gaussian(x, **kwargs)
    if key == "gaussian_hmm":
        return run_gaussian_hmm(x, **kwargs)
    if key == "markov_switching":
        return unavailable_result(key, "Use statsmodels.tsa.regime_switching for full Hamilton-style fits.")
    raise KeyError(f"Unknown baseline key: {key}")


def match_changepoints(
    truth: Iterable[int],
    detected: Iterable[int],
    tolerance: int,
) -> tuple[int, int, int, list[int]]:
    """Greedy one-to-one matching of detected changepoints to ground truth."""

    truth_list = list(truth)
    detected_list = list(detected)
    pairs = sorted(
        (abs(t - d), t_idx, d_idx)
        for t_idx, t in enumerate(truth_list)
        for d_idx, d in enumerate(detected_list)
        if abs(t - d) <= tolerance
    )
    used_t: set[int] = set()
    used_d: set[int] = set()
    errors: list[int] = []
    for error, t_idx, d_idx in pairs:
        if t_idx in used_t or d_idx in used_d:
            continue
        used_t.add(t_idx)
        used_d.add(d_idx)
        errors.append(error)
    tp = len(errors)
    fp = len(detected_list) - tp
    fn = len(truth_list) - tp
    return tp, fp, fn, errors


def detection_metrics(truth: Iterable[int], detected: Iterable[int], tolerance: int) -> dict[str, float]:
    tp, fp, fn, errors = match_changepoints(truth, detected, tolerance)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_abs_error": float(np.mean(errors)) if errors else np.nan,
        "max_abs_error": float(np.max(errors)) if errors else np.nan,
    }


def duplicate_rate(truth: Iterable[int], detected: Iterable[int], event_window: int) -> float:
    truth_list = list(truth)
    detected_list = list(detected)
    if not truth_list:
        return 0.0
    extras = 0
    for tau in truth_list:
        hits = [d for d in detected_list if abs(d - tau) <= event_window]
        extras += max(0, len(hits) - 1)
    return extras / len(truth_list)


def label_matching_accuracy(true_labels: np.ndarray, pred_labels: np.ndarray) -> float:
    """Best label-permutation accuracy for small regime-label experiments."""

    true_labels = np.asarray(true_labels)
    pred_labels = np.asarray(pred_labels)
    unique_true = np.unique(true_labels)
    unique_pred = np.unique(pred_labels)
    if len(unique_true) > 8 or len(unique_pred) > 8:
        # Greedy fallback for larger K.
        total = 0
        for label in unique_pred:
            mask = pred_labels == label
            if np.any(mask):
                total += np.max(np.bincount(true_labels[mask].astype(int)))
        return float(total / len(true_labels))
    best = 0
    for true_order in permutations(unique_true, min(len(unique_true), len(unique_pred))):
        mapping = dict(zip(unique_pred, true_order))
        mapped = np.array([mapping.get(label, -999999) for label in pred_labels])
        best = max(best, int(np.sum(mapped == true_labels)))
    return float(best / len(true_labels))


def simple_kmeans(
    x: np.ndarray,
    n_clusters: int,
    random_state: int = 0,
    max_iter: int = 100,
) -> np.ndarray:
    """Small NumPy K-means used to avoid optional BLAS/threadpool issues."""

    rng = np.random.default_rng(random_state)
    x = np.asarray(x, dtype=float)
    if len(x) < n_clusters:
        raise ValueError("Need at least as many samples as clusters")
    centers = x[rng.choice(len(x), size=n_clusters, replace=False)]
    labels = np.zeros(len(x), dtype=int)
    for _ in range(max_iter):
        distances = cdist(x, centers, metric="sqeuclidean")
        new_labels = np.argmin(distances, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for k in range(n_clusters):
            mask = labels == k
            if np.any(mask):
                centers[k] = x[mask].mean(axis=0)
            else:
                centers[k] = x[int(rng.integers(0, len(x)))]
    return labels


def cluster_rolling_windows(
    x: np.ndarray,
    window: int,
    n_clusters: int,
    random_state: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Simple detect-then-cluster baseline features for S7 interpretability."""

    data = as_2d(x)
    centers = np.arange(window, len(data) + 1)
    features = []
    for end in centers:
        block = data[end - window : end]
        features.append(np.r_[np.mean(block, axis=0), np.std(block, axis=0), covariance_matrix(block).ravel()])
    features_arr = np.asarray(features)
    labels = simple_kmeans(features_arr, n_clusters=n_clusters, random_state=random_state)
    return centers, labels


def clustering_metrics(true_labels: np.ndarray, pred_labels: np.ndarray) -> dict[str, float]:
    return {
        "label_accuracy": label_matching_accuracy(true_labels, pred_labels),
        "ari": float(adjusted_rand_score(true_labels, pred_labels)),
        "nmi": float(normalized_mutual_info_score(true_labels, pred_labels)),
    }
