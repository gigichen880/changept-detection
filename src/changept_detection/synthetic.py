"""
Synthetic data generators and experiment orchestration for docs/experiment_plan.md.

The suite covers S0-S7 from the plan: mean/variance shifts, tail shifts,
mixture reweighting, fixed-marginal correlation shifts, low-rank factor shocks,
transient-vs-persistent shocks, duplicate local peak suppression, and recurring
regime-label interpretability.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Any

import numpy as np

from changept_detection.baselines import (
    BASELINE_RESOURCES,
    cluster_rolling_windows,
    clustering_metrics,
    detection_metrics,
    duplicate_rate,
    run_baseline,
)


@dataclass
class SyntheticCase:
    """One generated CPD dataset with exact labels."""

    experiment: str
    name: str
    x: np.ndarray
    changepoints: list[int]
    params: dict[str, Any]
    regime_labels: np.ndarray | None = None


@dataclass
class ExperimentResult:
    experiment: str
    case_name: str
    method: str
    changepoints: list[int]
    metrics: dict[str, float]
    params: dict[str, Any]
    resource: dict[str, str]
    metadata: dict[str, Any] = field(default_factory=dict)


BASELINE_SETS: dict[str, list[str]] = {
    "S0": [
        "pelt_l2",
        "pelt_normal",
        "binseg",
        "cusum_mean",
        "cusum_vol",
        "bocpd_gaussian",
        "local_w2t",
        "proposed_local_global",
    ],
    "S1": [
        "ewma_vol",
        "cusum_vol",
        "ks",
        "cvm",
        "mmd",
        "energy",
        "pelt_rbf",
        "local_w2t",
        "proposed_local_global",
    ],
    "S2": [
        "ks",
        "mmd",
        "energy",
        "pelt_rbf",
        "local_w2t",
        "bocpd_gaussian",
        "proposed_local_global",
    ],
    "S3": [
        "coordinate_w2t",
        "ks",
        "pelt_l2",
        "pelt_rbf",
        "mmd",
        "energy",
        "bures",
        "sliced_wasserstein",
        "sinkhorn",
        "proposed_local_global",
    ],
    "S4": [
        "pelt_rbf",
        "mmd",
        "sliced_wasserstein",
        "sinkhorn",
        "covariance_frobenius",
        "pca_subspace",
        "bures",
        "proposed_local_global",
    ],
    "S5": [
        "local_w2t",
        "mmd",
        "pelt_rbf",
        "bocpd_gaussian",
        "cusum_mean",
        "ewma_vol",
        "proposed_local_global",
    ],
    "S6": [
        "local_w2t",
        "mmd",
        "window_rbf",
        "proposed_local_global",
    ],
    "S7": [
        "gaussian_hmm",
        "markov_switching",
        "local_w2t",
        "proposed_local_global",
    ],
}


EXPERIMENT_DESCRIPTIONS = {
    "S0": "Sanity-check Gaussian mean/variance shifts.",
    "S1": "Variance-matched Student-t tail shift.",
    "S2": "Scenario-mixture weight shift.",
    "S3": "Fixed-marginal correlation crisis.",
    "S4": "Low-rank factor covariance shock.",
    "S5": "Transient shock versus persistent regime.",
    "S6": "Duplicate local peak suppression.",
    "S7": "Recurring-regime posterior/label interpretability.",
}


def equicorrelation(d: int, rho: float) -> np.ndarray:
    lower_bound = -1.0 / max(d - 1, 1)
    if not lower_bound < rho < 1.0:
        raise ValueError(f"rho={rho} is not positive definite for d={d}")
    return (1.0 - rho) * np.eye(d) + rho * np.ones((d, d))


def make_factor_vectors(
    d: int,
    k: int,
    sparsity: str,
    rng: np.random.Generator,
) -> np.ndarray:
    vectors = []
    for _ in range(k):
        v = np.zeros(d)
        if sparsity == "sector-sparse":
            width = max(2, d // 10)
            start = int(rng.integers(0, max(1, d - width + 1)))
            v[start : start + width] = rng.normal(size=width)
        elif sparsity == "random sparse":
            width = max(2, int(np.sqrt(d)))
            idx = rng.choice(d, size=width, replace=False)
            v[idx] = rng.normal(size=width)
        else:
            v = rng.normal(size=d)
        v /= np.linalg.norm(v) + 1e-12
        vectors.append(v)
    return np.asarray(vectors)


def apply_ar1(x: np.ndarray, phi: float, rng: np.random.Generator) -> np.ndarray:
    y = np.array(x, copy=True)
    for t in range(1, len(y)):
        y[t] = phi * y[t - 1] + np.sqrt(1.0 - phi**2) * y[t]
    return y


def generate_s0(
    seed: int = 0,
    n_per_segment: int = 250,
    d: int = 5,
    mean_shift: float = 0.2,
    volatility_ratio: float = 1.25,
) -> SyntheticCase:
    rng = np.random.default_rng(seed)
    mu1 = np.zeros(d)
    direction = np.ones(d) / np.sqrt(d)
    mu2 = mean_shift * direction
    x1 = rng.normal(loc=mu1, scale=1.0, size=(n_per_segment, d))
    x2 = rng.normal(loc=mu2, scale=volatility_ratio, size=(n_per_segment, d))
    x = np.vstack([x1, x2])
    return SyntheticCase(
        "S0",
        "gaussian_mean_variance",
        x,
        [n_per_segment],
        {
            "seed": seed,
            "n_per_segment": n_per_segment,
            "d": d,
            "mean_shift": mean_shift,
            "volatility_ratio": volatility_ratio,
        },
    )


def generate_s1(
    seed: int = 0,
    n_per_segment: int = 250,
    nu: float = 6,
    n_changepoints: int = 1,
    garch_noise: str = "none",
) -> SyntheticCase:
    rng = np.random.default_rng(seed)
    segments = []
    labels = []
    n_segments = n_changepoints + 1
    t_scale = np.sqrt((nu - 2.0) / nu)
    for segment_id in range(n_segments):
        if segment_id % 2 == 0:
            segment = rng.normal(0.0, 1.0, size=(n_per_segment, 1))
        else:
            segment = t_scale * rng.standard_t(df=nu, size=(n_per_segment, 1))
        segments.append(segment)
        labels.extend([segment_id % 2] * n_per_segment)
    x = np.vstack(segments)
    if garch_noise != "none":
        alpha, beta = (0.04, 0.90) if garch_noise == "mild" else (0.08, 0.88)
        var = np.ones(len(x))
        eps = x[:, 0].copy()
        for t in range(1, len(eps)):
            var[t] = 1.0 - alpha - beta + alpha * eps[t - 1] ** 2 + beta * var[t - 1]
            eps[t] = np.sqrt(max(var[t], 1e-8)) * eps[t]
        x[:, 0] = eps / (np.std(eps) + 1e-12)
    cps = [n_per_segment * i for i in range(1, n_segments)]
    return SyntheticCase(
        "S1",
        "variance_matched_tail_shift",
        x,
        cps,
        {
            "seed": seed,
            "n_per_segment": n_per_segment,
            "nu": nu,
            "n_changepoints": n_changepoints,
            "garch_noise": garch_noise,
        },
        np.asarray(labels),
    )


def sample_mixture(
    rng: np.random.Generator,
    n: int,
    weight_left: float,
    a: float,
    sigma: float,
) -> np.ndarray:
    left_component = rng.random(n) < weight_left
    means = np.where(left_component, -a, a)
    return rng.normal(means, sigma, size=n)


def generate_s2(
    seed: int = 0,
    n_per_segment: int = 250,
    delta: float = 0.1,
    mode_separation: float = 3.0,
    serial_dependence: str = "IID",
) -> SyntheticCase:
    rng = np.random.default_rng(seed)
    sigma = 1.0
    a = mode_separation * sigma
    x1 = sample_mixture(rng, n_per_segment, 0.5, a, sigma)[:, None]
    x2 = sample_mixture(rng, n_per_segment, 0.5 + delta, a, sigma)[:, None]
    x = np.vstack([x1, x2])
    if serial_dependence == "AR(1)":
        x = apply_ar1(x, 0.4, rng)
    elif serial_dependence == "block dependence":
        block = 5
        for start in range(0, len(x), block):
            x[start : start + block] += rng.normal(0.0, 0.2)
    return SyntheticCase(
        "S2",
        "scenario_mixture_weight_shift",
        x,
        [n_per_segment],
        {
            "seed": seed,
            "n_per_segment": n_per_segment,
            "delta": delta,
            "mode_separation": mode_separation,
            "serial_dependence": serial_dependence,
        },
    )


def generate_s3(
    seed: int = 0,
    n_per_segment: int = 250,
    d: int = 20,
    rho1: float = 0.2,
    delta_rho: float = 0.2,
) -> SyntheticCase:
    rng = np.random.default_rng(seed)
    rho2 = rho1 + delta_rho
    x1 = rng.multivariate_normal(np.zeros(d), equicorrelation(d, rho1), size=n_per_segment)
    x2 = rng.multivariate_normal(np.zeros(d), equicorrelation(d, rho2), size=n_per_segment)
    return SyntheticCase(
        "S3",
        "fixed_marginal_correlation_crisis",
        np.vstack([x1, x2]),
        [n_per_segment],
        {
            "seed": seed,
            "n_per_segment": n_per_segment,
            "d": d,
            "rho1": rho1,
            "delta_rho": delta_rho,
        },
    )


def generate_s4(
    seed: int = 0,
    n_per_segment: int = 250,
    d: int = 50,
    epsilon: float = 0.5,
    n_factors: int = 1,
    sparsity: str = "dense",
) -> SyntheticCase:
    rng = np.random.default_rng(seed)
    vectors = make_factor_vectors(d, n_factors, sparsity, rng)
    shock_cov = np.eye(d)
    for v in vectors:
        shock_cov += epsilon * np.outer(v, v)
    x1 = rng.normal(0.0, 1.0, size=(n_per_segment, d))
    x2 = rng.multivariate_normal(np.zeros(d), shock_cov, size=n_per_segment)
    return SyntheticCase(
        "S4",
        "low_rank_factor_shock",
        np.vstack([x1, x2]),
        [n_per_segment],
        {
            "seed": seed,
            "n_per_segment": n_per_segment,
            "d": d,
            "epsilon": epsilon,
            "n_factors": n_factors,
            "sparsity": sparsity,
        },
    )


def generate_s5(
    seed: int = 0,
    n_before: int = 250,
    shock_length: int = 5,
    n_after: int = 250,
    shock_type: str = "volatility",
    magnitude: str = "medium",
    persistent: bool = False,
) -> SyntheticCase:
    rng = np.random.default_rng(seed)
    mag = {"small": 1.5, "medium": 2.5, "large": 4.0}[magnitude]
    calm1 = rng.normal(0.0, 1.0, size=(n_before, 1))
    if shock_type == "mean":
        shock = rng.normal(mag, 1.0, size=(shock_length, 1))
        persistent_segment = rng.normal(mag, 1.0, size=(n_after, 1))
    elif shock_type == "tail":
        shock = rng.standard_t(df=4, size=(shock_length, 1)) * mag / np.sqrt(2.0)
        persistent_segment = rng.standard_t(df=4, size=(n_after, 1)) * mag / np.sqrt(2.0)
    else:
        shock = rng.normal(0.0, mag, size=(shock_length, 1))
        persistent_segment = rng.normal(0.0, mag, size=(n_after, 1))
    if persistent:
        x = np.vstack([calm1, persistent_segment])
        cps = [n_before]
        name = "persistent_regime_shift"
    else:
        calm2 = rng.normal(0.0, 1.0, size=(n_after, 1))
        x = np.vstack([calm1, shock, calm2])
        cps = [n_before, n_before + shock_length]
        name = "transient_shock"
    return SyntheticCase(
        "S5",
        name,
        x,
        cps,
        {
            "seed": seed,
            "n_before": n_before,
            "shock_length": shock_length,
            "n_after": n_after,
            "shock_type": shock_type,
            "magnitude": magnitude,
            "persistent": persistent,
        },
    )


def generate_s6(
    seed: int = 0,
    n_per_segment: int = 250,
    shift_family: str = "tail",
    signal_strength: str = "medium",
) -> SyntheticCase:
    if shift_family == "correlation":
        strength = {"weak": 0.05, "medium": 0.2, "strong": 0.4}[signal_strength]
        case = generate_s3(seed=seed, n_per_segment=n_per_segment, d=20, rho1=0.2, delta_rho=strength)
    elif shift_family == "factor":
        strength = {"weak": 0.1, "medium": 0.5, "strong": 1.0}[signal_strength]
        case = generate_s4(seed=seed, n_per_segment=n_per_segment, d=30, epsilon=strength)
    elif shift_family == "mixture":
        strength = {"weak": 0.05, "medium": 0.1, "strong": 0.3}[signal_strength]
        case = generate_s2(seed=seed, n_per_segment=n_per_segment, delta=strength)
    else:
        nu = {"weak": 20, "medium": 8, "strong": 4}[signal_strength]
        case = generate_s1(seed=seed, n_per_segment=n_per_segment, nu=nu)
    case.experiment = "S6"
    case.name = f"duplicate_peak_{shift_family}"
    case.params.update({"shift_family": shift_family, "signal_strength": signal_strength})
    return case


def generate_s7(
    seed: int = 0,
    regime_duration: int = 100,
    d: int = 5,
    similarity: str = "medium",
) -> SyntheticCase:
    rng = np.random.default_rng(seed)
    strength = {"easy": 1.0, "medium": 0.6, "hard": 0.3}[similarity]
    order = [0, 1, 0, 2, 1, 3]
    segments = []
    labels = []
    for label in order:
        if label == 0:
            segment = rng.normal(0.0, 1.0, size=(regime_duration, d))
        elif label == 1:
            segment = rng.standard_t(df=4, size=(regime_duration, d)) * np.sqrt(0.5) * (1.0 + strength)
        elif label == 2:
            cov = equicorrelation(d, min(0.8, 0.2 + strength * 0.5))
            segment = rng.multivariate_normal(np.zeros(d), cov, size=regime_duration)
        else:
            v = make_factor_vectors(d, 1, "dense", rng)[0]
            cov = np.eye(d) + strength * np.outer(v, v)
            segment = rng.multivariate_normal(np.zeros(d), cov, size=regime_duration)
        segments.append(segment)
        labels.extend([label] * regime_duration)
    cps = [regime_duration * i for i in range(1, len(order))]
    return SyntheticCase(
        "S7",
        "recurring_regimes",
        np.vstack(segments),
        cps,
        {
            "seed": seed,
            "regime_duration": regime_duration,
            "d": d,
            "similarity": similarity,
            "regime_order": order,
        },
        np.asarray(labels),
    )


GENERATORS = {
    "S0": generate_s0,
    "S1": generate_s1,
    "S2": generate_s2,
    "S3": generate_s3,
    "S4": generate_s4,
    "S5": generate_s5,
    "S6": generate_s6,
    "S7": generate_s7,
}


def quick_grid(experiment: str) -> list[dict[str, Any]]:
    """Small grids intended for smoke tests and local iteration."""

    if experiment == "S0":
        return [{"mean_shift": s, "volatility_ratio": v, "d": 5, "n_per_segment": 120} for s, v in [(0.2, 1.25), (0.5, 1.5)]]
    if experiment == "S1":
        return [{"nu": nu, "n_per_segment": 120} for nu in [6, 20]]
    if experiment == "S2":
        return [{"delta": delta, "mode_separation": 3.0, "n_per_segment": 120} for delta in [0.05, 0.2]]
    if experiment == "S3":
        return [{"delta_rho": dr, "rho1": 0.2, "d": 10, "n_per_segment": 120} for dr in [0.1, 0.3]]
    if experiment == "S4":
        return [{"epsilon": eps, "d": 20, "n_per_segment": 120} for eps in [0.1, 0.5]]
    if experiment == "S5":
        return [
            {"shock_length": 5, "magnitude": "large", "persistent": False},
            {"shock_length": 50, "magnitude": "medium", "persistent": True},
        ]
    if experiment == "S6":
        return [{"shift_family": fam, "signal_strength": "medium", "n_per_segment": 120} for fam in ["tail", "correlation"]]
    if experiment == "S7":
        return [{"regime_duration": 80, "d": 5, "similarity": "medium"}]
    raise KeyError(f"Unknown experiment: {experiment}")


def full_grid(experiment: str) -> list[dict[str, Any]]:
    """Broader grids following the difficulty knobs in docs/experiment_plan.md."""

    if experiment == "S0":
        return [
            {"mean_shift": s, "volatility_ratio": v, "n_per_segment": n, "d": d}
            for s, v, n, d in product([0.05, 0.1, 0.2, 0.5, 1.0], [1.05, 1.25, 1.5], [100, 250], [1, 5, 20])
        ]
    if experiment == "S1":
        return [
            {"nu": nu, "n_per_segment": n, "garch_noise": garch, "n_changepoints": cps}
            for nu, n, garch, cps in product([4, 6, 8, 12, 20, 50], [100, 250], ["none", "mild"], [1, 3])
        ]
    if experiment == "S2":
        return [
            {"delta": delta, "mode_separation": sep, "n_per_segment": n, "serial_dependence": dep}
            for delta, sep, n, dep in product([0.02, 0.05, 0.1, 0.2, 0.3], [1, 2, 3, 5], [100, 250], ["IID", "AR(1)"])
        ]
    if experiment == "S3":
        return [
            {"delta_rho": dr, "rho1": rho, "d": d, "n_per_segment": n}
            for dr, rho, d, n in product([0.05, 0.1, 0.2, 0.4], [0.0, 0.2, 0.5], [5, 20, 50], [100, 250])
            if rho + dr < 0.95
        ]
    if experiment == "S4":
        return [
            {"epsilon": eps, "d": d, "n_factors": k, "sparsity": sparsity, "n_per_segment": n}
            for eps, d, k, sparsity, n in product([0.05, 0.1, 0.2, 0.5, 1.0], [10, 50, 100], [1, 3], ["dense", "sector-sparse"], [100, 250])
        ]
    if experiment == "S5":
        return [
            {"shock_length": m, "magnitude": mag, "shock_type": typ, "persistent": persistent}
            for m, mag, typ, persistent in product([1, 2, 5, 10, 20], ["small", "medium", "large"], ["mean", "volatility", "tail"], [False, True])
        ]
    if experiment == "S6":
        return [
            {"shift_family": fam, "signal_strength": strength, "n_per_segment": n}
            for fam, strength, n in product(["tail", "mixture", "correlation", "factor"], ["weak", "medium", "strong"], [100, 250])
        ]
    if experiment == "S7":
        return [
            {"regime_duration": duration, "d": 5, "similarity": similarity}
            for duration, similarity in product([50, 100, 250], ["easy", "medium", "hard"])
        ]
    raise KeyError(f"Unknown experiment: {experiment}")


def baseline_kwargs(key: str, case: SyntheticCase, window: int, n_bkps: int | None) -> dict[str, Any]:
    """Route only compatible options to each detector family."""

    tolerance = max(5, window // 2)
    if key in {"pelt_l2", "pelt_rbf", "pelt_normal", "binseg", "bottomup"}:
        return {"penalty": 8.0, "n_bkps": n_bkps, "min_size": max(5, window // 2)}
    if key in {"cusum_mean", "cusum_vol"}:
        return {"threshold_quantile": 0.99, "min_distance": window}
    if key == "ewma_vol":
        return {"threshold_quantile": 0.99, "min_distance": window}
    if key == "bocpd_gaussian":
        return {"hazard": 1.0 / max(window * 4, 1), "threshold": 0.2, "min_distance": window}
    if key == "gaussian_hmm":
        return {"n_states": min(4, max(2, len(case.changepoints) + 1)), "min_distance": window}
    if key == "proposed_local_global":
        metric = "bures" if case.experiment in {"S3", "S4"} else "sliced_wasserstein"
        return {
            "window": window,
            "threshold_quantile": 0.98,
            "min_persistence": 2,
            "min_distance": window,
            "metric": metric,
        }
    if key == "sinkhorn":
        return {"window": window, "threshold_quantile": 0.99, "min_distance": window}
    if key == "sliced_wasserstein":
        return {
            "window": window,
            "threshold_quantile": 0.99,
            "min_distance": window,
            "n_projections": 64,
            "seed": int(case.params.get("seed", 0)),
        }
    if key in {"bures", "covariance_frobenius", "pca_subspace"}:
        return {"window": window, "threshold_quantile": 0.99, "min_distance": window}
    return {"window": window, "threshold_quantile": 0.99, "min_distance": window, "smooth": max(1, window // 10)}


def run_case(
    case: SyntheticCase,
    baselines: list[str] | None = None,
    window: int | None = None,
    n_bkps: int | None = None,
) -> list[ExperimentResult]:
    if baselines is None:
        baselines = BASELINE_SETS[case.experiment]
    window = window or max(20, min(80, len(case.x) // 8))
    n_bkps = len(case.changepoints) if n_bkps is None else n_bkps
    tolerance = max(5, window // 2)
    results = []
    for key in baselines:
        kwargs = baseline_kwargs(key, case, window=window, n_bkps=n_bkps)
        result = run_baseline(key, case.x, **kwargs)
        metrics = detection_metrics(case.changepoints, result.changepoints, tolerance=tolerance)
        if case.experiment == "S6":
            metrics["duplicate_rate"] = duplicate_rate(case.changepoints, result.changepoints, event_window=window)
        if result.metadata.get("unavailable"):
            metrics = {**metrics, "unavailable": 1.0}
        else:
            metrics = {**metrics, "unavailable": 0.0}
        resource = result.metadata.get("resource", BASELINE_RESOURCES[key])
        if not isinstance(resource, dict):
            resource = resource.__dict__
        results.append(
            ExperimentResult(
                experiment=case.experiment,
                case_name=case.name,
                method=key,
                changepoints=result.changepoints,
                metrics=metrics,
                params=case.params,
                resource=resource,
                metadata={k: v for k, v in result.metadata.items() if k != "resource"},
            )
        )
    if case.experiment == "S7" and case.regime_labels is not None:
        centers, labels = cluster_rolling_windows(case.x, window=window, n_clusters=len(np.unique(case.regime_labels)))
        true_labels = case.regime_labels[centers - 1]
        metrics = clustering_metrics(true_labels, labels)
        results.append(
            ExperimentResult(
                experiment=case.experiment,
                case_name=case.name,
                method="rolling_feature_kmeans",
                changepoints=[],
                metrics={**metrics, "unavailable": 0.0},
                params=case.params,
                resource={
                    "key": "rolling_feature_kmeans",
                    "name": "K-means on rolling features",
                    "category": "Regime-label baseline",
                    "source": "docs/experiment_plan.md S7 baseline",
                    "url": "docs/experiment_plan.md",
                    "implementation": "KMeans on rolling mean/std/covariance features.",
                    "notes": "",
                },
            )
        )
    return results


def make_cases(
    experiment: str,
    grid: str = "quick",
    seeds: int | list[int] = 1,
) -> list[SyntheticCase]:
    if isinstance(seeds, int):
        seed_values = list(range(seeds))
    else:
        seed_values = seeds
    params_grid = quick_grid(experiment) if grid == "quick" else full_grid(experiment)
    generator = GENERATORS[experiment]
    cases = []
    for params in params_grid:
        for seed in seed_values:
            cases.append(generator(seed=seed, **params))
    return cases


def run_synthetic_suite(
    experiments: list[str] | None = None,
    grid: str = "quick",
    seeds: int | list[int] = 1,
    baselines: list[str] | None = None,
    window: int | None = None,
) -> list[ExperimentResult]:
    experiments = experiments or list(BASELINE_SETS)
    all_results: list[ExperimentResult] = []
    for experiment in experiments:
        for case in make_cases(experiment, grid=grid, seeds=seeds):
            all_results.extend(run_case(case, baselines=baselines, window=window))
    return all_results


def flatten_result(result: ExperimentResult) -> dict[str, Any]:
    row: dict[str, Any] = {
        "experiment": result.experiment,
        "case_name": result.case_name,
        "method": result.method,
        "changepoints": ";".join(map(str, result.changepoints)),
        "resource_name": result.resource.get("name", ""),
        "resource_source": result.resource.get("source", ""),
        "resource_url": result.resource.get("url", ""),
    }
    row.update({f"param_{k}": v for k, v in result.params.items() if np.isscalar(v) or isinstance(v, str)})
    row.update(result.metrics)
    row.update({f"meta_{k}": v for k, v in result.metadata.items() if np.isscalar(v) or isinstance(v, str)})
    return row
