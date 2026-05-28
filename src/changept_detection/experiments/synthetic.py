"""
Synthetic data generators and experiment orchestration for docs/experiment_plan.md Set A.

Experiment ids match docs/experiment_plan.md Set A: A1–A7 plus A_regime extension.
Method lists and metrics are defined in ``experiments.spec``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Any

import numpy as np

from changept_detection.baselines.core import (
    BASELINE_RESOURCES,
    cluster_rolling_windows,
    clustering_metrics,
    detection_metrics,
    run_baseline,
)
from changept_detection.experiments.calibration import (
    CalibratedThresholds,
    calibrate_for_case,
    calibration_config_key,
)
from changept_detection.experiments.metrics import (
    score_diagnostics,
    s6_metrics,
    with_localization_alias,
)
from changept_detection.experiments.spec import (
    BASELINE_SETS,
    EXPERIMENT_DESCRIPTIONS,
    PROPOSED_PRIMARY,
    detection_tolerance,
)
from changept_detection.method.proposed import regime_labels_from_prototypes


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


def generate_a1(
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
        "A1",
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


def generate_a2(
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
        "A2",
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


def generate_a3(
    seed: int = 0,
    n_per_segment: int = 250,
    delta: float = 0.1,
    mode_separation: float = 3.0,
    serial_dependence: str = "IID",
    centered: bool = False,
    demeaned: bool | None = None,
) -> SyntheticCase:
    rng = np.random.default_rng(seed)
    sigma = 1.0
    a = mode_separation * sigma
    use_demeaned = demeaned if demeaned is not None else centered
    if use_demeaned:
        # Symmetric modes at +/-a; reweight then remove segment mean so E[X] ~ 0.
        def draw(n: int, w_left: float) -> np.ndarray:
            z = rng.random(n) < w_left
            return np.where(z, -a, a) + rng.normal(0.0, sigma, size=n)

        x1 = draw(n_per_segment, 0.5)
        x2 = draw(n_per_segment, 0.5 + delta)
        x2 = x2 - (-2.0 * a * delta)  # E[draw(0.5+delta)] = -2a*delta
        x1 = x1[:, None]
        x2 = x2[:, None]
    else:
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
        "A3",
        "scenario_mixture_weight_shift",
        x,
        [n_per_segment],
        {
            "seed": seed,
            "n_per_segment": n_per_segment,
            "delta": delta,
            "mode_separation": mode_separation,
            "serial_dependence": serial_dependence,
            "centered": use_demeaned,
            "demeaned": use_demeaned,
        },
    )


def generate_a4(
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
        "A4",
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


def generate_a5(
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
        "A5",
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


def generate_a6(
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
        shock_length = 0  # not used in persistent branch
    else:
        calm2 = rng.normal(0.0, 1.0, size=(n_after, 1))
        x = np.vstack([calm1, shock, calm2])
        cps = [n_before, n_before + shock_length]
        name = "transient_shock"
    return SyntheticCase(
        "A6",
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


def generate_a7(
    seed: int = 0,
    n_per_segment: int = 250,
    shift_family: str = "tail",
    signal_strength: str = "medium",
    window_length: int | None = None,
    noise_level: str = "low",
) -> SyntheticCase:
    if shift_family == "correlation":
        strength = {"weak": 0.05, "medium": 0.2, "strong": 0.4}[signal_strength]
        case = generate_a4(seed=seed, n_per_segment=n_per_segment, d=20, rho1=0.2, delta_rho=strength)
    elif shift_family == "factor":
        strength = {"weak": 0.1, "medium": 0.5, "strong": 1.0}[signal_strength]
        case = generate_a5(seed=seed, n_per_segment=n_per_segment, d=30, epsilon=strength)
    elif shift_family == "mixture":
        strength = {"weak": 0.05, "medium": 0.1, "strong": 0.3}[signal_strength]
        case = generate_a3(seed=seed, n_per_segment=n_per_segment, delta=strength)
    else:
        nu = {"weak": 20, "medium": 8, "strong": 4}[signal_strength]
        case = generate_a2(seed=seed, n_per_segment=n_per_segment, nu=nu)
    case.experiment = "A7"
    case.name = f"duplicate_peak_{shift_family}"
    noise_scale = {"low": 0.05, "medium": 0.15, "high": 0.35}[noise_level]
    rng = np.random.default_rng(seed + 99)
    case.x = case.x + rng.normal(0.0, noise_scale, size=case.x.shape)
    case.params.update(
        {
            "shift_family": shift_family,
            "signal_strength": signal_strength,
            "window_length": window_length,
            "noise_level": noise_level,
        }
    )
    return case


def generate_a_regime(
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
        "A_regime",
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
    "A1": generate_a1,
    "A2": generate_a2,
    "A3": generate_a3,
    "A4": generate_a4,
    "A5": generate_a5,
    "A6": generate_a6,
    "A7": generate_a7,
    "A_regime": generate_a_regime,
}


def quick_grid(experiment: str) -> list[dict[str, Any]]:
    """Small grids intended for smoke tests and local iteration."""

    if experiment == "A1":
        return [{"mean_shift": s, "volatility_ratio": v, "d": 5, "n_per_segment": 120} for s, v in [(0.2, 1.25), (0.5, 1.5)]]
    if experiment == "A2":
        return [{"nu": nu, "n_per_segment": 120} for nu in [6, 20]]
    if experiment == "A3":
        return [
            {"delta": delta, "mode_separation": 3.0, "n_per_segment": 120, "demeaned": True}
            for delta in [0.05, 0.2]
        ]
    if experiment == "A4":
        return [{"delta_rho": dr, "rho1": 0.2, "d": 10, "n_per_segment": 120} for dr in [0.1, 0.3]]
    if experiment == "A5":
        return [{"epsilon": eps, "d": 20, "n_per_segment": 120} for eps in [0.1, 0.5]]
    if experiment == "A6":
        return [
            {"shock_length": 5, "magnitude": "large", "persistent": False},
            {"shock_length": 50, "magnitude": "medium", "persistent": True},
        ]
    if experiment == "A7":
        return [
            {
                "shift_family": fam,
                "signal_strength": strength,
                "n_per_segment": 120,
                "window_length": w,
                "noise_level": noise,
            }
            for fam, strength, w, noise in product(
                ["tail", "correlation"],
                ["medium", "strong"],
                [50, 100],
                ["low", "high"],
            )
        ]
    if experiment == "A_regime":
        return [{"regime_duration": 80, "d": 5, "similarity": "medium"}]
    raise KeyError(f"Unknown experiment: {experiment}")


def full_grid(experiment: str) -> list[dict[str, Any]]:
    """Broader grids following the difficulty knobs in docs/experiment_plan.md."""

    if experiment == "A1":
        return [
            {"mean_shift": s, "volatility_ratio": v, "n_per_segment": n, "d": d}
            for s, v, n, d in product(
                [0.05, 0.1, 0.2, 0.5, 1.0],
                [1.05, 1.1, 1.25, 1.5, 2.0],
                [50, 100, 250, 500],
                [1, 5, 20, 100],
            )
        ]
    if experiment == "A2":
        return [
            {"nu": nu, "n_per_segment": n, "garch_noise": garch, "n_changepoints": cps}
            for nu, n, garch, cps in product([4, 6, 8, 12, 20, 50], [100, 250], ["none", "mild"], [1, 3])
        ]
    if experiment == "A3":
        return [
            {
                "delta": delta,
                "mode_separation": sep,
                "n_per_segment": n,
                "serial_dependence": dep,
                "centered": centered,
            }
            for delta, sep, n, dep, centered in product(
                [0.02, 0.05, 0.1, 0.2, 0.3],
                [1, 2, 3, 5],
                [100, 250],
                ["IID", "AR(1)"],
                [False, True],
            )
        ]
    if experiment == "A4":
        return [
            {"delta_rho": dr, "rho1": rho, "d": d, "n_per_segment": n}
            for dr, rho, d, n in product([0.05, 0.1, 0.2, 0.4], [0.0, 0.2, 0.5], [5, 20, 50], [100, 250])
            if rho + dr < 0.95
        ]
    if experiment == "A5":
        return [
            {"epsilon": eps, "d": d, "n_factors": k, "sparsity": sparsity, "n_per_segment": n}
            for eps, d, k, sparsity, n in product([0.05, 0.1, 0.2, 0.5, 1.0], [10, 50, 100], [1, 3], ["dense", "sector-sparse"], [100, 250])
        ]
    if experiment == "A6":
        return [
            {"shock_length": m, "magnitude": mag, "shock_type": typ, "persistent": persistent}
            for m, mag, typ, persistent in product([1, 2, 5, 10, 20], ["small", "medium", "large"], ["mean", "volatility", "tail"], [False, True])
        ]
    if experiment == "A7":
        return [
            {
                "shift_family": fam,
                "signal_strength": strength,
                "n_per_segment": n,
                "window_length": w,
                "noise_level": noise,
            }
            for fam, strength, n, w, noise in product(
                ["tail", "mixture", "correlation", "factor"],
                ["weak", "medium", "strong"],
                [100, 250],
                [20, 50, 100, 250],
                ["low", "medium", "high"],
            )
        ]
    if experiment == "A_regime":
        return [
            {"regime_duration": duration, "d": 5, "similarity": similarity}
            for duration, similarity in product([50, 100, 250], ["easy", "medium", "hard"])
        ]
    raise KeyError(f"Unknown experiment: {experiment}")


def resolve_window(case: SyntheticCase, default_window: int) -> int:
    wl = case.params.get("window_length")
    if wl is not None and not (isinstance(wl, float) and np.isnan(wl)):
        return int(wl)
    return default_window


def generate_null_series(case: SyntheticCase, n_null: int, base_seed: int = 10_000) -> list[np.ndarray]:
    """Stationary (no-change) series matched to the case DGP parameters."""
    params = {k: v for k, v in case.params.items() if k != "seed"}
    exp = case.experiment
    series: list[np.ndarray] = []
    length = len(case.x)

    for i in range(n_null):
        seed = base_seed + i
        rng = np.random.default_rng(seed)
        if exp == "A1":
            d = int(params.get("d", 1))
            x = rng.normal(0.0, 1.0, size=(length, d))
        elif exp == "A2":
            nu = float(params.get("nu", 6))
            scale = np.sqrt((nu - 2.0) / nu)
            x = scale * rng.standard_t(df=nu, size=(length, 1))
        elif exp == "A3":
            a = float(params.get("mode_separation", 3.0))
            use_demeaned = bool(params.get("demeaned", params.get("centered", False)))
            if use_demeaned:
                z = rng.random(length) < 0.5
                x = np.where(z, -a, a)[:, None] + rng.normal(0.0, 1.0, size=(length, 1))
            else:
                x = sample_mixture(rng, length, 0.5, a, 1.0)[:, None]
        elif exp == "A4":
            d = int(params.get("d", 5))
            rho = float(params.get("rho1", 0.2))
            x = rng.multivariate_normal(np.zeros(d), equicorrelation(d, rho), size=length)
        elif exp == "A5":
            d = int(params.get("d", 10))
            x = rng.normal(0.0, 1.0, size=(length, d))
        elif exp == "A6":
            x = rng.normal(0.0, 1.0, size=(length, 1))
        elif exp == "A7":
            null_case = generate_a2(seed=seed, n_per_segment=length, nu=20)
            x = null_case.x
        elif exp == "A_regime":
            d = int(params.get("d", 5))
            x = rng.normal(0.0, 1.0, size=(length, d))
        else:
            x = rng.normal(0.0, 1.0, size=case.x.shape)
        series.append(np.asarray(x, dtype=float))
    return series


def baseline_kwargs(key: str, case: SyntheticCase, window: int, n_bkps: int | None) -> dict[str, Any]:
    """Route only compatible options to each detector family."""

    window = resolve_window(case, window)
    if key in {"pelt_l2", "pelt_rbf", "pelt_normal", "binseg", "bottomup"}:
        return {"penalty": 8.0, "n_bkps": n_bkps, "min_size": max(5, window // 2)}
    if key in {"cusum_mean", "cusum_vol"}:
        return {"min_distance": window, "burn_in": window}
    if key == "ewma_vol":
        return {"min_distance": window}
    if key == "bocpd_gaussian":
        return {"hazard": 1.0 / max(window * 4, 1), "min_distance": window, "burn_in": window}
    if key == "gaussian_hmm":
        return {"n_states": min(4, max(2, len(case.changepoints) + 1)), "min_distance": window}
    if key.startswith("proposed"):
        metric = "bures" if case.experiment in {"A4", "A5"} else "sliced_wasserstein"
        if case.regime_labels is None:
            n_proto = 3
        else:
            n_proto = len(np.unique(case.regime_labels))
        return {
            "window": window,
            "min_persistence": 2,
            "min_distance": window,
            "metric": metric,
            "horizon": max(2 * window, 80),
            "n_prototypes": min(4, max(2, n_proto)),
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
    calibration: CalibratedThresholds | None = None,
) -> list[ExperimentResult]:
    if baselines is None:
        baselines = BASELINE_SETS[case.experiment]
    window = resolve_window(case, window or max(20, min(80, len(case.x) // 8)))
    n_bkps = len(case.changepoints) if n_bkps is None else n_bkps
    tolerance = detection_tolerance(window)
    results = []
    for key in baselines:
        kwargs = baseline_kwargs(key, case, window=window, n_bkps=n_bkps)
        if calibration is not None:
            cfg = calibration_config_key(case, window)
            th = calibration.get(cfg, key)
            if th is not None and np.isfinite(th):
                kwargs["threshold"] = th
                if key.startswith("proposed"):
                    kwargs["alert_threshold"] = th
        result = run_baseline(key, case.x, **kwargs)
        metrics = with_localization_alias(
            detection_metrics(case.changepoints, result.changepoints, tolerance=tolerance)
        )
        if not result.metadata.get("unavailable"):
            metrics.update(score_diagnostics(result))
        if case.experiment == "A7":
            metrics.update(s6_metrics(case.changepoints, result.changepoints, window, metrics))
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
    if case.experiment == "A_regime" and case.regime_labels is not None:
        k_true = len(np.unique(case.regime_labels))
        centers, labels = cluster_rolling_windows(case.x, window=window, n_clusters=k_true)
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
                    "source": "docs/experiment_plan.md A_regime baseline",
                    "url": "docs/experiment_plan.md",
                    "implementation": "KMeans on rolling mean/std/covariance features.",
                    "notes": "",
                },
            )
        )
        p_centers, p_labels, _, p_entropy = regime_labels_from_prototypes(
            case.x, window=window, n_prototypes=k_true
        )
        p_true = case.regime_labels[p_centers - 1]
        p_metrics = clustering_metrics(p_true, p_labels)
        p_metrics["mean_posterior_entropy"] = float(np.mean(p_entropy))
        for r in results:
            if r.method in {PROPOSED_PRIMARY, "proposed_local_proto_no_global"}:
                r.metrics.update(
                    {
                        "ari": p_metrics["ari"],
                        "nmi": p_metrics["nmi"],
                        "label_accuracy": p_metrics["label_accuracy"],
                        "mean_posterior_entropy": p_metrics["mean_posterior_entropy"],
                    }
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
    calibrate: bool = True,
    null_seeds: int = 15,
    false_alarm_quantile: float = 0.95,
) -> list[ExperimentResult]:
    experiments = experiments or list(BASELINE_SETS)
    all_results: list[ExperimentResult] = []
    for experiment in experiments:
        methods = baselines or BASELINE_SETS[experiment]
        cases = make_cases(experiment, grid=grid, seeds=seeds)
        calibration_cache: dict[tuple[Any, ...], CalibratedThresholds] = {}
        for case in cases:
            calibration: CalibratedThresholds | None = None
            if calibrate:
                w = resolve_window(case, window or max(20, min(80, len(case.x) // 8)))
                cfg = calibration_config_key(case, w)
                if cfg not in calibration_cache:
                    null_series = generate_null_series(case, n_null=null_seeds)
                    kwargs_map = {
                        m: baseline_kwargs(m, case, window=w, n_bkps=len(case.changepoints))
                        for m in methods
                    }
                    calibration_cache[cfg] = calibrate_for_case(
                        case, methods, null_series, kwargs_map, false_alarm_quantile=false_alarm_quantile
                    )
                calibration = calibration_cache[cfg]
            all_results.extend(
                run_case(case, baselines=methods, window=window, calibration=calibration)
            )
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
