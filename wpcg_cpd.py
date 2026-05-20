"""
Implementation of Equation (13) from the project proposal:

    max_tau  J(tau) = sum_{i=1}^{S-1}  W_2^2( mu_i(tau), mu_{i+1}(tau) )

where mu_i(tau) is the empirical distribution of segment i induced by the
change-point vector tau.  We optimize J(tau) directly via the coordinate-wise
sweep on tau (the discrete optimization step described in Sec. 4.1 / Eq. 19):

    tau_i  <-  argmax_{tau in (tau_{i-1}, tau_{i+1})}  W_2^2( mu_i(tau), mu_{i+1}(tau) )

The 2-Wasserstein distance between two 1-D empirical distributions is computed
via the closed-form quantile formula  W_2^2(p,q) = int_0^1 (F^{-1}-G^{-1})^2 du.
"""

import numpy as np


# -----------------------------------------------------------------------------
# Core: squared 2-Wasserstein distance between two 1-D empirical distributions
# -----------------------------------------------------------------------------
def w2_squared_1d(x, y):
    """
    Closed-form squared 2-Wasserstein distance between two 1-D empirical
    distributions with (possibly different) sample sizes.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    nx, ny = len(x), len(y)
    if nx == 0 or ny == 0:
        return 0.0
    xs = np.sort(x)
    ys = np.sort(y)
    if nx == ny:
        return float(np.mean((xs - ys) ** 2))
    # General case: integrate (F^{-1}(u) - G^{-1}(u))^2 on a fine grid of u
    n_q = 4 * max(nx, ny)
    u = (np.arange(n_q) + 0.5) / n_q
    fx = np.interp(u, (np.arange(nx) + 0.5) / nx, xs)
    fy = np.interp(u, (np.arange(ny) + 0.5) / ny, ys)
    return float(np.mean((fx - fy) ** 2))


# -----------------------------------------------------------------------------
# Equation (13): the global objective
# -----------------------------------------------------------------------------
def objective_J(X, tau):
    """
    J(tau) = sum_{i=1}^{S-1} W_2^2( mu_i, mu_{i+1} )
    tau is the list of internal change points (length S-1), 0 < tau_1 < ... < tau_{S-1} < T.
    """
    T = len(X)
    boundaries = [0] + list(tau) + [T]
    total = 0.0
    for i in range(len(boundaries) - 2):
        a, b, c = boundaries[i], boundaries[i + 1], boundaries[i + 2]
        total += w2_squared_1d(X[a:b], X[b:c])
    return total


# -----------------------------------------------------------------------------
# Coordinate-wise sweep optimizer for Eq. (13)
# -----------------------------------------------------------------------------
def coordinate_sweep_optimize(X, tau_init, min_seg_len=10, max_iter=50, verbose=True):
    """
    Cyclic coordinate ascent on J(tau).  For each i, hold the other taus fixed
    and solve
        tau_i <- argmax_{tau in (tau_{i-1}+m, tau_{i+1}-m)} W_2^2(mu_i, mu_{i+1}),
    where m = min_seg_len guards against degenerate empty / tiny segments.
    Since changing tau_i only affects segments i and i+1, this is exactly
    the discrete coordinate update advertised in the paper.
    """
    T = len(X)
    tau = list(tau_init)
    S_minus_1 = len(tau)
    history = [list(tau)]

    for it in range(max_iter):
        prev = list(tau)
        for i in range(S_minus_1):
            left_bd  = tau[i - 1] if i > 0 else 0
            right_bd = tau[i + 1] if i < S_minus_1 - 1 else T
            lo = left_bd + min_seg_len
            hi = right_bd - min_seg_len
            if hi <= lo:
                continue
            best_val = -np.inf
            best_t = tau[i]
            # exhaustive 1-D search over the allowed interval
            for t in range(lo, hi + 1):
                val = w2_squared_1d(X[left_bd:t], X[t:right_bd])
                if val > best_val:
                    best_val = val
                    best_t = t
            tau[i] = best_t

        history.append(list(tau))
        if verbose:
            print(f"  iter {it+1:2d}:  tau = {tau}   J = {objective_J(X, tau):.4f}")
        if tau == prev:
            if verbose:
                print(f"  converged at iter {it+1}")
            break

    return tau, history


# -----------------------------------------------------------------------------
# Build the synthetic time series specified by the user
# -----------------------------------------------------------------------------
def build_series(seed=42):
    rng = np.random.default_rng(seed)
    s1 = rng.normal(0.0, 1.0, 150)              # N(0,1)            len 150
    s2 = rng.normal(5.0, 1.0, 100)              # N(5,1)            len 100
    s3 = rng.poisson(3.0, 200).astype(float)    # Poisson(3)        len 200
    s4 = rng.binomial(10, 0.5, 120).astype(float)  # Binomial(10,.5) len 120
    s5 = rng.exponential(2.0, 180)              # Exponential(2)    len 180
    return np.concatenate([s1, s2, s3, s4, s5])


# -----------------------------------------------------------------------------
# Run the experiment
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    X = build_series(seed=42)
    T = len(X)
    true_tau = [150, 250, 450, 570]
    init_tau = [125, 300, 400, 600]

    print("=" * 70)
    print("Setup")
    print("=" * 70)
    print(f"Total length T              = {T}")
    print(f"True change points          = {true_tau}")
    print(f"J(tau) at true change points = {objective_J(X, true_tau):.4f}")
    print()
    print(f"Initial change points       = {init_tau}")
    print(f"J(tau) at initial points    = {objective_J(X, init_tau):.4f}")
    print()

    print("=" * 70)
    print("Coordinate-wise sweep optimization of Eq. (13)")
    print("=" * 70)
    final_tau, history = coordinate_sweep_optimize(
        X, init_tau, min_seg_len=10, max_iter=20, verbose=True
    )

    print()
    print("=" * 70)
    print("Results")
    print("=" * 70)
    print(f"Initial   change points : {init_tau}")
    print(f"Final     change points : {final_tau}")
    print(f"True      change points : {true_tau}")
    errors = [f - t for f, t in zip(final_tau, true_tau)]
    abs_errors = [abs(e) for e in errors]
    print(f"Signed    errors        : {errors}")
    print(f"Absolute  errors        : {abs_errors}")
    print(f"Max abs error           : {max(abs_errors)}")
    print(f"Mean abs error          : {np.mean(abs_errors):.2f}")
    print()
    print(f"J(final) = {objective_J(X, final_tau):.4f}")
    print(f"J(true)  = {objective_J(X, true_tau):.4f}")
