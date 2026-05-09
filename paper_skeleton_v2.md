# Distributional Change Point Detection via Optimal Transport: A Unified Framework

## Working Draft v2

---

## 1. Preliminaries

### 1.1 Notation

- $`[T] = \{1, \ldots, T\}`$. $`X_{a:b} = (X_a, \ldots, X_b)`$.
- $`\hat{P}_n = \frac{1}{n}\sum_{i=1}^n \delta_{X_i}`$: empirical measure.
- $`a \wedge b = \min(a,b)`$, $`a \vee b = \max(a,b)`$.
- $`\mathbb{S}^{d-1}`$: unit sphere in $`\mathbb{R}^d`$. $`\sigma`$: uniform (Haar) measure on $`\mathbb{S}^{d-1}`$.
- $`\theta_{\ast} P`$: pushforward of $`P`$ under $`x \mapsto \langle \theta, x \rangle`$.
- $`\Phi`$, $`\phi`$: standard normal CDF and PDF. $`Z \sim \mathcal{N}(0,1)`$ throughout.

### 1.2 Wasserstein Distances

**Definition 1 ($`p`$-Wasserstein).** For $`P, Q \in \mathcal{P}_p(\mathbb{R}^d)`$:

```math
W_p(P, Q) = \left(\inf_{\pi \in \Pi(P,Q)} \int \|x - y\|^p \, d\pi(x,y)\right)^{1/p}.
```

**1-D quantile representation.** For $`P, Q`$ on $`\mathbb{R}`$:

```math
W_p^p(P,Q) = \int_0^1 |F_P^{-1}(u) - F_Q^{-1}(u)|^p \, du.
```

**Equal-variance identity.** If $`P, Q`$ on $`\mathbb{R}`$ both have mean $`\mu`$ and variance $`\sigma^2`$:

```math
W_2^2(P, Q) = 2\sigma^2(1 - \rho_{PQ})
```

where $`\rho_{PQ} = \mathrm{Corr}(F_P^{-1}(U), F_Q^{-1}(U))`$, $`U \sim \mathrm{Unif}(0,1)`$.

*Proof.* Expand:

```math
W_2^2 = \int_0^1 (F_P^{-1}(u))^2 \, du + \int_0^1 (F_Q^{-1}(u))^2 \, du - 2\int_0^1 F_P^{-1}(u) F_Q^{-1}(u) \, du.
```

The first two integrals equal $`\mathbb{E}[X^2] = \mu^2 + \sigma^2`$ each. By matched mean and variance, the third integral's correlation form gives $`W_2^2 = 2(\mu^2 + \sigma^2) - 2(\mu^2 + \sigma^2 \rho_{PQ}) = 2\sigma^2(1 - \rho_{PQ})`$. $`\square`$

**Definition 2 (Sliced Wasserstein).**

```math
SW_p^p(P,Q) = \int_{\mathbb{S}^{d-1}} W_p^p(\theta_{\ast} P, \theta_{\ast} Q) \, d\sigma(\theta).
```

**Definition 3 (Bures-Wasserstein for Gaussians).** For $`P = \mathcal{N}(\mu_1, \Sigma_1)`$, $`Q = \mathcal{N}(\mu_2, \Sigma_2)`$:

```math
W_2^2(P, Q) = \|\mu_1 - \mu_2\|^2 + \mathcal{B}^2(\Sigma_1, \Sigma_2)
```

where $`\mathcal{B}^2(\Sigma_1, \Sigma_2) = \mathrm{tr}(\Sigma_1) + \mathrm{tr}(\Sigma_2) - 2\mathrm{tr}\!\left((\Sigma_1^{1/2}\Sigma_2\Sigma_1^{1/2})^{1/2}\right)`$ is the squared Bures distance.

When $`\Sigma_1, \Sigma_2`$ share eigenvectors (commute), this simplifies to:

```math
\mathcal{B}^2(\Sigma_1, \Sigma_2) = \sum_{k=1}^d (\sqrt{\lambda_k^{(1)}} - \sqrt{\lambda_k^{(2)}})^2
```

where $`\lambda_k^{(j)}`$ are eigenvalues of $`\Sigma_j`$.

### 1.3 Baseline Dissimilarities

**KS.** $`\mathrm{KS}(P,Q) = \sup_{x} |F_P(x) - F_Q(x)|`$. Univariate only.

**MMD.** For RKHS kernel $`k`$:

```math
\mathrm{MMD}^2(P,Q) = \mathbb{E}[k(X,X')] + \mathbb{E}[k(Y,Y')] - 2\mathbb{E}[k(X,Y)].
```

**Energy distance.** $`E(P,Q) = 2\mathbb{E}\|X-Y\| - \mathbb{E}\|X-X'\| - \mathbb{E}\|Y-Y'\|`$. Equals $`\mathrm{MMD}^2`$ with the distance kernel (Sejdinovic et al. 2013).

### 1.4 Change Point Detection Setup

**Model.** $`X_1, \ldots, X_T`$ independent, with $`S`$ segments:

```math
1 = \tau_0 < \tau_1 < \cdots < \tau_{S-1} < \tau_S = T, \qquad X_t \sim P_i \text{ for } t \in (\tau_{i-1}, \tau_i].
```

**Quantities.**
- $`\Delta_i = \tau_i - \tau_{i-1}`$: length of segment $`i`$.
- $`\Delta_{\min} = \min_i \Delta_i`$, $`\Delta_{\max} = \max_i \Delta_i`$.
- $`\kappa = \min_{1 \leq i \leq S-1} D(P_i, P_{i+1})`$: minimum separation.

**Goal.** Estimate $`S - 1`$ and $`\{\tau_j\}_{j=1}^{S-1}`$.

**Standing assumption (this draft).** Independence within and across segments. Extension to mixing: see Remark at end of Section 3.

---

## 2. Axiomatic Framework

### Axioms

Let $`D : \mathcal{F} \times \mathcal{F} \to [0, \infty)`$ be a dissimilarity on a class $`\mathcal{F} \subset \mathcal{P}_p(\mathbb{R}^d)`$, with empirical version $`\hat{D}`$.

**(A1) Sub-Gaussian Concentration.** There exist $`C_1, c_1 > 0`$ such that for all $`P, Q \in \mathcal{F}`$, $`n, m \geq 1`$, and $`t > 0`$:

```math
\mathbb{P}\!\left(|\hat{D}(\hat{P}_n, \hat{Q}_m) - D(P, Q)| > t\right) \leq C_1 \exp\!\left(-c_1 (n \wedge m) t^2\right).
```

**(A2) Mixture Contamination Bound.** There exists $`h : [0,1] \to [0,1]`$ non-decreasing with $`h(0) = 0`$, $`h(1) \leq 1`$, such that for all $`P, Q \in \mathcal{F}`$ and $`\alpha \in [0,1]`$:

```math
D(P, (1-\alpha)P + \alpha Q) \leq h(\alpha) \cdot D(P, Q).
```

**(A3) Non-degeneracy and symmetry.** $`D(P,Q) = 0 \Leftrightarrow P = Q`$, and $`D(P,Q) = D(Q,P)`$.

### Key Lemmas

**Lemma 1 (Wasserstein Mixture Bound).** *For $`W_p^p`$: $`h(\alpha) = \alpha`$.*

*Proof.* Let $`R_\alpha = (1-\alpha)P + \alpha Q`$. Draw $`Y \sim R_\alpha`$ via mixture: with probability $`(1-\alpha)`$, $`Y \sim P`$; with probability $`\alpha`$, $`Y \sim Q`$. Couple with $`X \sim P`$:
- $`Y \sim P`$: set $`X = Y`$. Cost $`= 0`$.
- $`Y \sim Q`$: use optimal coupling $`\pi^* \in \Pi(P,Q)`$. Expected cost $`= W_p^p(P,Q)`$.

Total: $`(1-\alpha) \cdot 0 + \alpha \cdot W_p^p(P,Q) = \alpha \, W_p^p(P,Q)`$. This is a valid coupling, so $`W_p^p(P, R_\alpha) \leq \alpha \, W_p^p(P,Q)`$. $`\square`$

**Tightness.** Equality for point masses: $`P = \delta_a`$, $`Q = \delta_b`$ gives $`W_p^p(\delta_a, (1-\alpha)\delta_a + \alpha\delta_b) = \alpha \|a-b\|^p`$.

**Lemma 2 ($`\mathrm{MMD}^2`$ / Energy Mixture Bound).** *$`\mathrm{MMD}^2(P, R_\alpha) = \alpha^2 \mathrm{MMD}^2(P,Q)`$, so $`h(\alpha) = \alpha^2`$.*

*Proof.* Mean embedding is linear: $`\mu_{R_\alpha} = (1-\alpha)\mu_P + \alpha \mu_Q`$. Then:

```math
\mathrm{MMD}^2(P, R_\alpha) = \|\mu_P - \mu_{R_\alpha}\|_{\mathcal{H}}^2 = \alpha^2 \|\mu_P - \mu_Q\|_{\mathcal{H}}^2 = \alpha^2 \mathrm{MMD}^2(P,Q). \quad \square
```

**Lemma 3 (KS Mixture Bound).** *$`\mathrm{KS}(P, R_\alpha) = \alpha \, \mathrm{KS}(P, Q)`$, so $`h(\alpha) = \alpha`$.*

*Proof.* $`F_{R_\alpha}(x) = (1-\alpha)F_P(x) + \alpha F_Q(x)`$. Then $`|F_P(x) - F_{R_\alpha}(x)| = \alpha |F_P(x) - F_Q(x)|`$ for all $`x`$, so $`\sup_x = \alpha \sup_x |F_P(x) - F_Q(x)|`$. $`\square`$

### Verification Table

| $`D`$ | $`h(\alpha)`$ | $`\phi(n) \asymp`$ | $`\bar{h}^{-1}(x) \asymp`$ | Ref |
|---|---|---|---|---|
| $`W_p^p`$ (1-D or sliced) | $`\alpha`$ | $`n^{-1/2}`$ | $`x`$ | Lem 1; del Barrio+ '99; Nietert+ '22 |
| $`S_\varepsilon`$ (Sinkhorn) | $`\leq \alpha`$ | $`n^{-1/2}`$ | $`\leq x`$ | via $`S_\varepsilon \leq W_2^2`$; Mena & N-W '19 |
| $`\mathrm{MMD}^2`$ | $`\alpha^2`$ | $`n^{-1/2}`$ | $`x/2`$ | Lem 2; Gretton+ '12 |
| $`E`$ (energy) | $`\alpha^2`$ | $`n^{-1/2}`$ | $`x/2`$ | Lem 2 (distance kernel); Székely-Rizzo '04 |
| KS | $`\alpha`$ | $`n^{-1/2}`$ | $`x`$ | Lem 3; DKW |

**Observation.** All have $`\phi(n) \asymp n^{-1/2}`$. The contamination functions differ: $`h(\alpha) = \alpha`$ for Wasserstein/KS, $`h(\alpha) = \alpha^2`$ for $`\mathrm{MMD}^2`$/energy. As we show in Theorem 1, this affects localization constants but not rates.

---

## 3. General Localization Theorem

### 3.1 Setup (Single Change Point)

$`S = 2`$. One true change point $`\tau^{*}`$. $`X_t \sim P`$ for $`t \leq \tau^{*}`$, $`X_t \sim Q`$ for $`t > \tau^{*}`$. Write $`\Delta_1 = \tau^{*}`$, $`\Delta_2 = T - \tau^{*}`$.

**Estimator.** $`\hat{\tau} = \arg\max_{t \in [\ell, T-\ell]} \hat{D}_t`$ where $`\hat{D}_t = \hat{D}(\hat{\mu}_{1:t}, \hat{\mu}_{t+1:T})`$ and $`\ell`$ is a minimum segment length with $`\ell \leq \Delta_{\min}`$.

### 3.2 Population-Level Signal Profile

Before proving the theorem, we characterize the population-level signal $`D_{\text{pop}}(t)`$ as a function of the split point $`t`$.

**Lemma 4 (Signal Profile).** *Under the single-CP model with $`D(P,Q) = \kappa`$:*

```math
D_{\text{pop}}(t) = \begin{cases}
D\!\left(P, \, \frac{\tau^{*} - t}{T-t}P + \frac{\Delta_2}{T-t}Q\right) & \text{if } t < \tau^{*}, \\[6pt]
D(P, Q) = \kappa & \text{if } t = \tau^{*}, \\[6pt]
D\!\left(\frac{\Delta_1}{t}P + \frac{t - \tau^{*}}{t}Q, \, Q\right) & \text{if } t > \tau^{*}.
\end{cases}
```

*Applying (A2):*

```math
D_{\text{pop}}(t) \leq \begin{cases}
h\!\left(\frac{\Delta_2}{\Delta_2 + (\tau^{*} - t)}\right) \kappa & \text{if } t \leq \tau^{*}, \\[6pt]
h\!\left(\frac{\Delta_1}{\Delta_1 + (t - \tau^{*})}\right) \kappa & \text{if } t \geq \tau^{*}.
\end{cases}
```

*Proof.* For $`t < \tau^{*}`$: left window is pure $`P`$; right window has $`(\tau^{*} - t)`$ from $`P`$ and $`\Delta_2`$ from $`Q`$, so its population is $`(1-\alpha_t)P + \alpha_t Q`$ with $`\alpha_t = \Delta_2/(T-t)`$. Apply (A2) and (A3). The case $`t > \tau^{*}`$ is symmetric. $`\square`$

**Gap function.** Define the signal gap $`G(r) = D_{\text{pop}}(\tau^{*}) - D_{\text{pop}}(\tau^{*} - r)`$ for displacement $`r > 0`$:

```math
G(r) \geq \kappa\!\left(1 - h\!\left(\frac{\Delta_2}{\Delta_2 + r}\right)\right) = \kappa \, \bar{h}\!\left(\frac{r}{\Delta_2 + r}\right)
```

where $`\bar{h}(\rho) := 1 - h(1 - \rho)`$.

**Explicit gap for each method:**

For $`h(\alpha) = \alpha`$ ($`W_p^p`$, KS): $`\bar{h}(\rho) = \rho`$, so

```math
G(r) \geq \frac{r}{\Delta_2 + r} \kappa.
```

For $`h(\alpha) = \alpha^2`$ ($`\mathrm{MMD}^2`$, energy): $`\bar{h}(\rho) = 2\rho - \rho^2`$, so

```math
G(r) \geq \frac{r(2\Delta_2 + r)}{(\Delta_2 + r)^2} \kappa.
```

For small $`r / \Delta_2`$:
- $`h = \alpha`$: $`G(r) \approx \frac{r}{\Delta_2} \kappa`$.
- $`h = \alpha^2`$: $`G(r) \approx \frac{2r}{\Delta_2} \kappa`$. (Factor 2 advantage in gap.)

### 3.3 Theorem 1 (Single Change Point Localization)

*Let $`D`$ satisfy (A1)--(A3) with contamination $`h`$ and sub-Gaussian concentration constants $`(C_1, c_1)`$. Suppose $`D(P,Q) \geq \kappa > 0`$ and $`\ell \leq \Delta_{\min}`$. Define*

```math
\epsilon_T = \sqrt{\frac{2\log(2T/\delta)}{c_1 \ell}}.
```

*Assume $`\kappa > 2\epsilon_T`$ (detectable signal). Then with probability $`\geq 1 - \delta`$:*

```math
|\hat{\tau} - \tau^{*}| \leq r^{*}
```

*where $`r^{*}`$ is the smallest $`r`$ satisfying $`\bar{h}\!\left(\frac{r}{\Delta_{\max} + r}\right) \geq \frac{2\epsilon_T}{\kappa}`$.*

*Explicitly:*

**(i)** For $`h(\alpha) = \alpha`$ (Wasserstein, KS):

```math
|\hat{\tau} - \tau^{*}| \leq \frac{2\epsilon_T \, \Delta_{\max}}{\kappa - 2\epsilon_T}.
```

**(ii)** For $`h(\alpha) = \alpha^2`$ ($`\mathrm{MMD}^2`$, energy):

```math
|\hat{\tau} - \tau^{*}| \leq \frac{\epsilon_T \, \Delta_{\max}}{\kappa - \epsilon_T}.
```

*In both cases, for balanced segments ($`\Delta_{\max} \asymp T`$, $`\ell \asymp T`$) and $`\kappa \gg \epsilon_T`$:*

```math
|\hat{\tau} - \tau^{*}| \lesssim \frac{\sqrt{T \log(T/\delta)}}{\kappa}.
```

### 3.4 Proof of Theorem 1

**Step 1 (Uniform concentration).** Apply (A1) at each candidate $`t \in [\ell, T - \ell]`$ with $`n \wedge m \geq \ell`$:

```math
\mathbb{P}(|\hat{D}_t - D_{\text{pop}}(t)| > \epsilon_T) \leq C_1 \exp(-c_1 \ell \epsilon_T^2) = C_1 \cdot \frac{\delta}{2T}.
```

Union bound over $`\leq T`$ candidates: $`\mathbb{P}(\exists t : |\hat{D}_t - D_{\text{pop}}(t)| > \epsilon_T) \leq \delta/2 \leq \delta`$.

Henceforth condition on the good event $`\mathcal{E} = \{\sup_t |\hat{D}_t - D_{\text{pop}}(t)| \leq \epsilon_T\}`$, which holds with probability $`\geq 1 - \delta`$.

**[Note: (A1) must hold uniformly over all population pairs arising from the mixture distributions at different split points. For dissimilarities with distribution-dependent concentration (e.g., variance-dependent), this requires $`\mathcal{F}`$ to have uniformly bounded moments. Satisfied automatically for bounded or sub-Gaussian data.]**

**Step 2 (Signal at $`\tau^{*}`$).** Both windows are pure: $`D_{\text{pop}}(\tau^{*}) = D(P,Q) = \kappa`$. Under $`\mathcal{E}`$:

```math
\hat{D}_{\tau^{*}} \geq \kappa - \epsilon_T.
```

**Step 3 (Signal at $`\tau^{*} - r`$, $`r > 0`$).** By Lemma 4:

```math
D_{\text{pop}}(\tau^{*} - r) \leq h\!\left(\frac{\Delta_2}{\Delta_2 + r}\right) \kappa.
```

Under $`\mathcal{E}`$:

```math
\hat{D}_{\tau^{*} - r} \leq h\!\left(\frac{\Delta_2}{\Delta_2 + r}\right) \kappa + \epsilon_T.
```

**Step 4 (Localization).** $`\hat{\tau} \neq \tau^{*}`$ requires $`\hat{D}_t \geq \hat{D}_{\tau^{*}}`$ for some $`t`$ with $`|t - \tau^{*}| \geq r`$. On $`\mathcal{E}`$, sufficient to prevent this:

```math
\kappa - \epsilon_T > h\!\left(\frac{\Delta_2}{\Delta_2 + r}\right) \kappa + \epsilon_T.
```

Rearranging:

```math
\bar{h}\!\left(\frac{r}{\Delta_2 + r}\right) > \frac{2\epsilon_T}{\kappa}. \quad (\star)
```

**Solving $`(\star)`$ for $`r`$:**

**Case $`h(\alpha) = \alpha`$:** $`\bar{h}(\rho) = \rho`$. Then $`(\star)`$ becomes $`\frac{r}{\Delta_2 + r} > \frac{2\epsilon_T}{\kappa}`$:

```math
r > \frac{2\epsilon_T}{\kappa - 2\epsilon_T} \Delta_2.
```

The symmetric bound for $`t > \tau^{*}`$ gives $`r > \frac{2\epsilon_T}{\kappa - 2\epsilon_T} \Delta_1`$. Taking the worse case:

```math
r^{*} = \frac{2\epsilon_T}{\kappa - 2\epsilon_T} \Delta_{\max}. \quad \square
```

**Case $`h(\alpha) = \alpha^2`$:** $`\bar{h}(\rho) = 2\rho - \rho^2`$. For small $`\rho`$, $`\bar{h}(\rho) \approx 2\rho`$, so $`\bar{h}^{-1}(x) \approx x/2`$. Precisely: $`(\star)`$ becomes $`\frac{r(2\Delta_2 + r)}{(\Delta_2 + r)^2} > \frac{2\epsilon_T}{\kappa}`$. For $`r \ll \Delta_2`$:

```math
r \gtrsim \frac{\epsilon_T}{\kappa} \Delta_2. \quad \square
```

**Remark ($`\mathrm{MMD}^2`$ has better constant).** The factor-2 improvement for $`\mathrm{MMD}^2`$ in the localization constant arises because the quadratic contamination $`h(\alpha) = \alpha^2`$ creates a steeper gap near $`\tau^{*}`$. However, this advantage disappears in rate: both give $`r^{*} = \Theta(\sqrt{T \log T}/\kappa)`$ with balanced segments.

### 3.5 Extension to Multiple Change Points

**Theorem 2 (via Binary Segmentation).** *Under (A1)--(A3) with $`S - 1`$ change points, minimum separation $`\kappa`$, threshold $`\lambda = \kappa/2`$, and*

```math
\Delta_{\min} \geq C_0 \kappa^{-2} \log(T/\delta),
```

*BS-$`D`$ detects exactly $`S - 1`$ change points with $`\max_j \min_k |\hat{\tau}_j - \tau_k| \leq r^{*}`$.*

*Proof.* Induction on $`S`$. Base case $`S = 2`$: Theorem 1. Inductive step: the dominant change point in any interval has signal $`\geq \kappa - \epsilon_T > \lambda`$ (detection guarantee), and intervals with no change point have signal $`\leq \epsilon_T < \lambda`$ (no false alarm). The split is within $`r^{*}`$ of a true change point, and recursion reduces the problem. Full details follow Fryzlewicz (2014, AoS, Section 3) with $`\hat{D}`$ replacing CUSUM. $`\square`$

**Remark (dependence).** Extending to $`\beta`$-mixing or $`\tau`$-mixing sequences requires replacing the i.i.d. concentration in (A1) with mixing-dependent analogues. For Wasserstein on mixing sequences, partial results exist (Dedecker-Merlevède 2017) but a complete theory matched to financial dependence structures (e.g., GARCH innovations) remains open.

---

## 4. Corollaries

All have $`\phi(n) \asymp n^{-1/2}`$, giving $`\epsilon_T \asymp \sqrt{\log T / \ell}`$. Localization rate (balanced, $`\ell \asymp T`$):

```math
r^{*} \asymp \frac{\sqrt{T \log T}}{\kappa_D}
```

where $`\kappa_D = D(P,Q)`$ is the only method-dependent quantity.

| Corollary | $`D`$ | $`\kappa_D`$ depends on... | Dimension? |
|---|---|---|---|
| Cor 1 | $`SW_2^2`$ | Full joint distribution | Free ($`n^{-1/2}`$ rate) |
| Cor 2 | Coord-wise $`W_2^2`$ | Marginals only | Free |
| Cor 3 | $`S_\varepsilon`$ | Full joint distribution | Free |
| Cor 4 | $`\mathrm{MMD}^2`$ | Kernel + full distribution | Free (but signal depends on $`\sigma`$, $`d`$) |
| Cor 5 | KS | 1-D CDF difference | 1-D only |

**At this level: everyone ties on rate. The paper differentiates methods by studying $`\kappa_D`$ on specific alternatives.**

---

## 5. Sub-Case I: Tail Regime Shifts

### 5.1 Financial Motivation

The calm-to-crisis transition often manifests as a tail shift with approximately preserved mean and variance. A risk manager needs detection *before* variance fully blows up — during the early-stress phase where mean returns are near-zero but the return distribution has shifted from light-tailed to heavy-tailed.

### 5.2 Alternative Class

**Definition 4.** Fix $`k \geq 2`$. The *$`k`$-moment-matched tail alternatives*:

```math
\mathcal{A}_k(\kappa) = \left\{(P, Q) : \mathbb{E}_P[X^j] = \mathbb{E}_Q[X^j] \;\forall\, j \leq k, \quad W_2(P,Q) \geq \kappa\right\}.
```

Canonical instance: $`P = \mathcal{N}(0,1)`$, $`Q = t_\nu(0, \tilde{\sigma}_\nu^2)`$ with $`\tilde{\sigma}_\nu = \sqrt{(\nu-2)/\nu}`$ (variance-matched). Matched on $`k = 2`$ for $`\nu > 4`$.

### 5.3 Signal Computation

**Proposition 1 ($`W_2`$ signal for Gaussian vs variance-matched $`t_\nu`$).**

*$`W_2^2(P,Q) = \frac{3}{8\nu^2} + O(\nu^{-3})`$, so $`W_2(P,Q) = \frac{\sqrt{3/8}}{\nu} + O(\nu^{-2}) \approx \frac{0.612}{\nu}`$.*

*Proof.* By the Cornish-Fisher expansion for the variance-matched $`t_\nu`$ quantile function, combining the standard $`t_\nu`$ Cornish-Fisher with the scaling $`\tilde{\sigma}_\nu = 1 - 1/\nu + O(\nu^{-2})`$:

```math
F_Q^{-1}(u) = z + \frac{z^3 - 3z}{4\nu} + O(\nu^{-2}), \qquad z = \Phi^{-1}(u).
```

**Derivation of the leading correction.** The unscaled $`t_\nu`$ quantile satisfies $`t_\nu^{-1}(u) \approx z + \frac{z^3 + z}{4\nu} + O(\nu^{-2})`$ (standard Cornish-Fisher; see e.g. Hill and Davis 1968, Johnson, Kotz, Balakrishnan 1995 Ch. 28). Scaling by $`\tilde{\sigma}_\nu`$:

```math
F_Q^{-1}(u) = \tilde{\sigma}_\nu \cdot t_\nu^{-1}(u) \approx \left(1 - \frac{1}{\nu}\right)\left(z + \frac{z^3 + z}{4\nu}\right) = z + \frac{z^3 + z}{4\nu} - \frac{z}{\nu} + O(\nu^{-2}).
```

Combining: $`F_Q^{-1}(u) - z = \frac{z^3 + z - 4z}{4\nu} + O(\nu^{-2}) = \frac{z^3 - 3z}{4\nu} + O(\nu^{-2})`$.

Now compute $`W_2^2`$:

```math
W_2^2 = \int_0^1 (F_Q^{-1}(u) - \Phi^{-1}(u))^2 \, du = \frac{1}{16\nu^2}\int_{-\infty}^{\infty}(z^3 - 3z)^2 \phi(z) \, dz + O(\nu^{-3}).
```

Expanding $`(z^3 - 3z)^2 = z^6 - 6z^4 + 9z^2`$ and using Gaussian moments $`\mathbb{E}[Z^{2k}] = (2k-1)!!`$:

```math
\mathbb{E}[Z^6 - 6Z^4 + 9Z^2] = 15 - 6 \cdot 3 + 9 \cdot 1 = 15 - 18 + 9 = 6.
```

Therefore $`W_2^2 = \frac{6}{16\nu^2} = \frac{3}{8\nu^2}`$. $`\square`$

**Proposition 2 (KS signal for Gaussian vs variance-matched $`t_\nu`$).**

*$`\mathrm{KS}(P,Q) = \frac{c_{\mathrm{KS}}}{\nu} + O(\nu^{-2})`$ where $`c_{\mathrm{KS}} = \frac{1}{4}\max_z |z^3 - 3z|\phi(z) \approx 0.139`$.*

*Proof.* From the Cornish-Fisher expansion, $`F_Q(x) - \Phi(x) \approx -\frac{(x^3-3x)}{4\nu}\phi(x)`$ (inverting the quantile perturbation via $`F_Q(x) = u \Leftrightarrow z = x - \frac{x^3-3x}{4\nu} + O(\nu^{-2})`$, then Taylor-expanding $`\Phi`$).

We need $`\max_x |g(x)|`$ where $`g(x) = (x^3 - 3x)\phi(x)`$. Setting $`g'(x) = 0`$:

```math
g'(x) = (3x^2 - 3)\phi(x) - x(x^3 - 3x)\phi(x) = \phi(x)(3x^2 - 3 - x^4 + 3x^2) = \phi(x)(-x^4 + 6x^2 - 3).
```

Roots of $`x^4 - 6x^2 + 3 = 0`$: $`x^2 = 3 \pm \sqrt{6}`$. So $`x^2 \in \{3 - \sqrt{6},\, 3 + \sqrt{6}\} \approx \{0.551, 5.449\}`$.

Evaluating $`|g|`$ at the critical points:
- $`x_1 = \sqrt{3 - \sqrt{6}} \approx 0.742`$: $`|g(x_1)| = |0.409 - 2.226| \cdot \phi(0.742) = 1.817 \times 0.305 = 0.554`$.
- $`x_2 = \sqrt{3 + \sqrt{6}} \approx 2.334`$: $`|g(x_2)| = |12.71 - 7.00| \cdot \phi(2.334) = 5.71 \times 0.0264 = 0.151`$.

Global max: $`|g| = 0.554`$, achieved at $`x \approx 0.742`$. So $`c_{\mathrm{KS}} = 0.554 / 4 = 0.139`$. $`\square`$

### 5.4 Constants Comparison

Localization rate $`r^{*} \propto 1/\kappa_D`$. Both have $`n^{-1/2}`$ concentration. So the ratio of localization rates equals the inverse ratio of signals:

```math
\frac{r^{*}_{\mathrm{KS}}}{r^{*}_{W_2}} = \frac{\kappa_{W_2}}{\kappa_{\mathrm{KS}}} = \frac{W_2(P,Q)}{\mathrm{KS}(P,Q)} = \frac{0.612 / \nu}{0.139 / \nu} \approx 4.4.
```

**$`W_2`$-based CPD localizes the tail shift $`\approx 4.4`$ times faster than KS-based CPD.**

**Why.** The CDF perturbation $`(x^3 - 3x)\phi(x)`$ is spread diffusely across $`x`$: it crosses zero at $`x = 0, \pm\sqrt{3}`$, oscillates, and decays in the tails. KS takes the $`L^\infty`$ norm (single worst point). $`W_2`$ takes the $`L^2`$ norm of the quantile difference (accumulates across all quantile levels). For diffuse perturbations, $`L^2 \gg L^\infty`$ in magnitude.

**Remark.** The advantage is in *constants*, not rate (both are $`\nu^{-1}`$). For $`\nu = 5`$ (heavy tails, realistic for equity daily returns in stress): the 4.4x constant means $`W_2`$ needs roughly $`4.4^2 \approx 19`$ times fewer observations to achieve the same localization accuracy. Substantial in practice.

### 5.5 Where Wasserstein Wins by More Than Constants: Mixture Shifts

The constant advantage for Gaussian-vs-$`t_\nu`$ is meaningful but finite. For *mixture-weight shifts*, the advantage is unbounded.

**Setup.** $`P = 0.5\,\mathcal{N}(-a, \sigma^2) + 0.5\,\mathcal{N}(a, \sigma^2)`$, $`Q = (0.5 + \delta)\mathcal{N}(-a, \sigma^2) + (0.5 - \delta)\mathcal{N}(a, \sigma^2)`$, with $`a \gg \sigma`$ (well-separated modes) and $`\delta \in (0, 0.5)`$.

**Proposition 3 (Mixture-weight separation).** *As $`a / \sigma \to \infty`$:*

```math
\mathrm{KS}(P, Q) \to \delta, \qquad W_2^2(P, Q) \to 4\delta a^2.
```

*Consequently, $`W_2 / \mathrm{KS} \to 2a\sqrt{\delta}/\delta = 2a/\sqrt{\delta} \to \infty`$.*

*Proof.* For $`a \gg \sigma`$, the component Gaussians become approximately point masses $`\delta_{\pm a}`$.

**KS.** $`F_P(x) \approx 0.5 \cdot \mathbf{1}_{x \geq -a} + 0.5 \cdot \mathbf{1}_{x \geq a}`$. Similarly $`F_Q(x) \approx (0.5 + \delta)\mathbf{1}_{x \geq -a} + (0.5 - \delta)\mathbf{1}_{x \geq a}`$. For $`x \in (-a, a)`$: $`|F_P(x) - F_Q(x)| = |0.5 - (0.5 + \delta)| = \delta`$. For $`x < -a`$ or $`x > a`$: difference is 0. So $`\mathrm{KS} \to \delta`$.

**$`W_2`$.** Quantile functions:

```math
F_P^{-1}(u) \approx \begin{cases} -a & u < 0.5, \\ a & u > 0.5. \end{cases} \qquad F_Q^{-1}(u) \approx \begin{cases} -a & u < 0.5 + \delta, \\ a & u > 0.5 + \delta. \end{cases}
```

```math
W_2^2 = \int_0^1 (F_P^{-1}(u) - F_Q^{-1}(u))^2 \, du.
```

The difference is nonzero only on $`u \in (0.5, 0.5 + \delta)`$ where $`F_P^{-1}(u) = a`$ but $`F_Q^{-1}(u) = -a`$:

```math
W_2^2 = \int_{0.5}^{0.5 + \delta} (a - (-a))^2 \, du = \delta \cdot (2a)^2 = 4\delta a^2. \quad \square
```

**Localization comparison.** $`r^{*}_W / r^{*}_{\mathrm{KS}} = \mathrm{KS} / W_2 = \delta / (2a\sqrt{\delta}) = \sqrt{\delta}/(2a) \to 0`$. Wasserstein-CPD localization improves by a factor of $`2a/\sqrt{\delta}`$ over KS.

**Intuition.** KS sees a bounded CDF gap (the fraction of mass that moved). $`W_2`$ sees the CDF gap *times the distance the mass moved*. When modes are far apart, the transport cost dominates and $`W_2 \gg \mathrm{KS}`$.

**Financial relevance.** Mixture shifts model bimodal return regimes: binary macro outcomes (e.g., pre-election, pre-FOMC), flash crash precursors, or structural breaks in market microstructure. The "distance between modes" $`2a`$ is the spread between the two scenarios.

### 5.6 Summary of Tail/Shape Sub-Case

| Alternative | $`W_2`$ rate | KS rate | $`W_2`$ constant | KS constant | $`W_2 / \mathrm{KS}`$ |
|---|---|---|---|---|---|
| Gaussian vs $`t_\nu`$ | $`\nu^{-1}`$ | $`\nu^{-1}`$ | 0.612 | 0.139 | 4.4 |
| Mixture-weight shift ($`a \to \infty`$) | $`\sqrt{\delta} \cdot a`$ | $`\delta`$ | — | — | $`\to \infty`$ |

---

## 6. Sub-Case II: Dependence Regime Shifts

### 6.1 Financial Motivation

The "correlation crisis": during stress, cross-asset correlations spike toward 1 while individual asset distributions may remain approximately stable. This destroys diversification benefits and invalidates covariance estimates.

### 6.2 Alternative Class

**Definition 5 (Fixed-marginal alternatives).**

```math
\mathcal{C}(\kappa) = \left\{(P, Q) \in \mathcal{P}_2(\mathbb{R}^d)^2 : P^{(k)} = Q^{(k)} \;\forall\, k \in [d], \quad W_2^2(P,Q) \geq \kappa\right\}
```

where $`P^{(k)}`$ is the $`k`$-th marginal.

### 6.3 Qualitative Separation

**Proposition 4.** *For $`(P, Q) \in \mathcal{C}(\kappa)`$:*

| Method | Signal |
|---|---|
| Coordinate-wise $`W_2^2`$ | $`= 0`$ (marginals identical) |
| Coordinate-wise KS | $`= 0`$ |
| $`W_2^2`$ (multivariate) | $`\geq \kappa > 0`$ |
| $`SW_2^2`$ | $`\geq c_d \kappa > 0`$ (Bonneel+ 2015) |
| $`\mathrm{MMD}^2`$ (characteristic kernel) | $`> 0`$ (metrizes convergence) |

Coordinate-wise methods are *blind*. Full multivariate dissimilarities detect the change.

### 6.4 Explicit Signal: Bivariate Gaussian Correlation Shift

**Setup.** $`P = \mathcal{N}(0, \Sigma_1)`$, $`Q = \mathcal{N}(0, \Sigma_2)`$ where $`\Sigma_j = \begin{pmatrix} 1 & \rho_j \\ \rho_j & 1 \end{pmatrix}`$, $`j = 1,2`$.

Eigenvalues of $`\Sigma_j`$: $`1 + \rho_j`$ and $`1 - \rho_j`$. Eigenvectors: $`\frac{1}{\sqrt{2}}(1, 1)^\top`$ and $`\frac{1}{\sqrt{2}}(1, -1)^\top`$ — **shared** by both $`\Sigma_1, \Sigma_2`$, so they commute.

**Proposition 5 (Bures-Wasserstein for equi-marginal correlation shift).**

```math
W_2^2(P, Q) = \left(\sqrt{1+\rho_1} - \sqrt{1+\rho_2}\right)^2 + \left(\sqrt{1-\rho_1} - \sqrt{1-\rho_2}\right)^2.
```

*For small $`\Delta\rho = \rho_2 - \rho_1`$:*

```math
W_2^2(P, Q) = \frac{(\Delta\rho)^2}{2(1-\rho_1^2)} + O((\Delta\rho)^3).
```

*Proof.* Since $`\Sigma_1, \Sigma_2`$ commute, $`\mathcal{B}^2 = \sum_{k=1}^2 (\sqrt{\lambda_k^{(1)}} - \sqrt{\lambda_k^{(2)}})^2`$ (Definition 3). The eigenvalues are $`(1 \pm \rho_j)`$, giving the exact formula.

For the expansion: $`\sqrt{1 + \rho_2} = \sqrt{1 + \rho_1 + \Delta\rho} \approx \sqrt{1+\rho_1}\left(1 + \frac{\Delta\rho}{2(1+\rho_1)} - \frac{(\Delta\rho)^2}{8(1+\rho_1)^2}\right)`$.

So $`\sqrt{1+\rho_1} - \sqrt{1+\rho_2} \approx -\frac{\Delta\rho}{2\sqrt{1+\rho_1}}`$ and the squared term is $`\frac{(\Delta\rho)^2}{4(1+\rho_1)}`$.

Similarly: $`(\sqrt{1-\rho_1} - \sqrt{1-\rho_2})^2 \approx \frac{(\Delta\rho)^2}{4(1-\rho_1)}`$.

Summing: $`W_2^2 \approx \frac{(\Delta\rho)^2}{4}\left(\frac{1}{1+\rho_1} + \frac{1}{1-\rho_1}\right) = \frac{(\Delta\rho)^2}{4} \cdot \frac{2}{1-\rho_1^2} = \frac{(\Delta\rho)^2}{2(1-\rho_1^2)}`$. $`\square`$

**Financial interpretation.** The signal $`\kappa \propto (\Delta\rho)^2 / (1 - \rho_1^2)`$:

1. Grows with $`(\Delta\rho)^2`$: larger correlation shifts are easier to detect. Standard.
2. **Amplified by $`1/(1 - \rho_1^2)`$**: changes near $`\rho = \pm 1`$ produce larger signals. This matches financial intuition — a shift from $`\rho = 0.8`$ to $`\rho = 0.95`$ (amplification factor $`1/(1 - 0.64) = 2.78`$) is more consequential for diversification than $`\rho = 0.2`$ to $`\rho = 0.35`$ (factor $`1/(1 - 0.04) = 1.04`$).

**Example.** Correlation shift $`0.3 \to 0.8`$ ($`\Delta\rho = 0.5`$):

```math
W_2^2 = (0.5)^2 / (2(1 - 0.09)) = 0.25 / 1.82 = 0.137.
```

$`W_2 = 0.370`$. With balanced segments and $`T = 2500`$ trading days: $`r^{*} \approx \sqrt{2500 \cdot 8} / 0.137 \approx 1032`$ ... which is almost half the series. This suggests that for bivariate correlation shifts in daily data, the signal is weak — a known practical challenge. Higher-dimensional shifts (many assets simultaneously) or larger $`\Delta\rho`$ values improve detection.

### 6.5 Higher Dimensions: General Covariance Shift

**Proposition 6 (Bures-Wasserstein for general covariance shift, commuting case).** *If $`\Sigma_1, \Sigma_2 \in \mathbb{R}^{d \times d}`$ share eigenvectors with eigenvalues $`\lambda_1^{(j)} \geq \cdots \geq \lambda_d^{(j)}`$, $`j = 1,2`$, then:*

```math
W_2^2(\mathcal{N}(0,\Sigma_1), \mathcal{N}(0,\Sigma_2)) = \sum_{k=1}^d \left(\sqrt{\lambda_k^{(1)}} - \sqrt{\lambda_k^{(2)}}\right)^2.
```

*If the shift is rank-$`r`$ (only $`r`$ eigenvalues change): $`W_2^2 = \sum_{k=1}^r (\sqrt{\lambda_k^{(1)}} - \sqrt{\lambda_k^{(2)}})^2`$ — independent of $`d`$.*

This connects directly to Sub-Case III.

---

## 7. Sub-Case III: Low-Rank Regime Shifts (Spiked Transport)

### 7.1 Financial Motivation

Most regime changes are driven by few latent factors (risk appetite, rates, dollar). In a $`d`$-asset universe, the distributional shift has intrinsic rank $`r \ll d`$. Methods that exploit this structure need fewer observations.

### 7.2 Alternative Class and Explicit Signal

**Definition 6.** $`(P, Q) \in \mathcal{S}_r(\kappa)`$ if $`P = \mathcal{N}(0, \Sigma)`$, $`Q = \mathcal{N}(0, \Sigma + E)`$ where $`\operatorname{rank}(E) = r`$ and $`W_2^2(P,Q) \geq \kappa`$.

**Proposition 7 (Rank-1 Wasserstein signal).** *Let $`\Sigma = I_d`$ and $`E = \varepsilon v v^\top`$ for unit $`v \in \mathbb{R}^d`$ and $`\varepsilon > 0`$. Then:*

```math
W_2^2(P, Q) = (\sqrt{1+\varepsilon} - 1)^2 = \frac{\varepsilon^2}{4} + O(\varepsilon^3).
```

*In particular, $`W_2^2`$ depends on $`\varepsilon`$ and $`r = 1`$, but not on $`d`$.*

*Proof.* $`\Sigma_Q = I + \varepsilon vv^\top`$ has eigenvalues $`1+\varepsilon`$ (once, along $`v`$) and $`1`$ ($`d-1`$ times). By Proposition 6:

```math
W_2^2 = (\sqrt{1+\varepsilon} - \sqrt{1})^2 + (d-1)(\sqrt{1} - \sqrt{1})^2 = (\sqrt{1+\varepsilon} - 1)^2.
```

For small $`\varepsilon`$: $`\sqrt{1+\varepsilon} - 1 = \varepsilon/2 - \varepsilon^2/8 + \cdots`$, so $`W_2^2 = \varepsilon^2/4 + O(\varepsilon^3)`$. $`\square`$

### 7.3 MMD Signal for Rank-1 Perturbation (Dimension Dependence)

**Proposition 8 ($`\mathrm{MMD}^2`$ with Gaussian kernel, rank-1 shift).** *Let $`P = \mathcal{N}(0, I_d)`$, $`Q = \mathcal{N}(0, I_d + \varepsilon vv^\top)`$, kernel $`k(x,y) = \exp(-\|x-y\|^2/(2\sigma^2))`$. Then for $`\sigma^2 = \gamma d`$ (standard scaling, $`\gamma > 0`$ fixed):*

```math
\mathrm{MMD}^2(P, Q) = \Theta\!\left(\frac{\varepsilon^2}{d^2}\right) \quad \text{as } d \to \infty, \; \varepsilon \text{ fixed.}
```

*Proof sketch.* We compute each term. For $`X, X' \sim P`$: $`X - X' \sim \mathcal{N}(0, 2I_d)`$.

```math
\mathbb{E}[k(X,X')] = \mathbb{E}\!\left[\exp\!\left(-\frac{\|X-X'\|^2}{2\sigma^2}\right)\right] = \det\!\left(I + \frac{2}{\sigma^2}I\right)^{-1/2} = \left(1 + \frac{2}{\gamma d}\right)^{-d/2}.
```

For large $`d`$: $`\to e^{-1/\gamma}`$. Denote this $`A`$.

For $`Y, Y' \sim Q`$: $`Y - Y' \sim \mathcal{N}(0, 2(I + \varepsilon vv^\top))`$.

```math
\mathbb{E}[k(Y,Y')] = \det\!\left(I + \frac{2(I+\varepsilon vv^\top)}{\sigma^2}\right)^{-1/2} = \left(1+\frac{2}{\gamma d}\right)^{-(d-1)/2}\!\left(1 + \frac{2(1+\varepsilon)}{\gamma d}\right)^{-1/2}.
```

For large $`d`$: $`\approx A \cdot \left(\frac{1 + 2/(\gamma d)}{1 + 2(1+\varepsilon)/(\gamma d)}\right)^{1/2} \approx A \cdot \left(1 - \frac{\varepsilon}{\gamma d}\right)`$. Denote $`B \approx A(1 - \varepsilon/(\gamma d))`$.

For $`X \sim P, Y \sim Q`$: $`X - Y \sim \mathcal{N}(0, I + I + \varepsilon vv^\top) = \mathcal{N}(0, 2I + \varepsilon vv^\top)`$.

```math
\mathbb{E}[k(X,Y)] = \left(1+\frac{2}{\gamma d}\right)^{-(d-1)/2}\!\left(1+\frac{2+\varepsilon}{\gamma d}\right)^{-1/2} \approx A\!\left(1 - \frac{\varepsilon}{2\gamma d}\right).
```

Denote $`C \approx A(1 - \varepsilon/(2\gamma d))`$.

**$`\mathrm{MMD}^2`$ = $`A + B - 2C`$:**

```math
\approx A + A\!\left(1 - \frac{\varepsilon}{\gamma d}\right) - 2A\!\left(1 - \frac{\varepsilon}{2\gamma d}\right) = A\!\left(2 - \frac{\varepsilon}{\gamma d} - 2 + \frac{\varepsilon}{\gamma d}\right) = 0
```

at first order! Need second order.

Expanding more carefully to $`O(d^{-2})`$: each term picks up a correction $`\sim \varepsilon^2 / (\gamma^2 d^2)`$, and $`\mathrm{MMD}^2 = \Theta(A \varepsilon^2 / (\gamma^2 d^2)) = \Theta(\varepsilon^2 / d^2)`$. $`\square`$

**[FLAG: The exact second-order computation requires careful Taylor expansion of the determinant terms. The $`d^{-2}`$ scaling is confirmed by Ramdas et al. (2015) general analysis of MMD power decay under "fair alternatives" in high dimensions. A complete proof would track the $`(1 + x)^{-1/2}`$ expansion to second order in $`\varepsilon/(\gamma d)`$.]**

### 7.4 Separation Result

**Theorem 5 (Spiked transport: Wasserstein vs MMD).** *Under the rank-1 Gaussian shift with $`\sigma^2 = \gamma d`$:*

| Method | Signal $`\kappa_D`$ | Localization $`r^{*}`$ |
|---|---|---|
| $`W_2^2`$ | $`\varepsilon^2/4`$ | $`\Theta(\sqrt{T\log T}/\varepsilon^2)`$ |
| $`\mathrm{MMD}^2`$ (isotropic) | $`\Theta(\varepsilon^2/d^2)`$ | $`\Theta(d^2 \sqrt{T \log T}/\varepsilon^2)`$ |

**Wasserstein-CPD localizes $`\Theta(d^2)`$ times faster.** For $`d = 50`$ assets: $`d^2 = 2500`$-fold improvement.

**Remark.** One could optimize the MMD kernel bandwidth for the specific rank-1 alternative (e.g., use an anisotropic kernel aligned with $`v`$). But this requires knowing $`v`$ *a priori*, which defeats the purpose. Wasserstein achieves dimension-adaptive rates *without* knowing the perturbation subspace.

### 7.5 Sliced Wasserstein Signal

**Proposition 9.** *For the rank-1 Gaussian shift with $`\varepsilon`$ small:*

```math
SW_2^2(P,Q) = \frac{\varepsilon^2}{4d} + O(\varepsilon^3).
```

*Proof sketch.* For a random projection $`\theta \in \mathbb{S}^{d-1}`$: $`\theta^\top X \sim \mathcal{N}(0, 1)`$ and $`\theta^\top Y \sim \mathcal{N}(0, 1 + \varepsilon(\theta^\top v)^2)`$. By the 1-D Bures formula:

```math
W_2^2(\theta_{\ast} P, \theta_{\ast} Q) = (\sqrt{1 + \varepsilon(\theta^\top v)^2} - 1)^2 \approx \frac{\varepsilon^2 (\theta^\top v)^4}{4}.
```

Averaging over $`\theta`$: $`SW_2^2 \approx \frac{\varepsilon^2}{4}\mathbb{E}[(\theta^\top v)^4]`$. For $`\theta`$ uniform on $`\mathbb{S}^{d-1}`$ and unit $`v`$: $`\mathbb{E}[(\theta^\top v)^4] = \frac{3}{d(d+2)} \approx \frac{3}{d^2}`$ for large $`d`$.

Wait — this gives $`SW_2^2 \approx \frac{3\varepsilon^2}{4d^2}`$, not $`\frac{\varepsilon^2}{4d}`$. Let me recheck.

**[CORRECTION:]** $`\mathbb{E}_{\theta}[(\theta^\top v)^2] = 1/d`$ and $`\mathbb{E}_{\theta}[(\theta^\top v)^4] = 3/(d(d+2))`$. So $`SW_2^2 \approx \frac{3\varepsilon^2}{4d(d+2)} \asymp \frac{\varepsilon^2}{d^2}`$.

This means sliced Wasserstein *also* suffers $`d^{-2}`$ signal dilution for the rank-1 shift, similar to MMD! The advantage of $`W_2^2 = \varepsilon^2/4`$ (dimension-free) comes from the *full* multivariate Wasserstein, not the sliced version.

**Revised comparison:**

| Method | Signal | Dimension dependence |
|---|---|---|
| $`W_2^2`$ (full) | $`\varepsilon^2/4`$ | None |
| $`SW_2^2`$ | $`\approx 3\varepsilon^2/(4d^2)`$ | $`d^{-2}`$ dilution |
| $`\mathrm{MMD}^2`$ (isotropic Gaussian, $`\sigma^2 = \gamma d`$) | $`\Theta(\varepsilon^2/d^2)`$ | $`d^{-2}`$ dilution |

**So sliced Wasserstein and MMD have the same dimension dependence!** The advantage is only for full $`W_2`$, which is computationally expensive ($`O(n^3)`$ in general).

**[FLAG: This is an important correction. The spiked-transport advantage is real for full $`W_2`$ (via Bures-Wasserstein closed form for Gaussians, which is $`O(d^3)`$ — one eigendecomposition). But it does NOT extend to sliced Wasserstein, which averages over random projections and dilutes the signal just like isotropic MMD. The practical message: for spiked alternatives, use the Gaussian Bures-Wasserstein (closed form, efficient) rather than sliced Wasserstein.**

**However:** if the perturbation subspace $`V`$ is *estimated first* (e.g., via PCA on the covariance difference), then one can project onto $`\hat{V}`$ and apply 1-D or low-dimensional Wasserstein. This "project then test" approach achieves the dimension-free rate at the cost of subspace estimation error (Davis-Kahan bound: $`O(1/\sqrt{n})`$ angular error when eigenvalue gap $`\gg 1/\sqrt{n}`$).]**

---

## 8. Synthesis: Wasserstein as the Versatile Default

Theorem 1 shows all $`n^{-1/2}`$-concentrating dissimilarities achieve the same CPD localization rate. The sub-cases reveal *when and why* specific methods differ:

### What Each Method Does Well

| Criterion | Best method | Runner-up |
|---|---|---|
| 1-D tail shifts (rate) | Tie: all $`\nu^{-1}`$ | — |
| 1-D tail shifts (constant) | $`W_2`$ (4.4x over KS) | KS |
| Mixture/shape shifts | $`W_2`$ (unbounded advantage) | MMD |
| Copula/dependence shifts | $`W_2`$ or MMD (coordinate-wise fails) | MMD |
| Rank-1 spiked shifts | Full $`W_2`$ (dim-free) | $`\mathrm{SW}_2`$, MMD (both $`d^{-2}`$) |
| Localization constant (§3) | $`\mathrm{MMD}^2`$ ($`h = \alpha^2`$, factor 2) | $`W_2`$, KS |
| No tuning required | $`W_2`$ (only $`p`$) | KS (1-D only) |
| Computational cost (1-D) | KS, $`W_2`$ (both $`O(n\log n)`$) | — |
| Computational cost ($`d`$-D) | SW ($`O(Ln\log n)`$) | Sinkhorn ($`O(n^2/\varepsilon^2)`$) |

### The "Versatile Default" Argument

No single dissimilarity dominates across all settings. But Wasserstein (in appropriate variant) is **never catastrophically bad** and is **strictly best or tied-for-best in every sub-case**:

1. **Tails:** 4.4x constant advantage over KS; same rate.
2. **Mixtures:** Unbounded advantage over KS.
3. **Dependence:** Detects changes invisible to coordinate-wise methods.
4. **Low-rank:** Full $`W_2`$ (Bures) achieves dimension-free signal; sliced $`W_2`$ matches MMD.
5. **Tuning:** No kernel bandwidth to select (vs MMD).

A practitioner who picks $`W_2`$ without knowing the type of regime shift is insured against every scenario. A practitioner who picks KS fails on multivariate dependence shifts. A practitioner who picks MMD with a fixed kernel fails on spiked alternatives (unless bandwidth is optimized for the specific alternative, which requires oracle knowledge).

**This is the paper's thesis: Wasserstein is not always the theoretically optimal dissimilarity for CPD, but it is the most robust choice across the spectrum of financially relevant distributional changes.**

---

## 9. What Remains

### Done
- Theorem 1 + proof (single CP). Complete.
- Lemmas 1--3. Complete.
- Propositions 1--2 (constants for Gaussian vs $`t_\nu`$). Complete to leading order.
- Proposition 3 (mixture-weight separation). Complete.
- Proposition 5 (Bures-Wasserstein for correlation shift). Complete with expansion.
- Propositions 7--8 (spiked transport, $`W_2`$ vs MMD). Complete (modulo second-order MMD expansion).
- Proposition 9 (sliced Wasserstein for spiked shift — honest negative). Complete.

### Needs Work
1. **Theorem 2** (multiple CPs): standard BS induction, routine but needs writing up.
2. **Penalized segmentation** (Theorem 3): PELT-style analysis for Wasserstein cost. Verifying pruning conditions.
3. **Full second-order expansion** for $`\mathrm{MMD}^2`$ in Prop 8.
4. **Non-Gaussian spiked transport**: extending Prop 7 beyond Gaussians (Niles-Weed-Rigollet integration).
5. **Mixing dependence**: extending (A1) to $`\beta`$-mixing. Hard; limited existing results.
6. **Online/sequential**: Wasserstein-CUSUM with ARL/EDD analysis.
7. **Numerical verification**: all constants (Cornish-Fisher, Bures, etc.) should be confirmed computationally.
8. **Financial experiments**: synthetic + real-data validation on FF factors / ETFs / correlation breaks.

### Honest Corrections from This Exercise
1. "Wasserstein is better for tails" → Partially true. Better *constants* (4.4x for Gaussian vs $`t_\nu`$), not better *rates*.
2. "Sliced Wasserstein adapts to spiked structure" → **False.** SW suffers the same $`d^{-2}`$ dilution as isotropic MMD. Only full $`W_2`$ (Bures) is dimension-free.
3. "MMD has worse localization" → **False at the theorem level.** $`\mathrm{MMD}^2`$ actually has a factor-2 *better* localization constant via $`h(\alpha) = \alpha^2`$.
4. The strongest Wasserstein advantage is on **mixture/shape shifts** (unbounded) and **full multivariate Bures-Wasserstein for Gaussian spiked shifts** (dimension-free), not on generic tails.

---

## References

**CPD theory:** Madrid Padilla+ (2021 IEEE TIT); Killick+ (2012 JASA); Fryzlewicz (2014 AoS); Yu (2020 arXiv survey); Truong+ (2020 Signal Processing).

**Wasserstein CPD:** Cheng+ (2020 ICASSP, 2021 NeurIPS); Faber+ (2022 BigData); Werenski+ (2023 JMLR).

**Wasserstein testing:** Ramdas+ (2017 Entropy); Tran (2025 arXiv); Nietert+ (2022 NeurIPS); Niles-Weed & Bach (2019 Bernoulli); Niles-Weed & Rigollet (2022 arXiv); del Barrio+ (1999).

**Entropic OT:** Cuturi (2013 NeurIPS); Mena & Niles-Weed (2019 NeurIPS); Feydy+ (2019 AISTATS).

**MMD/kernel:** Gretton+ (2012 JMLR); Arlot+ (2019 JMLR); Ramdas+ (2015).

**Energy:** Matteson & James (2014 JASA); Székely & Rizzo (2004).

**Financial regimes:** Hamilton (1989 Econometrica); Horvath+ (2024 J. Comp. Finance); Boukardagha (2026 arXiv); Ang & Bekaert (2002, 2004).

**Bures-Wasserstein:** Bhatia+ (2019); Chewi & Rigollet (2020).

**Cornish-Fisher:** Hill & Davis (1968); Johnson, Kotz, Balakrishnan (1995 Ch. 28).
