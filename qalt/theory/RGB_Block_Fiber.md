# Joint-RGB block fiber: derivation and claim boundary

Status: **Proposal under independent adversarial audit.** This mechanism is the preregistered response to the scalar CIFAR B1 failure. It is not yet empirical evidence.

## 1. Observed signature and proxy

The scalar study ordered the nine detail channels by three RGB coordinates inside each of three Haar bands. The first coordinate of each band tied its autoregressive counterpart, but the next two incurred large regret. Four and eight scalar mixture components tied. The smallest proxy that preserves this signature is therefore a three-dimensional within-band residual with dense cross-color shape and a shared non-Gaussian radial scale. It deliberately omits learned coarse generation, spatial coupling beyond the fixed location features, and cross-band dependence.

## 2. Generative story

Fix a band, spatial site, coarse condition `u`, and energy stratum. Let the fitted location be `mu(u) in R^3`. The residual `R=H-mu(u)` follows:

1. Draw `J in {1,...,K}` from a categorical distribution with probabilities `w_1,...,w_K`.
2. Draw `Z ~ N_3(0,I_3)`.
3. Set `R=s_J L Z`, where `L L^T=Sigma`, `Sigma` is positive definite, and `det(Sigma)=1`.
4. Set `H=mu(u)+R`.

Exactly three standard-normal coordinates suffice. Given `G~N_3(0,I)`, set `U=Phi(G_1)`, choose the component interval `(P_{k-1},P_k]`, and remap `V=(U-P_{k-1})/w_k`. For every `v in [0,1]`, `Pr(J=k,V<=v)=w_k v=Pr(J=k)Pr(V<=v)`. Thus `V` is uniform and independent of `J`; it is also independent of `(G_2,G_3)` because `G_1` is. Hence `Z=(Phi^{-1}(V),G_2,G_3)` is standard normal independently of the selected component. This constructs the categorical branch and within-component noise from the same three-dimensional Gaussian prior; it is piecewise and is not a smooth normalizing flow.

The determinant constraint is a gauge: replacing `(Sigma,s_k^2)` by `(c Sigma,s_k^2/c)` leaves every covariance `s_k^2 Sigma` unchanged. Fixing `det(Sigma)=1` removes that scale ambiguity. It does not identify an arbitrary orthogonal factor `L`, but the density depends only on `Sigma`; a lower-triangular Cholesky factor with positive diagonal fixes sampling coordinates.

For a known minimal component count, positive weights, and distinct ordered scales, this parameterization is identifiable. The covariance `C=E[s_J^2]Sigma` identifies the shape because `det(C)=E[s_J^2]^3` and hence `Sigma=C/det(C)^(1/3)`. With `Sigma` known, the characteristic function on any ray is the finite Laplace sum `sum_k w_k exp(-s_k^2 q/2)`. Distinct real exponentials are linearly independent, so equality for every `q>=0` identifies the ordered scales and weights. The positive-diagonal Cholesky factor is then unique. Zero weights, duplicate scales, or an overfitted nonminimal count break uniqueness. Under conditioning, the parameter functions are identified only almost surely in the coarse condition; neural-network weights and values on null sets need not be identifiable.

## 3. Normalized density, line by line

Conditioned on component `k`, the covariance is `s_k^2 Sigma`. In three dimensions,

`det(s_k^2 Sigma) = (s_k^2)^3 det(Sigma) = s_k^6`.

Its square-root determinant is `s_k^3`. The inverse covariance is `s_k^{-2} Sigma^{-1}`. Therefore

`N_3(r;0,s_k^2 Sigma) = (2 pi)^(-3/2) s_k^(-3) exp{-rho(r)^2/(2s_k^2)}`,

where `rho(r)^2=r^T Sigma^{-1}r`. Summing normalized components gives

`q_B(r|u)=(2 pi)^(-3/2) sum_k w_k s_k^(-3) exp{-rho(r)^2/(2s_k^2)}`.

The weights are nonnegative and sum to one, so integrating term by term proves that `q_B` integrates to one. Numerically, compute each component log density,

`ell_k = log w_k - 3 log s_k - 3/2 log(2 pi) - rho^2/(2s_k^2)`,

then use log-sum-exp.

## 4. Fitting updates

Fit the common location first on training data. For residuals `r_i`, take the uncentered second moment, shrink it toward its isotropic trace with fixed weight `0.001`, and project its log eigenvalues onto the sum-zero box `[log(0.1),log(10)]`. Recompose

`Sigma=V diag(exp(theta)) V^T`, where `sum_j theta_j=0`.

This has determinant one because `det(Sigma)=exp(sum_j theta_j)=1` and condition number at most 100. The radial mixture then uses this fixed `Sigma`. Its E-step is

`gamma_ik = exp(ell_ik)/sum_j exp(ell_ij)`.

Let `M_k=sum_i gamma_ik`. Under the registered constraint `w_k>=epsilon_w`, the KKT equations give the exact water-filling solution

`w_k=max(epsilon_w,M_k/lambda)`,

where the unique `lambda>0` makes `sum_k w_k=1`. An interior coordinate obeys `M_k/w_k=lambda`; a boundary coordinate obeys `M_k/epsilon_w<=lambda`. The scale-dependent objective is

`-3 M_k log s_k - {1/(2s_k^2)} sum_i gamma_ik rho_i^2`.

Differentiate with respect to `s_k`:

`-3M_k/s_k + {sum_i gamma_ik rho_i^2}/s_k^3 = 0`.

Multiplying by `s_k^3` and solving gives

`s_k^2 = {sum_i gamma_ik rho_i^2}/(3M_k)`.

On the registered interval `[s_min^2,s_max^2]`, the unique maximizer is this unconstrained value clamped to the interval because the derivative changes sign only once. Ordering components by scale merely relabels the mixture. Since the shape is fitted once before EM and frozen, exact water filling and constrained scale maximization preserve EM monotonicity. The executable nevertheless recomputes observed likelihood after every iteration and fails on a decrease beyond numerical tolerance.

## 5. Exact block-parity class

Suppose the true conditional residual law is exactly the generative story in Section 2 for each band and the three bands are conditionally independent given the coarse field. An exact block sampler draws the declared categorical and Gaussian variables, so its conditional law equals the true fiber by construction. Sampling all three blocks and composing them with an exact coarse sampler and the inverse orthonormal Haar chart therefore reproduces the declared full endpoint law. More generally, for conditionally factorized true fibers `p_g` and approximate fibers `q_g`, direct expansion of the log density ratio gives

`KL(p_A prod_g p_g || q_A prod_g q_g) = KL(p_A||q_A) + E_pA sum_g KL(p_g(.|A)||q_g(.|A))`.

The same block law admits an exact scalar autoregressive factorization. Partition `Sigma` at coordinate `j` and define

`beta_j=Sigma_j,<j Sigma_<j,<j^{-1}`,

`nu_j=Sigma_jj-Sigma_j,<j Sigma_<j,<j^{-1} Sigma_<j,j`,

and `Q_{j-1}=r_<j^T Sigma_<j,<j^{-1}r_<j`. Bayes' rule gives the parent-dependent component weights

`rho_jk(r_<j) = a_jk / sum_l a_jl`, where `a_jk=w_k s_k^{-(j-1)} exp{-Q_{j-1}/(2s_k^2)}`.

For `j=1`, the parent block is empty, `Q_0=0`, `s_k^0=1`, `rho_1k=w_k`, `beta_1 r_<1=0`, and `nu_1=Sigma_11`.

The scalar conditional is

`q(r_j|r_<j,u)=sum_k rho_jk N(r_j; beta_j r_<j, s_k^2 nu_j)`.

Therefore the probability chain rule is not merely existential:

`log q_block(r|u)=sum_j log q(r_j|r_<j,u)`.

This equality is a mandatory samplewise implementation oracle. An unrestricted exact scalar conditional sampler ties the block sampler in distribution. With three bands in parallel it needs three conditional stages, while the joint sampler uses one. Depth nine applies only to a global scalar order with cross-band parents. An optimized comparator may refactor the fitted GSM into the joint sampler and tie depth. Both approaches touch all nine output coordinates once and require linear total output work and storage; the scalar-call depth result is not a wall-clock or allocator-memory theorem.

## 6. Strict restricted separation

Consider a factorized diagonal Gaussian residual comparator. Let `C=E[s_J^2] Sigma` be the GSM covariance and let `R` be its correlation matrix. Direct minimization of Gaussian cross-entropy over diagonal mean/covariance gives the conditional identity

`inf_D diagonal KL(P_GSM || N(mu,D)) = -0.5 log det(R) + KL(P_GSM || N(mu,C))`.

Both terms are nonnegative. Pointwise in condition `u`, if some absolute correlation is at least `rho_0`, the Schur-complement bound `det(R)<=1-rho_0^2` gives a margin of at least `-0.5 log(1-rho_0^2)` nats per affected RGB block. If this event has conditional-feature probability `alpha`, averaging gives the margin `alpha[-0.5 log(1-rho_0^2)]`; strict average separation requires correlation or nondegenerate scale mixing on a set of positive probability.

If `Sigma` has a nonzero off-diagonal entry, the block law has covariance

`Cov(R)=E[s_J^2] Sigma`,

which also has that nonzero off-diagonal entry. A diagonal Gaussian cannot equal this law.

Now suppose `Sigma` is diagonal but at least two scales are distinct with positive weights. For distinct coordinates `a` and `b`, conditional Gaussian independence gives

`E[R_a^2 R_b^2 | J]=s_J^4 Sigma_aa Sigma_bb`.

Averaging over `J` and subtracting the product of second moments yields

`Cov(R_a^2,R_b^2)=Sigma_aa Sigma_bb Var(s_J^2)>0`.

Thus the coordinates are dependent through the shared scale, whereas a factorized Gaussian has independent squared coordinates. In either case the densities differ on a set of positive measure. If the block law is absolutely continuous relative to the comparator, Gibbs' inequality gives strictly positive forward KL and the same positive excess expected log score.

There is no uniform significant margin: correlations may approach zero and distinct scales may approach one another. This is a restriction result, not dominance over a complete variational autoencoder or unrestricted latent method. Either can implement the same stochastic block decoder and tie. Haar maps `N` endpoint coordinates to exactly `N` coarse-plus-detail coordinates, and RGB grouping drops no coefficient. The active coarse field is smaller, but the generated fiber completes the `N`-dimensional state. The recycled-prior map gives equal base dimension for each block, but it is piecewise and many-to-one rather than a smooth invertible flow. Equal dimensionality does not increase mutual information or unrestricted sufficiency.

Two no-refit fitted-model ablations isolate dependence without changing marginal laws. Let `P4=prod_j q_B,j` be the product of the exact fitted block marginals. Under the declared block law,

`E_B[-log P4(R)+log q_B(R)] = KL(q_B || prod_j q_B,j)`,

the conditional total correlation, which vanishes exactly under independence. To remove only componentwise correlation, put `d=(det(diag(Sigma)))^(1/3)`, `Sigma_0=diag(Sigma)/d`, and `s_0k=s_k sqrt(d)`. Then `det(Sigma_0)=1` and `s_0k^2 Sigma_0=s_k^2 diag(Sigma)`, so every component marginal variance and the shared radial component are preserved while off-diagonal covariance is removed. The derived `s_0k` need not remain inside the fitting scale box and must not be clipped; code may evaluate the covariance `s_k^2 diag(Sigma)` directly. `P4/Z4` are no-refit ablations and are not used in route density-range bounds.

## 7. Diagnostics implied by the model

Under component `k`, `rho^2/s_k^2` has a chi-square distribution with three degrees of freedom. The radial probability-integral transform is therefore

`F(rho^2)=sum_k w_k F_chi2_3(rho^2/s_k^2)`.

It should be uniform under the fitted law on independent calibration data. Whitening gives `Y=Sigma^{-1/2}R`; the direction `A=Y/||Y||` is uniform on the unit sphere and satisfies `E[AA^T]=I_3/3`. Radial PIT failure indicates scale-mixture misspecification; angular second-moment failure indicates shape or non-elliptical dependence; cross-band energy correlation indicates failure of conditional block independence. These diagnostics correspond to distinct repairs and cannot be replaced by one aggregate NLL.

## 8. Remaining empirical and novelty burden

The scalar failure predicts that a dense Gaussian block `B1` should close the cross-color part of the gap, while `B4` combines that repair with the already observed radial-tail mechanism. The v2 repair holdout tests that prediction only as adaptive development: those records entered the earlier v1 fits, so ordinary independent finite-class coverage is invalid even though v2 models are refit and v2 scores remain unopened. A theorem-backed certificate requires untouched images after a separate freeze. A passing conditional study still lacks an active generator, endpoint samples, accelerator measurements, VAE/full-flow baselines, and FID/FVD. Dense elliptical mixtures, wavelet charts, and block sampling are established ingredients; any contribution claim must be limited to the audited routing/certificate combination and supported by a separate primary-source novelty review.

Concrete counterexamples delimit the result. A one-component diagonal shape has exactly zero factorized-Gaussian gap. Component-dependent means, skewness, or nonelliptical angular structure can make a richer scalar autoregression better. A scale shared across bands or sites preserves each block marginal while invalidating endpoint product closure. Exact conditional NLL does not imply improved FID/FVD, and every density statement concerns the frozen dequantized law rather than discrete pixel mass.
