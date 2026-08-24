# Independent joint-RGB block audit trace — 24 August 2026

Three independent agents audited the theorem, recycled-data inference, and numerical implementation after the scalar CIFAR failure. Each called the standing goal first and made no repository edits.

## Theorem adversary

The normalized model is `sum_k pi_k N_3(mu,s_k^2 Omega)` with `Omega` positive definite, `det(Omega)=1`, and positive-diagonal Cholesky factor. The audit checked the Jacobian normalization, log-sum-exp density, exact sampling, gauge, and identifiability. It supplied these corrections:

- A joint GSM has an exact scalar autoregressive factorization. With `beta_j=Omega_j,<j Omega_<j,<j^{-1}`, conditional variance factor `nu_j`, and `Q_{j-1}=r_<j^T Omega_<j,<j^{-1}r_<j`, the posterior component weights are proportional to `pi_k s_k^{-(j-1)} exp{-Q_{j-1}/(2s_k^2)}`. The exact scalarization must tie the joint density samplewise. A scalar model with weights fixed by coarse stratum is restricted and need not tie.
- Exactly three standard-normal coordinates suffice for a three-dimensional mixture. Transform the first Gaussian to a uniform, use its mixture-weight interval to choose the component, remap within that interval to a new independent uniform/Gaussian coordinate, and retain the other two normals. Thus no extra categorical prior dimension is required; the map is piecewise, not a smooth normalizing flow.
- Let `C=E[s_K^2] Omega` be the GSM covariance and `R` its correlation matrix. The optimal factorized-Gaussian gap equals `-0.5 log det(R)+KL(P_GSM || N(mu,C))`. Correlation gives a margin `-0.5 log(1-rho0^2)` when `|rho|>=rho0`; nondegenerate scale mixing gives a strict fourth-moment gap. No uniform margin exists without separated correlation or scales.
- If the declared block factorization and active law are exact, endpoint KL equals active-law KL. Approximate block errors add by the conditional KL chain rule.
- A within-band scalar autoregression needs three stages with bands parallel, not nine. An optimized comparator may refactor the same GSM into the joint sampler and tie depth. The valid theorem is equality to the scalar factorization under a scalar-call restriction, not universal efficiency dominance.
- Identifiability requires known component count, positive weights, distinct ordered scales, determinant-one shape, and a Cholesky gauge. Duplicate components, zero weights, or unrestricted neural parameters are not identifiable.

## Statistical adversary

The proposed 5,000-image v2 split is recycled development data, not fresh calibration: those records entered the v1 fits whose validation signature selected this repair. Refitting and excluding them from v2 fitting do not restore ordinary finite-class coverage. The v2 score is prospectively unopened but adaptive development. A theorem-backed empirical Bernstein certificate requires untouched sources, such as the official test only after a separate final freeze.

The audited v2 estimand averages five frozen seeds and all coefficients inside each image, then resamples images within class. The required arms are product-scalar `O4`, shared-component diagonal block `D4`, dense block `B1/B4/B8`, within-band scalar autoregression `A4/A8`, and full cross-band `I4/I8`. A second audit caught that independently optimized `D4-O4` mixes joint dependence with marginal-family restrictions. The corrected clean ablations are `P4`, the product of exact fitted `B4` marginals, and `Z4`, a zero-correlation transform that preserves every component marginal variance and the shared radial label. `D4` and `O4` remain practical controls. The audit recommends 9,999 fixed stratified cluster-bootstrap draws, exact stratified Welch tests, TOST equivalence, and Holm-adjusted claim-level p-values rather than calling ordinary percentile endpoints multiplicity corrected.

For a future untouched stratified certificate, form classwise empirical Bernstein bounds with `n_h=500` and multiplicity over routes and ten classes, then average class bounds. A v2 value computed on recycled data must be labeled a diagnostic pseudo-certificate. The density range must use registered final shape eigenvalue, scale, weight, and residual bounds rather than observed extrema.

## Numerical adversary

The first implementation should fit a common residual shape once and freeze it across `K=1,4,8`; otherwise radial mixing is confounded with different shapes. Use an uncentered residual second moment after the fitted zero-mean location, shrink it toward its isotropic trace, project log eigenvalues to registered determinant-one and condition constraints, and store the actual spectrum and log determinant. Never form a matrix inverse; use triangular solves.

With fixed shape, the EM updates are `gamma_ik` from stable component log densities and `s_k^2=sum_i gamma_ik m_i/(3 sum_i gamma_ik)`. The exact lower-bounded simplex M-step needs water filling; the earlier affine weight-floor update is stable but not the constrained maximum. Recompute likelihood after each M-step and fail on a material decrease. Fit all component counts with the same means, shapes, samples, start count, and budget.

Required diagnostics are chi-square-mixture radial PIT grids, whitened angular second and fourth moments, cross-band normalized energy correlations, band-by-spatial score heatmaps, shape spectra/condition numbers, component boundary hits, exact scalarization/copy ties, common-prior invariance, round trip, and leakage hashes.

## Decisions

1. V2 is an adaptive repair-development study; it cannot promote a finite-sample routing certificate.
2. The main efficiency comparator is depth-three within-band autoregression. Full depth-nine autoregression remains a cross-band control.
3. Joint GSM quality must tie its exact scalarization. Any claimed benefit versus independently optimized autoregression is restricted-family or finite-sample evidence.
4. The novelty candidate remains the certified adaptive choice of a lower-depth equal-dimensional block closure, not the classical GSM itself.
5. Mixed routing uses `B4` or within-band `A8` per band and measures regret directly against the nonroute `I8` fallback. Its unbatched call count is `9-2|S|`, with depth three unless all three bands are joint. Exact quality parity and one-versus-three depth belong to `B4/E4`; `A8` is a separately fitted empirical control.
6. The recycled v2 split supports only adaptive-development summaries. Its downstream pixel arrays exclude discovery indices, but the discovery data selected the repair and were already deserialized in the prior run.
