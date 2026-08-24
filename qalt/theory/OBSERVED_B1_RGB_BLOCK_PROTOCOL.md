# Prospective B1-v2 joint-RGB block protocol

Status: **Frozen adaptive repair-development protocol, registered before any v2 holdout score.** The mechanism was derived from the frozen v1 signature: autoregressive gains appeared only in the second and third color channels within each Haar band, while increasing scalar mixture capacity did not help. Independent theorem, statistical, and numerical audits are preserved in `research/agent_traces/20260824_rgb_block_audit.md`.

## Adaptive-data boundary

The original 5,000-image validation set is discovery data and is not reused. Deterministically repartition the former 45,000-example fitting set within each class using `numpy.default_rng(20260824)`: 4,000 images per class form the v2 fitting set and 500 per class form a 5,000-image repair holdout. Refit every comparator from scratch on the 40,000 v2 fitting images. The repair is frozen before its own holdout score, but the holdout is recycled development data because it contributed to the v1 fits underlying the failure signature. It cannot support an ordinary independent empirical Bernstein certificate. No earlier parameter is reused. The official test batch remains sealed for a separately frozen confirmation.

Use development seeds `2100..2104`. Each seed supplies paired record-keyed uniform dequantization and a shared counter-based sample of at most 250,000 fitting sites. Seeds are algorithmic variability, not independent datasets. Images are the inference clusters.

## Block law and algorithm

For Haar band `b in {HL,LH,HH}` and site `j`, let `R_bj in R^3` be the RGB residual after the common coarse-only ridge location model. Within each coarse-energy stratum, the proposed one-shot density is

`q_B(R=r | u) = sum_{k=1}^K w_k N_3(r; 0, s_k^2 Sigma)`.

The covariance shape is positive definite and has determinant one. Fit the uncentered residual second moment, shrink it toward its isotropic trace with fixed weight `0.001`, and project its log eigenvalues onto the sum-zero box `[log(0.1),log(10)]`; this enforces determinant one and condition number at most 100 after normalization. Fit this shape once and reuse it bitwise for `K=1,4,8`. Scale absorbs the removed determinant. Mixture weights are at least `1e-4`; radial scales lie in `[0.05,2]`. EM uses the Mahalanobis radius, exact three-dimensional normalization, deterministic geometric/radial initializations, and the exact water-filling lower-bounded-simplex weight update. Recompute observed likelihood after each M-step and fail on a material decrease. All three color coordinates of one band are sampled jointly. The three bands ignore one another conditional on coarse features and can be sampled in one parallel stage.

The exact model-class mechanism is elliptical: draw a shared component, then one three-dimensional Gaussian vector. A dense shape represents cross-color conditional means through the equivalent Gaussian autoregressive factorization, while the shared component represents non-Gaussian radial tails. This is not merely three independent scalar mixtures.

## Candidates and information fairness

- `B4`: proposed dense-shape four-scale RGB block mixture for each band and energy stratum.
- `B1`: dense conditional Gaussian block with identical means and strata.
- `P4`: no-refit product of the three exact fitted `B4` marginals, `prod_c sum_k w_k N(r_c;0,s_k^2 Sigma_cc)`. It preserves every fitted marginal law and removes all RGB dependence.
- `Z4`: no-refit marginal-preserving zero-correlation transform of the fitted `B4`. If `d=(det(diag(Sigma)))^(1/3)`, set `Sigma_0=diag(Sigma)/d` and `s_0k=s_k sqrt(d)`. Then `det(Sigma_0)=1` and every component marginal variance is unchanged, while off-diagonal covariance is removed. Derived scales are exempt from the fitted scale box and are never clipped; evaluate `s_k^2 diag(Sigma)` directly if convenient.
- `D4`: independently refitted, optimized shared-component radial mixture with diagonal determinant-one shape under the same budget. It is a strong practical restricted control, not a pure mechanism isolator.
- `B4-unconditional`: identical dense block mixture without energy strata.
- `B8`: eight-scale same-information block control.
- `O4`: product of the frozen scalar parallel four-mixture coordinates, independently refitted from scratch. It is a strong practical factorized control, but contrasts involving it mix dependence with marginal-family flexibility.
- `A4/A8`: independently optimized within-band RGB scalar autoregression, with all three bands parallel; each regression jointly fits the permitted coarse features and earlier within-band colors. This is the primary strong depth-three iterative comparator.
- `I4/I8`: independently optimized full nine-channel scalar autoregression, jointly fitting the permitted coarse features and all earlier detail channels; comparison with `A4/A8` diagnoses cross-band dependence.
- `E4`: the exact scalarization of the fitted `B4`, whose parent-dependent component weights follow Bayes' rule; it must tie `B4` samplewise.
- exact code-path copy controls that must tie bitwise.

Every fitted candidate receives the same images, dequantization, permitted coarse features, fitting sites, component starts, optimization budget, and complete nine-dimensional detail representation; `P4` and `Z4` are deterministic no-refit ablations of `B4`. `B1/B4/B8` reuse a bitwise common coarse-only location fit so radial capacity and shape are isolated. `O/A/I` independently optimize their coefficients using the same permitted coarse information; `A/I` additionally use only their declared data parents, making them stronger controls rather than coefficient-tied ablations. Freeze RGB color order; do not select it on the holdout. All methods receive exactly the same nine-dimensional standard-Gaussian base vector. A block reuses the first Gaussian coordinate through its normal CDF for component selection, remaps the within-component interval back to an independent Gaussian, and retains the other two normals. Thus no extra categorical dimension is hidden. Parameter counts and any ignored coordinates are reported.

## Scores, route, and gates

Evaluate proper holdout joint conditional log density at data parents. For arm `m`, average all sites and the five fixed seeds inside image, then target the balanced-class mean `mu_m=(1/10) sum_h E[L_im | class=h]`, conditional on the single realized 40,000-image fitting sample, frozen fitted models, and this realized five-seed ensemble. It estimates new-source score variation for that fixed ensemble, not retraining-sample variability or future seeds. Report nats per one of 2,304 detail coefficients. Use 9,999 fixed class-stratified image-bootstrap draws with RNG seed `20260824`, resampling each image's complete vector of arm scores within class; sites, coefficients, and seeds are not resampling units. Report nominal 2.5/97.5 percentile paired cluster-bootstrap intervals separately.

Freeze the stratified Welch test as follows. For an image-level paired contrast `Delta`, let `delta_hat=(1/10)sum_h mean_h`, `V_hat=sum_h (1/10)^2 s_h^2/n_h`, and `nu=V_hat^2 / sum_h {[(1/10)^2 s_h^2/n_h]^2/(n_h-1)}`. A superiority null at margin `m` has `p=1-F_t,nu((delta_hat-m)/sqrt(V_hat))`. Equivalence at half-width `e=0.01` uses `p_lower=1-F_t,nu((delta_hat+e)/sqrt(V_hat))`, `p_upper=F_t,nu((delta_hat-e)/sqrt(V_hat))`, and `p_equiv=max(p_lower,p_upper)`. Apply Holm to the ten claim-level p-values. A gate passes only when its Holm-adjusted p-value is at most `0.05` and its point estimate lies on the required side. All intervals and p-values are nominal adaptive-development summaries that decide advancement to untouched confirmation, not coverage claims. The primary paired quantities are:

1. `NLL(O4)-NLL(B4) > 0.01` (practical factorized-control gain).
2. `NLL(P4)-NLL(B4) > 0.01` (marginal-preserving joint-dependence gain).
3. `NLL(B1)-NLL(B4) > 0.01` (non-Gaussian radial gain at common shape).
4. `NLL(Z4)-NLL(B4) > 0.01` (component-marginal-preserving dense-correlation gain).
5. `NLL(D4)-NLL(B4) > 0.01` (gain over an independently optimized diagonal control).
6. `NLL(B4-unconditional)-NLL(B4) > 0` (coarse-energy conditioning gain).
7. `NLL(B4)-NLL(B8)` is equivalent within `[-0.01,0.01]`.
8. `NLL(B4)-NLL(A8)` is equivalent within `[-0.01,0.01]`.
9. `NLL(B4)-NLL(I8)` is equivalent within `[-0.01,0.01]`.
10. `NLL(I4)-NLL(I8)` is equivalent within `[-0.01,0.01]`; otherwise `I4` is a weak reference.

The frozen route family has three band blocks and eight routes. A route density is the product of `B4` for selected bands and the corresponding within-band `A8` scalar conditionals elsewhere, evaluated on data parents; `I8` is the nonroute reference and fallback. For image `i`, define `D_ir=(1/5) sum_s {-log q_r^s(X_i^s)+log q_I8^s(X_i^s)}/2304`; there is no log of an averaged density. Compute the registered class-stratified empirical Bernstein statistic with `M=8`, `alpha=0.05`, and unchanged `epsilon=0.01`, but label it a **diagnostic pseudo-certificate** because this holdout is recycled. For route `r` and class `h`, with `n_h=500`, sample variance `V_rh`, deterministic range `[a_rh,b_rh]`, and `l=log(2*8*10/0.05)`, set `U_rh=mean(D_rh)+sqrt(2 V_rh l/n_h)+7(b_rh-a_rh)l/{3(n_h-1)}` and `U_r=(1/10)sum_h U_rh`. A route is diagnostically eligible only if `U_r<=0.01`. Among eligible routes choose lexicographically by `(critical depth, unbatched calls, mask)`; if none improves on the fallback, abstain to `I8`. The statistic targets only the finite fitted ensemble. Do not attach coverage or call eligibility certified. On future untouched data replace `500` by the frozen class count. Analytic density ranges use residual coordinates in `[-2,2]`, final shape eigenvalues in `[0.1,10]`, scales in `[0.05,2]`, and weights at least `1e-4`; observed extrema never define a bound. A theorem-backed route decision is deferred to untouched confirmation images.

The analytic critical-depth proxy is one for three parallel `B4` bands, three for the fair fitted within-band `A8` comparator, and nine for full `I8`. A mixed `B4/A8` route has critical depth one only when all bands are joint and three otherwise; its unbatched conditional-call count is `9-2|S|` for joint-band set `S`. If like heads are batched, the separate head-invocation count is `3*1[|S|<3]+1[|S|>0]`. Exact scalarization `E4` ties `B4` samplewise at scalar depth three. `A8` is an independently optimized strong depth-three empirical control and must pass its registered equivalence gate; it is not the exact chain-rule factorization. An optimized implementation can refactor `E4/B4` and tie depth one. Report measured latency and peak allocator memory separately; this pilot does not establish full diffusion or latent-diffusion efficiency.

## Mandatory diagnostics

For each band and stratum report radial probability-integral-transform calibration under the fitted chi-square mixture, whitened angular second moments against `I_3/3`, angular fourth moments against `1/5` on the diagonal and `1/15` off diagonal, normalized cross-band residual-energy correlations, and a `3x16x16` spatial heatmap of `B4-I8` block-score difference. Frozen mechanism tolerances are maximum PIT-grid deviation `0.02`, angular second/fourth-moment deviation `0.03`, and absolute cross-band energy correlation `0.05`. Report image-cluster uncertainty, covariance spectra/condition numbers, boundary hits, effective component masses, clipped-location fraction, finite scores, `B4-E4`, fitted marginal, zero-correlation component-marginal, and copy ties, same-prior sample checks, round trip, hashes, and exact index intersections. The v2 runner proves that no discovery pixel enters a v2 fitting array, repair-holdout scoring array, model-input statistic, or diagnostic after the recorded loader-wide training-batch deserialization. Discovery records did inform the already recorded decision to launch B1-v2; this is why v2 has no coverage. Class labels are used only to apply the frozen stratified manifest and are not model inputs. Exact downstream index intersections and hashes are emitted. The official test batch remains undeserialized.

## Stop and revision rules

- If `B4` fails the practical `O4` gate, retire this `B4` repair, not every possible joint family.
- If `B4` fails the clean `P4` gate, reject the claimed joint-dependence explanation even if it beats independently fitted controls.
- If `B4` fails against `Z4`, retire the off-diagonal mechanism. A `Z4` pass with an optimized `D4` failure means the fitted ablation mattered but dense capacity was not practically necessary. Any simpler shared-radial reinterpretation must be registered on untouched data rather than selected here.
- If `B8` materially beats `B4`, increase radial capacity only through a new registration.
- If `B4` loses to `A8`, the within-block elliptical family is inadequate. If it ties `A8` but loses to `I8`, the missing mechanism is cross-band dependence.
- If descriptive equivalence passes but the pseudo-certificate abstains, retain only descriptive quality and seek untouched data; never change `epsilon`.
- Even if every gate passes, this is conditional-density development evidence. Endpoint scaling still requires a learned active generator and fair full/latent diffusion baselines.
