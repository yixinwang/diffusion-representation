# LIFT-FM preregistration: VAE-free lifting-induced fiber transport

Frozen before implementing the new runner. Date: 2026-08-26.

## Target and comparison class

Let dequantized observations be `X in R^D` and conditioning be `C`. A fixed orthonormal lifting transform `H` gives

`Y = H X = (Z, R_1, ..., R_B)`,

where `Z` is the coarse active quotient and the `R_b` are same-dimensional retained detail blocks. LIFT-FM runs a K-evaluation rectified flow only on `Z`; each `R_b` is sampled once from a fitted conditional block transport. The inverse lifting is exact and is the only decoder.

The primary full-flow baseline receives the same `H`, source law, conditioning, split, optimizer, network family, training updates, and NFE, but runs the repeated field on every coordinate of `Y`. It is allowed the same standardization. A same-information full baseline that copies the LIFT split is expected to tie endpoint quality; no universal strict-quality claim is registered.

The equal-dimensional VAE control has latent dimension D, the same conditioning, comparable MLP depth/width, a standard-normal prior, and a diagonal Gaussian posterior. Its decoder cost, beta search, and training updates are counted.

## Theorem-aligned advantage registered before implementation

### Endpoint law

If the active sampler has endpoint law `Q_Z` and each one-pass map has conditional law

`Q_b(d r_b | z, r_<b, c)`,

then, because `H` is bijective,

`KL(P_X || Q_X) = KL(P_Z || Q_Z) + sum_b E_P KL(P_b(.|Z,R_<b,C) || Q_b(.|Z,R_<b,C))`.

Therefore exact conditional fibers give exact endpoint-law parity with a full sampler sharing `Q_Z`. Approximate parity to epsilon follows when the sum of held-out conditional KL excesses is at most epsilon. Because H is orthonormal, both KL and Euclidean W2 are invariant under H; no noninjective VAE decoder argument is used.

### Repeated-compute ratio

For one-level Haar lifting, the active coordinate fraction is rho=1/4 for images and rho=1/8 for three-dimensional video lifting. If the one-pass fiber costs at most one linear full-state pass, the registered token-update ratios are

`r_linear = rho + (1-rho)/K`.

At K=20 these are 0.2875 for images and 0.16875 for videos. For attention-dominated repeated fields, the conservative proxy is

`r_attention = rho^2 + 1/K`,

which is 0.1125 for images and 0.065625 for videos. Promotion requires measured end-to-end transport time below the full-flow time after including the lifting, conditional fiber, assembly, and any common output work.

### Why a joint block is needed

For a one-pass product fiber `Q(R|Z)=prod_j Q_j(R_j|Z)`, the best possible product approximation has irreducible conditional KL

`inf_Q E KL(P(R|Z)||Q(R|Z)) = E KL(P(R|Z)||prod_j P(R_j|Z))`,

the conditional total correlation. The existing scalar CIFAR route exposed within-band color dependence, so the first new fiber is a joint three-coordinate block: orientations at a spatial site for grayscale digits and RGB at a band/site for color data. Scalar independence is a required ablation.

### Equal-dimensional VAE boundary

An invertible lifting has exactly zero representation reconstruction error. A finite-rate stochastic VAE obeys `I(X;Z) <= E KL(q(Z|X)||p(Z))`; hence its squared-error distortion is bounded below by the source rate-distortion function `D_X(R)`. For an absolutely continuous nondegenerate source, `D_X(R)>0` at every finite R. This is a strict representation result against finite-rate VAEs, not against a beta=0 identity autoencoder or an unrestricted VAE allowed infinite rate.

## Frozen small-scale study

Dataset: `sklearn.datasets.load_digits`, stratified train/validation/test split with final test indices sealed by a SHA-256 manifest. No test sample is used for architecture, beta, NFE, checkpoint, or fiber selection.

Methods:

1. Full 64-coordinate rectified flow in the shared Haar chart.
2. LIFT-FM with 16 active coarse coordinates and a joint one-pass 3-orientation fiber.
3. LIFT-FM scalar-fiber ablation with the same active model.
4. Same-information split-copy control, which must exactly share the LIFT samples and tie.
5. Equal-dimensional conditional VAE, latent dimension 64, beta selected only on validation from a frozen grid.

Primary quality metric: class-conditional energy score on held-out data (lower is better). Secondary metrics: sliced Wasserstein distance, RBF MMD, pixel Frechet distance, requested-label accuracy under a train-only classifier, class balance, nearest-neighbor coverage, and exact/expected reconstruction MSE. Efficiency: measured CPU sampling time, analytic multiply-add proxy, parameter count, and peak activation proxy. Search/training time is reported separately.

NFE frontier: 4, 8, 16, and 32 for both flow methods. Main NFE is 16. Seeds: 4100--4104 after a single disjoint smoke seed 4099.

## Promotion gates and falsifiers

The method is promoted only if all conditions hold on the untouched test split across the five confirmation seeds:

1. Joint-fiber LIFT-FM has a lower mean energy score than the scalar-fiber ablation, with a paired 95% interval excluding zero.
2. At NFE 16, joint-fiber LIFT-FM is noninferior to the full flow within an energy-score margin of 0.02 and a SWD margin of 0.02.
3. Joint-fiber LIFT-FM is faster than full flow after all nonshared work is included; analytic operation accounting agrees in ordering.
4. The split-copy control is numerically identical in generated samples and metrics.
5. Haar round-trip error is below 1e-10.
6. The equal-dimensional VAE has strictly positive reconstruction MSE at its validation-selected finite beta; LIFT reconstruction is exact.

Immediate falsifiers are: conditional block held-out NLL not better than scalar; LIFT energy/SWD outside the margins; total measured time ratio at least one; test-manifest access before freeze; or an exact-copy mismatch. A failure triggers a mathematical diagnosis using conditional total correlation or closure defect before any revised method is evaluated.
