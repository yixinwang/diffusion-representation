# Independent multiscale audit trace — 23 August 2026

## Audit remit

Three independent routes were requested after the procedural Stage-A confirmation: (i) adversarial theorem and claim checking, (ii) alternative conditional-fiber families, and (iii) observed-data provenance, leakage, and fairness. The agents were asked for concrete equations, counterexamples, or executable checks rather than status reports. A fourth route is deriving a certified adaptive-computation theorem and is recorded separately when its audit completes.

## Theorem and claim adversary

- Exact composition is valid only for a measurable volume-preserving bijection, a topologically ordered conditional graph, normalized measurable kernels, and exact active/fiber samplers. Discrete images require an explicit dequantization or probability-mass treatment.
- The local token count is `K n_a + (N-n_a)` and is smaller than `K N` exactly when `(K-1)(N-n_a)>0`. The `O(K n_a+N)` statement additionally assumes fixed-cost local kernels and a linear-time structured chart. It is not a universal lower bound. Both full and split generators require `Omega(N)` output memory; live-array ratios are implementation conventions, not a memory theorem.
- The log-score gap applies to a fixed chart and a restricted Gaussian conditional family. A bijection cannot improve mutual information, unrestricted Bayes risk, or sufficiency. The Stage-A arm named `diagonal_vae` is a scalar Gaussian detail model, not a complete VAE.
- Stage A samples the active coefficients from the procedural truth and fits only the fiber. It contains no learned active law, full flow/diffusion training, full endpoint NLL, FID/FVD, or observed data. The exact same-information tie uses the same conditional implementation and is a necessary identity check, not independent superiority evidence. The timer measures a proxy local kernel and decoder; the memory row is a formula, not allocator telemetry.
- The strongest general identity is the conditional KL chain rule. For a common topological factorization,
  `KL(P||Q)=KL(P_A||Q_A)+sum_j E_P KL(P_j(.|Pa_j)||Q_j(.|Pa_j))`.
  This suggests routing iterative computation only to blocks whose one-shot closure defect is not certified small.
- Prior-art overlap with wavelet flows, multiresolution continuous normalizing flows, wavelet diffusion, and latent diffusion makes the fixed-chart/fiber construction alone an insufficient novelty claim.

## Independent fiber-family portfolio

Four distinct families were proposed: a finite conditional Gaussian mixture; a few-color triangular conditional spline; a conditional radial generalized Gaussian; and a shared-energy innovation model. The radial family has
`p(r|s,beta) proportional to exp{-(r/s)^beta}` with `(R/s)^beta ~ Gamma(q/beta,1)`; at `beta=2`, each spherical coordinate has variance `s^2/2`. A dense shape matrix requires positive-definite/gauge constraints and is not a cheap local fiber merely because its determinant is tractable.

The radial family is a useful closure estimator, not by itself a novelty claim. Natural block sizes are 9 for RGB 2D Haar details and 21 for RGB separable 3D Haar details. Required failure diagnostics include radial probability-integral-transform calibration, angular misspecification, cross-block energy dependence, and temporal phase dependence.

## Data, leakage, and fairness adversary

- The immutable UCF subset archive has revision `b9984b8d2a95e4a1879e1b071e9433858d0bc24a`, size 171,386,880 bytes, and SHA-256 `e9fcc76af48d320be88c5265f2e0576ecd615956976f6ce4742fdf2b042b71eb`. Despite its `.tar.gz` suffix it is an uncompressed POSIX tar.
- It contains 405 AVI files in 10 classes: 300/30/75 videos and 195/25/30 `(class,gNN)` groups in train/validation/test, with zero cross-split group overlap.
- “Mixture-logistic with Gaussian scales” is incoherent. B1 will use paired uniform dequantization and a continuous conditional density under an orthonormal Haar chart.
- Every conditional comparator must receive the same learned location features, or all must share an explicitly registered zero-mean restriction. A factorized mixture is not an unrestricted same-information baseline; B1 needs a strong conditional coupling control.
- Every endpoint method must receive a common full `N`-dimensional Gaussian noise vector. If a method ignores coordinates, report it. A compressed deterministic VAE denied fiber noise is not a fair comparator. Full representation dimension and repeated-state dimension must be reported separately.
- Resample images and `(class,gNN)` video groups as independent data units. Seeds measure algorithmic variability, not independent datasets. UCF101-subset is only a conditional-density pilot.
- Loaders must be manifest-only and enforce a confirmation-phase/config-hash guard before test decoding. Metadata visibility is already recorded; the statistical promise is no test pixel decode or model selection.

## Decisions

1. Stage A is retained as a procedural conditional-fiber confirmation and no longer described as a complete VAE, full-flow, memory, or production-latency result.
2. The next mechanism is **proposal: certified adaptive transport-depth routing**. Blocks are routed to an iterative conditional sampler only when held-out closure-regret bounds do not certify a one-shot kernel. This may yield a relative log-score/compute certificate; it cannot certify FID/FVD or universal dominance.
3. Observed B1 is rewritten around coherent continuous likelihoods, matched location/noise information, source- or group-level uncertainty, and a strong same-information conditional control. No observed model is fitted until the routing theorem and protocol receive independent audit.
