# RMPF-R8: single-pass sparse reversible lifting trunk

This append-only milestone preserves every prior QALT, FiberLift, MCQF, RMPF/RMGL/HSRF, R5, R6, and R7 result. R8 changes only the systems-dominant recurrent B+S trunk while retaining the exact R7 coupled global-copula endpoint.

## Scientific status

`R8_COST_ORACLE_FAIL_DEVELOPMENT_NOT_OPENED_CONFIRMATION_SEALED`

The exact all-coordinate no-VAE flow uses a two-level orthonormal Haar hierarchy, class-conditional coarse normalization, triangular level-2 then level-1 lifting, a rank-r deterministic global mean, one sparse long-range Hadamard lifting stage, and the unchanged R7 endpoint. It has an explicit inverse and exact log-Jacobian.

Four executed realizations reduced image/video batch latency, fit memory, stored bytes, and operation count. The canonical C3 implementation passed every full-flow systems gate and all positive-rate latent-flow gates except image fit time. C3 required 1.64177 seconds and its exact reproduction 1.69543 seconds; the frozen positive-rate latent-flow limit was 1.44778 seconds. Video batch latency was threshold-sensitive across identical realizations. The preregistered cost prerequisite therefore blocked CIFAR/UCF quality evaluation. Confirmation remains sealed.

## Key exact checks

- maximum complete round-trip error: 1.14e-13;
- maximum log-Jacobian cancellation error: 0 in recorded arithmetic;
- retained collapsed affine R7/B+S compatibility: 7.99e-15 image and 1.95e-14 video;
- no dimension deleted, no VAE, no decoder, no ELBO, and no route selector.

The endpoint's earlier hidden-copula known-truth result is preserved: coordinatewise R6 minus coupled R7 NLL was 0.092466 [0.092273, 0.092659], with an exact copied-mechanism tie.

This branch records a negative systems result, not a real-data quality claim.