# RMPF-R6: low-rank projection rational-quadratic spline endpoint flow

This append-only milestone records a linked Blei-loop round after the RMPF-R5 residual-covariance failure. It changes exactly one scientific layer: the affine-Gaussian RCC endpoint is replaced by orthogonal low-rank global projections followed by monotone rational-quadratic splines with identity tails. The unchanged R4 `B+S` map and the spline endpoint compose into one exact normalized, all-coordinate, no-VAE flow.

## Exact map

For orthonormal `U` and coordinatewise monotone spline `R`,

```
T_U(b)      = b + U (R(U^T b) - U^T b)
T_U^{-1}(v) = v + U (R^{-1}(U^T v) - U^T v)
log|det DT_U(b)| = sum_j log R'_j(u_j^T b).
```

Rank zero is the exact identity and the copied-mechanism control ties exactly.

## Frozen result

The in-family non-Gaussian known-truth gate passed: NLL gain `0.01109532 [0.00999211,0.01219853]`, energy gain `0.03786306 [0.02872203,0.04700408]`, maximum round-trip error `8.88e-16`, and maximum forward/inverse logdet discrepancy `4.00e-15`.

On a fresh opened slice of 1,000 CIFAR-10 development images and 31 independent UCF source videos, every parent seed 9200--9204 selected rank zero. CIFAR positive ranks gained about `0.0057895` nat/dimension in exact NLL but had a joint proper-quality index above one. UCF had no projected-marginal headroom. Independent seeds 9400--9404 again selected rank zero. Forced positive image ranks produced NLL gain `0.00110816 [0.00030184,0.00191448]` but energy gain `0.00000455 [-0.00005825,0.00006735]`, clipping `0.919748`, and variance ratio `3.83087`.

Linked one-layer diagnoses also failed: coarse-conditioned splines selected rank zero; a first-order autoregressive projection chain failed its known-truth margin; exact atanh and rational support lifts removed clipping but not variance distortion or rank-zero selection; a frozen scalar support-temperature search also selected rank zero and left image/video variance ratios `3.9731` and `4.4347`.

## Decision

Not promoted. Confirmation stayed sealed. The first failure is the mismatch between projected marginal density fit and joint proper sample quality; the systems failure is independently unavoidable for an endpoint-only layer because the unchanged `B+S` realization is already slower and larger than the strongest full-flow route. Reopening requires a globally non-Gaussian copula transport jointly designed with a faster upstream exact flow, not another rank/bin/tail or support-temperature adjustment.

This directory does not modify or delete any prior QALT, FiberLift, FIQ-FM, MCQF, RMPF, RMGL, HSRF, data, split, result, or failure artifact.