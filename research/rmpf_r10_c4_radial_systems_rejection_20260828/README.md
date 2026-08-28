# RMPF-R10-C4 radial shared-scale systems rejection

This append-only milestone records the executed RMPF-R10-C4 Blei-loop child. It changes only the conditional radius law of the already color-whitened three-dimensional residual vectors inside the exact R10 FSMLF plus unchanged R7 global-copula flow.

## Scientific decision

`KNOWN_TRUTH_PASS | SYSTEMS_FAIL | REAL_QUALITY_NOT_OPENED | CONFIRMATION_SEALED`

The model is one exact normalized, all-coordinate, no-VAE flow. A monotone rational-quadratic log-radius spline acts on each class/level/orientation residual cell, has exact identity tails and a source-frozen identity fallback, and preserves direction. The exact inverse and log-Jacobian were checked numerically.

## Known-truth result

Five independent seeds 9700--9704 passed:

- identity minus C4 NLL: `0.138288 [0.135693, 0.140884]` nat/dimension;
- RCC affine minus C4 NLL: `0.102362 [0.100785, 0.103939]`;
- identity minus C4 proper energy: `0.012962 [0.011111, 0.014812]`;
- maximum round-trip error: `6.66e-16`;
- maximum log-Jacobian cancellation error: `2.55e-15`;
- finite-difference log-Jacobian error: `8.52e-11`.

The conditional radial model also beat unconditional, shuffled, random-label, generic one-projection, and coordinate-spline controls. The hidden-angular arm retained sign parity exactly and had radial NLL gain only `0.00015149`, confirming the first failure condition.

## Real-data systems gate

The frozen real smoke used seed 9400 and accessed only opened train/fiber-fit roles. It computed no development quality metric.

- CIFAR image: fit `2.87984 s`, batch `0.14880 s`, single `0.003136 s`, peak fit `46.92 MiB`, stored `36.19 KiB`, FLOP/full ratio `0.0823`.
- UCF video: fit `3.06913 s`, batch `0.11134 s`, single `0.007513 s`, peak fit `136.96 MiB`, stored `114.20 KiB`, FLOP/full ratio `0.0672`.

The complete systems gate failed in both domains. Image failed full/latent fit and batch limits and the stricter inherited child budget. Video failed latent fit, full/latent batch, latent single, and the stricter child budget. Memory, bytes, FLOPs, inverse, and exact normalization passed.

A component profile found median parent-plus-color fit time `1.502433 s` on images before radial fitting or R7 refitting, already above the frozen positive-rate latent allowance `1.447782 s`. Therefore another endpoint-only rank/bin/tail/threshold change is not a feasible small revision.

No C4 CIFAR/UCF quality result and no confirmation result were opened. All earlier QALT, FiberLift, FIQ-FM, MCQF, RMPF/RMGL/HSRF, R5--R10 evidence is preserved.
