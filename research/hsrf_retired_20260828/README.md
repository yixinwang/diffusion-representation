# Hierarchical-Sketch Residual Flow (HSRF): audited negative diagnostic

Date: 2026-08-28

This append-only milestone preserves every earlier QALT, FiberLift, FIQ-FM, MCQF-v1/v2, RMPF, and RMGL result and failure. It does not modify existing code, data, splits, confirmation roles, or artifacts.

HSRF is one exact normalized all-coordinate flow. It uses an orthonormal Haar organization, exact additive or diagonal-affine triangular couplings, a parity-capable hierarchical interaction sketch, and closed-form full- or reduced-rank regression. It has no VAE, encoder/decoder pair, stochastic bottleneck, discarded dimensions, reconstruction loss, or variational KL.

## Scientific decision

**Not promoted; family retired; genuine confirmation unopened.**

Known-truth development showed that a hierarchical global sketch can recover a hidden global diagnostic, but exact likelihood and joint systems gates failed:

- R1 full output: local-minus-candidate NLL `0.00139`, below the frozen `0.02` margin.
- R2 reduced output rank: zero-minus-candidate NLL `0.00299`, local-minus-candidate `0.00141`; positive-rate PPCA remained better by about `0.00488` nat/dimension.
- R3 conditional scale: affine-minus-mean NLL was approximately `-2.15e-6`; a rank/cap sweep found no held-out conditional-scale headroom.
- Exact-copy controls tied in every round.

Diagnostic-only opened development used fresh CIFAR-10 train/validation/fiber/development records and source-separated UCF clips. Confirmation files were not extracted. HSRF reduced stored model bytes but did not jointly improve exact NLL, feature geometry, precision/recall, training time, sampling latency, peak RSS, and positive-rate-control quality. Video candidate recall was zero. An extended convergence audit improved the charted full controls and preserved the negative conclusion.

The first failed layer is the joint local/global conditional distribution family and stage alignment, not invertibility, normalization, global communication, full-output variance, or conditional-scale implementation. A legitimate reopening requires a new exact stage-aligned conditional transport and fused reversible training path, not another post-hoc rank or threshold change.

## Included files

- `DERIVATION.md`: exact likelihood, finite-sample advantage, reduced-rank correction, and impossibility boundary.
- `KNOWN_TRUTH_SUMMARY.json`: machine-readable R1/R2/R3 decisions.
- `REALISTIC_DIAGNOSTIC.csv`: compact CIFAR/UCF development rows.
- `FAILURE_LEDGER.md`: reproduced failures and stop rule.
- `reproduce_known_truth.py`: self-contained deterministic algebra/finite-regression checks.

The complete local scientific branch, raw per-seed rows, generated arrays, release archive, and Git bundle are preserved outside this compact review branch.