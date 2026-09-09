# Inspected primary sources and provenance

## Immutable repository reads

Repository: `yixinwang/diffusion-representation`.
All GitHub operations in this review were GET reads. No repository was edited.
The active branch resolved during inspection to
`4ec7341be5b1d8a84700354ac492b854f12471eb`; this identifies the read snapshot,
not a claim that the branch cannot subsequently advance.

At the explicitly requested Pro10 commit
`22663881a72ab1ca0f0f92f55f8f7e7f9d51e52c`, inspected:

- `research/transport_iteration_20260908/pro8_artifacts/README.md`:
  exact non-Gaussian class, finite-learning/affinity argument, population field,
  global derivative bound, median-event lower bounds and scope limitations.
  Git blob `40e6770f6537cb41e811335390391a7a88346e22`.
- `research/transport_iteration_20260908/pro8_artifacts/positive_class_reference.py`:
  concrete source law, 3072/192/2880 dimensions, public dictionary construction,
  root/head fitting routines, quantile transport, field, Heun and common chart.
  Git blob `f54fb3c8e06953a22337e3f45296946378bc52e3`.
  Inspected, not imported/executed; no fit was run.
- `research/transport_iteration_20260908/pro8_artifacts/interval_tilt_check.py`:
  outward binary64 median-trajectory certificate with explicit Taylor remainder.
  Git blob `9c0069ab510d702662e6e39a52668b9d3934ab49`.
  This is a median lower-bound calculation, not compact full-KL integration.
- `research/transport_iteration_20260908/pro10_artifacts/pro10_kernels.py`:
  endpoint-specialized real Heun formula, discrete analytic derivative and
  central/tail quantile identities. Its local threshold is 8, not the PSC 5.
- `research/transport_iteration_20260908/pro10_artifacts/quality_quadrature.py`:
  forward-KL identity, nonnegative cancellation-aware integrand, ordinary
  256/512/1024-resolution integration and symbolic Gaussian tail expression.
  Its resolution agreement is explicitly not a certificate.
- `research/transport_iteration_20260908/pro10_artifacts/MATH_AUDIT.md`:
  correct-versus-wrong learning events, old risk correction, ideal-law and
  floating-output distinction. The new proof sharpens rather than changes it.

URL prefix for these exact reads:
https://github.com/yixinwang/diffusion-representation/blob/22663881a72ab1ca0f0f92f55f8f7e7f9d51e52c/

At the frozen cost commit
`9391d3ab89ad0667036d39a85f2e59409da34dc6`, inspected the commit diff containing
`qalt/experiments/tilted_sampler_cost/PROTOCOL.md` and `candidate.py`.
The protocol fixes threshold 5, 17 primary arms, three source seeds, batches
1/64, primary in-call time-cache construction, endpoint N-2 accounting and the
secondary ALL-call amortized schedule. No frozen candidate was modified.

At the result snapshot
`4ec7341be5b1d8a84700354ac492b854f12471eb`, inspected:

- `research/transport_iteration_20260908/psc_tilted_cost/summary.json`:
  all six cases and 17 primary medians, counts 3060/900.
  Git blob `3964bdb341e4d855a5e84f5e1e9c8c5548398608`.
- `research/transport_iteration_20260908/tilted_cost_machinecheck.json`:
  prior audit's 129 payload hashes, 102 outputs, 78 gates, parity/roundtrip
  errors, source reconstruction, schedule-family distinction, one archived fit.
  Git blob `6728721088bd52d7a9209a4dbdc5fede6749de96`.
- The commit diff for `check_tilted_cost_results.py`, showing how that prior
  audit checks payloads, source closure, outputs, orders and medians.
- The commit's recorded payload manifest, containing the summary hash below.

Result URL prefix:
https://github.com/yixinwang/diffusion-representation/blob/4ec7341be5b1d8a84700354ac492b854f12471eb/

The exact summary bytes are delivered as `sources/psc_summary.json`:

    bytes: 4137
    SHA256: e783b9d666f84a1cba1ed5e615410c8718649c2f8f286aaad115a5e18d9306ea

This local hash equals the archived payload manifest's value. The source bank
NPZ omitted from the published Pro10 text subset was neither acquired nor
recreated. None of the original PSC binary output arrays are delivered here.
No claim is made that this package is the original Pro10 full package.

## Numerical primary documentation

GNU MPFR 4.2.2 official manual, rounding semantics and exp/erf/log/sqrt interfaces:
https://www.mpfr.org/mpfr-current/mpfr.html

The runtime queried by the executable is MPFR 4.2.2. Directed rounding semantics
are relied on; no inferred last-ULP accuracy of SciPy/libm special functions is
used. `results/environment.json` records compiler/platform/library linkage.
No third-party library binary or font is redistributed.

## New versus historical evidence

New: the C++ interval engine, all five compact integral certificates, directed
tail/global-constant checks, sharper expected-risk proof/postprocessing, 256
consistency checks, an order/precision/mesh-changed N4 certificate, and the
paired-ratio/target-eligibility analysis of the archived medians.

Historical inspected evidence, not newly replayed: finite-catalog learning
source, prior median interval results, PSC 129-payload/102-output/78-gate audit,
3060+900 timings and reported traced-memory diagnostics. No new timing samples,
independent fitted models, native quality scores or training results are claimed.
