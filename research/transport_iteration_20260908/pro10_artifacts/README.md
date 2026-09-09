# Pro10 standalone inspection package

Read target: yixinwang/diffusion-representation,
agent/observation-transport-audit-20260908,
commit e7ee7938a9b146aa5b167a1176e53f99f9ccd4bd.
No repository modification or PSC job submission occurred.

Main result: the finite-learning plus interval expected-risk separation survives
its stated restricted comparison. A tail-safe, algebraically identical central
quantile branch is supplied, together with endpoint/time-cached uniform Heun.
In the COMPLETED local focused four-call check, the hybrid exact sampler remains
slower than cached Heun4 in every one of three seeds and both batch sizes.
These are single-process local exploratory results, not PSC confirmation.

New full-KL floating quadrature gives approximately .223725, .0320629, .00214650,
.000137748, .00000870961 nats/full array for N=4,8,16,32,64. These are NOT
interval-certified values. The next computation is compact-domain validated
integration, with the analytic tail bound and wrong-selection risk accounting
provided in MATH_AUDIT.md. Lower bounds alone do not establish target feasibility.

## Files

- pro10_kernels.py: hybrid forward/inverse; legacy formulas; optimized equivalent
  Heun; evaluation-only discrete Jacobian.
- run_local.py and local_numerical.json: numerical tests, independent 90-digit
  inversion checks, all five Heun parity checks, and full-D roundtrips.
- quality_quadrature.py and local_quality.json: three resolutions of explicitly
  non-certified forward-KL quadrature and the symbolic tail-bound evaluation.
- run_focused_local.py, local_focused_timing.json, local_focused_live.jsonl:
  completed four-call timings, all 216 raw measurements, all six condition cells.
- local_focused_sources.npz: exact Gaussian input and dictionary bytes for those
  local timing cells, regenerated and hash-verified on the same environment.
- local_timing_live.jsonl: 161 partial rows from the timed-out full-grid attempt.
  It is not a completed all-NFE frontier. RUN_NOTES.md preserves both timeouts
  and the earlier corrected test-caller broadcasting mistake.
- audit_numbers.json, MATH_AUDIT.md, PROPOSED_PSC_PROTOCOL.md: calculations,
  rigorous symbolic arguments, scopes, missing certificate, and prospective plan.
- PROVENANCE.json and SHA256SUMS: inspected sources and this package's hashes.

## Local reproduction

Requires Python, NumPy, SciPy and mpmath. No network or scheduler calls.
Run in a new result directory with these scripts copied in:

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python run_local.py --mode check
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python quality_quadrature.py
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python run_focused_local.py

The full-grid timing harness is run_local.py --mode bench; allocate sufficient
local time and retain its live JSONL. Its previous local attempts did not finish.
These scripts are diagnostic reference code, not a production frozen PSC launcher.
They write result files in the current directory; do not run over preserved files.

This package contains no fitted native model, training observations, official
or protected native images, PSC results copied as local outputs, or weights.
The preserved Gaussian banks are newly generated standalone mathematical inputs.
