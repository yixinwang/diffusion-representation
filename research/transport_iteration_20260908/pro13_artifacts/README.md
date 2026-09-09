# Pro13 artifact package

**Decision:** the proposed declared law and stable inverse are valid; it is not a globally C1 learned flow. One concrete lower-cost graph learner is exact packed-sign correlation. Its compiled implementation has a nonvacuous expected-risk theorem and a local speed gain; the Python implementation loses, zero-mean conditional dependence still fails, and the analytic copy ties. No native, PSC, or global-FM result is claimed.

## Mathematical results

The complete assumptions, derivations, regularity correction, precision limits, masked bounds, compute counts and zero-mean falsifier are in `THEORY.md`. The prospective same-information/compute experiment is in `PROTOCOL.md`.

| Quantity | Value |
|---|---:|
| Pair-density minimum | 0.325 |
| B_kappa | 0.918484500574 |
| Packed graph-error probability upper bound | 3.39486941007e-05 |
| Expected full-array KL upper bound | 0.561059832241 nats |
| Dense-statistic bound, same new law | 0.486488738194 nats |
| Restricted fixed-chart product floor | 88.2 nats |
| Masked psi/A excess squared-loss upper bound | 0.00010072526559 |

These are exact-arithmetic statistical results with rounded numerical evaluations. They are not per-fit guarantees, native claims, or comparisons against an unrestricted analytic/flow model. The law changed from cosine; do not compare its bound to Pro11's as a same-law improvement.

## New fabricated development evidence

| Law | Seed index | Correct pairs | Root KL | Conditional KL | Joint KL |
|---|---:|---:|---:|---:|---:|
| positive | 0 | 1440/1440 | 0.159748409 | 0.033866759 | 0.193615168 |
| centered_zero_mean | 0 | 0/1440 | 0.159748409 | 2.235039582 | 2.394787991 |
| positive | 1 | 1440/1440 | 0.157620782 | 0.046688021 | 0.204308803 |
| centered_zero_mean | 1 | 0/1440 | 0.157620782 | 2.173452339 | 2.331073121 |
| positive | 2 | 1440/1440 | 0.174734165 | 0.046897088 | 0.221631253 |
| centered_zero_mean | 2 | 0/1440 | 0.174734165 | 2.274597311 | 2.449331476 |

Every positive case recovered all pairs, and every stress case fell back in all groups. The stress sine is centered under the actual nonuniform root histogram. These are NEW Pro13 observations, not verification of Pro11's pending empirical results. The numerical KL integration is cross-checked, not interval-certified.

Python-packed graph discovery lost to BLAS in all six initial graph timings. The compiled identical-statistic graph kernel was 5.67–6.21x faster than full Gram in subsequent interleaved one-thread local checks. The separate stronger SYRK/1-or-4-thread comparison gave 4.54–7.21x ratios, still local/container measurements. Feature construction, packing and graph-degree checks are included. All raw times are retained. First compile/load was 0.058344 seconds and is separate from warm timings. Positive compiled internal fit timers took approximately 0.147–0.169 seconds before that one-time compile; these internal timers omit the initial whole-array validation scan. `outer_fit_benchmark.json` supplies separately timed full outer calls including that scan. Batch64 full unit-cube decoding took approximately 13.4–14.1 ms. The later outer-call benchmark, including initial validation, measured 0.167–0.175 seconds for compiled fits versus 0.312–0.335 seconds for dense fits (1.82–2.01x), with a separately measured 0.067-second compile/load. These are not a global matched-quality frontier or PSC results.

## Numerical checks

There are 100,057 inverse cases in each of FP64/FP32, including knot neighbors, an independent 80-digit boundary audit, exact integer bit-count parity checks, 323 KL-sandwich/series cross-checks, full D3072 fitted-state round trips, a small independent finite-difference Jacobian, and 425 Gaussianized tail cases. FP64 direct CDF residual was 2.23e-16 or less; FP32 recomputed residual was 1.04e-7 or less. The largest full-D fitted-state round-trip error was 1.64e-12. No global floating certificate is claimed. See precision qualifications in `PROVENANCE.md`.

## Source and reproduction

Call `pro13.fit(observed, method="packed_c")` for the proposed compiled variant; `method="packed"` preserves the slower Python reference. `pro13.py` implements the law, inverse, Gaussianized scalar-tail map, full-D learned sampler/encoder/log-density, observed-only estimator, and theorem constants. `packed_graph.c` and `packed_kernel.py` implement the compiled graph. `standalone.py` creates only fabricated observations; its learner never receives fixture truth. Numerical and timing runners are separate. The initial failed compiled-wrapper comparison and the original Python timing losses remain in the package.

Install the versions in `requirements.txt` in an isolated environment; a C compiler is needed for the compiled kernel. Preserve the delivered results by running in a new destination:

```sh
python reproduce.py --output /path/to/a/new/pro13_reproduction
```

This package includes source and runnable candidate checks, not a completed PSC/FM launcher. All original results are under this directory and `standalone_results/`; `SHA256SUMS` lists payload hashes. Large fabricated Gaussian/observation arrays were generated in memory and not retained; their exact hashes, seeds, configurations and generation code are included. Compiled shared objects are rebuildable platform-specific runtime outputs, not payloads. No old artifact or native-data file is represented as delivered.

Inspected repository pin: `51ba7e03aad3ee5c0493fba6ba05d976b7a95b2e`. Actual inspected files and background sources are listed in `PROVENANCE.md`.
