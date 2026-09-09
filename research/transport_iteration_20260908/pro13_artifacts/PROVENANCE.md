# Inspection, execution, and preservation record

## Repository inspection (read-only)

Pinned revision: `51ba7e03aad3ee5c0493fba6ba05d976b7a95b2e`, resolved from user-supplied `51ba7e0` in `yixinwang/diffusion-representation`. No later branch head was assumed. All GitHub calls were reads. No branch, issue, PR, workflow, fit, or job was created through the connector.

Actually inspected:

* Commit metadata and its status-note diff, including the statement that Pro11 empirical artifacts were not delivered/verified at this revision.
* `research/transport_iteration_20260908/README.md`, blob `504999dd58aec7a1c45fd4d7bf2dc6f1a1f9fdf1`.
* `qalt/src/qalt/positive_spline_flow.py`, full content, blob `66eae5f17b8aa57f8e0d165c4a8a520d2b86a26e`.
* `qalt/src/qalt/positive_density_spline.py`, lines 140–260, blob `a1389491bc51ad1af50665fbc47fc1b63690d590`. This already uses the same rationalized quadratic conditional inverse.
* Repository/research directory metadata to locate relevant files. No native array, checkpoint, or native outcome file was fetched.

The commit metadata happened to include the new Heun-certificate change list. That frontier was not recomputed or reviewed here. This work is independent of Pro12. Pro11's reported cosine bound/fits/changed-law failures remain supplied context, not verified results of this package. Job 45576312's status was not polled and no native quality result is claimed.

## Primary background consulted

Müller et al., *Neural Importance Sampling*, arXiv:1808.03856 (piecewise-polynomial couplings); Durkan et al., *Neural Spline Flows*, arXiv:1906.04032 (analytic invertibility); Lipman et al., *Flow Matching for Generative Modeling*, arXiv:2210.02747 (simulation-free field regression); official SciPy `ndtri_exp` documentation (inverse log-normal-CDF tail interface). These are background, not sources for an empirical superiority claim or the numerical constants derived here.

## Local work

All generated observations are newly fabricated Pro13 development arrays in memory. The only local file reads in the numerical runners are their own source, saved synthetic fixture configurations, saved Pro13 states/results, dependency metadata and hardware metadata. No native or repository data loader is imported.

The six cases are three positive-law fixtures and three properly centered-zero-mean fixtures. They are NOT Pro11 source banks. Each case has 4000 observed arrays of dimension 3072. Gaussian-source and observed-array hashes are retained. Large in-memory sources/observations were not saved as package files; deterministic generation code and exact parameters/seeds are supplied. Later local timing checks reconstructed these *new Pro13* arrays and verified exact hash equality before use. This is not regeneration or replacement of any pending original artifact.

The twelve first-stage fitted-state files (six Python-packed and six dense-moment) are retained unchanged. Six subsequent compiled-kernel fitted states use the same observed arrays and give exactly the same parameters. Actual copied-state sample/log-density parity checks pass. The independent analytic MLE and global-FM arms in the prospective protocol were NOT fitted here. The delivered code is the candidate/reference/check package, not a complete PSC/FM training driver.

## Preserved failures and qualifications

1. The first Python packed-sign graph implementation was slower than the full optimized-BLAS Gram in all six single-fit graph timings. Its function, original source snapshot, raw times, states and results are retained. Subsequent compiled popcount implements the same statistic, rather than replacing the failed measurements.
2. The first compiled benchmark stopped on a serialization comparison: freshly built edge tuples were compared to JSON edge lists. Integer kernel parity had passed. `benchmark_compiled_initial.py.txt`, its stdout/stderr traceback, and its build record are retained. Canonical JSON comparison fixed this check only; no estimator, threshold, source, theta coefficient, graph, or observed array changed.
3. All four groups fall back in each of the three centered-zero-mean laws. Their 2.33–2.45 full-array KL values remain reported. No context-weighted rescue was fitted.
4. The proposed global-C1 interpretation is false. The continuous response CDF is C1, but parent-feature knots and fitted context steps prevent a global C1 transport. The tests record this counterexample rather than smoothing it away.
5. Normal-CDF saturation is demonstrated at z=9. The full unit-cube decoder rejects endpoints. The tail-safe Gaussianized scalar routine is not a claimed complete full-D tail implementation.
6. The original `math_checks.json` field named `unit_interval_error_bound_from_residual` is a nominal floating residual/.325 diagnostic, not an interval certificate. `precision_addendum.json` independently integrates the polynomials at 80 digits and uses the rounded alpha's actual minimum density. These are numerical checks, not outward-rounded certification or floating-output KL claims.
7. Build/first-load time is separately recorded and excluded from warm kernel timings. Add it for the first cold fit. Both the initial failed build and successful benchmark build records remain. The separate SYRK benchmark's one-time compilation setup was not timed in its graph rows; no all-inclusive total execution-time claim is made for that research benchmark. The original internal fit timers also omit the first whole-array validation scan; `outer_fit_benchmark.json` separately times the full function call including that scan and preserves all earlier timers. Hardware is a local virtualized/container environment; there is no PSC/GPU timing or peak-memory measurement here.

## Deliverable boundary

The payload consists of source, mathematical notes/protocol, fitted-state/configuration JSON, raw numerical/timing records, stdout/stderr and a SHA256 manifest. Architecture-specific compiled shared objects are temporary runtime outputs, not payload files; successful-build hashes/sizes/compiler commands are recorded. Rebuild from the supplied C source. Neither native data nor original Pro11/Pro12 binary files are included or asserted to be included.

Run `reproduce.py --output NEW_DIRECTORY` in an environment meeting the recorded dependencies and with a C compiler; it refuses an existing destination and preserves this delivery's original results. Reproduced timing values and possibly floating last bits are platform-dependent. Original payload hashes refer only to the delivered files, never to future regenerated files.
