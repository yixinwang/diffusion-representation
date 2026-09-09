# Frozen CUDA spline benchmark audit

Job 45570945 completed exit 0 on v024 in 4:50, at frozen revision bb024437a761c096375a042239f77262724dca14. Script elapsed 283.7821 seconds; Slurm batch peak RSS 1,127,868 KiB (~1.076GiB). Hardware was Tesla V100-SXM2-32GB, driver 580.178.04, CUDA 12.8, Torch 2.10.0+cu128, Triton 3.6.0, GCC 8.5.0. This is a synthetic checked-call kernel benchmark, not a learned model or image/video result.

Audit PASS: all 8 published experiment payload hashes match; all 7 recorded source hashes match Git blobs at the frozen revision. The top-level closure is exact. Compiler caches/tmp were excluded from transfer and remain on PSC; they are excluded from the experiment manifest by protocol. The empty job log and full small experiment payload (including fabricated arrays) are preserved locally in `work/psc-spline-kernel`. The audit script is `work/audit-spline-kernel-results.py`, machine results are `work/psc-spline-kernel/machinecheck.json`.

All 78 numerical checks passed before timing. Largest recorded absolute discrepancies: full float32 values/logdets/roundtrips 9.11951e-6, small float32 2.14577e-6, small float64 4.24660e-15, gradients 2.02656e-6. Full/small float32 tolerance was 1e-4; float64 tolerance 1e-10; gradient allowance 1e-4+1e-3|reference|. Even the largest recorded gradient absolute error is below the absolute allowance. The provenance and executable gates are verified; GPU output and gradient arrays were not preserved for successful comparisons, so this audit does not claim independent GPU numerical recomputation.

All 180 timings are present: 30 per each of six arm×direction combinations. Their order exactly reproduces the frozen Python seed and permutation procedure. Every wall/event duration is positive; memory accounting identities hold. All published medians and peak summaries independently recompute exactly from the raw records. Ten per-variant warmups and per-call validity checks are established by the frozen source; they were not separately logged as timing samples.

| Implementation | Forward wall median (ms) | Inverse wall median (ms) | Forward CUDA event median (ms) | Inverse CUDA event median (ms) | Incremental peak forward/inverse (MB, decimal) |
|---|---:|---:|---:|---:|---:|
| Reference RQS |8.0286|8.4736|7.9956|8.4392|71.913 /78.719|
| Dense eager |7.4384|7.5673|7.4062|7.5344|66.356 /72.807|
| Dense compiled |0.4846|0.4846|0.4479|0.4519|34.480 /36.692|

Reference/compiled median wall ratios are 16.57 forward and 17.49 inverse; reference/eager ratios are only 1.08 and 1.12. Compiled incremental peak allocation is 47.95%/46.61% of reference in the two directions. These are observed ratios on one fixed (64,2880), 8-bin float32 synthetic workload. Event measurements include stream gaps induced by validation/host dispatch; they are not pure device-instruction timings. Every dense call charges its single aggregated validity check, and reference retains its existing checks.

The sum of compiled first-use wall measurements is 239.3512 seconds across 8 first-use cases, including float64 and gradient compilation. It includes execution/validation and is neither a pure compiler duration nor a production-only setup bill. Absolute GPU peak allocations include resident shared input/compiler state. Backward runtime, context networks, model composition, dynamic shapes, recompilation, learned coefficient distributions, numerical tails, amortization and full-model quality/cost remain unmeasured. Repeating fixed inputs 30 times is not independent algorithm-training evidence or a significance test.

The result supports a separately frozen checkpoint-level integration/equivalence and end-to-end cost test with checks retained. It does not authorize changing model defaults or imply a proven algorithmic complexity improvement. No generative data, official test data, native image bank or other user job was accessed in this audit.

The accompanying single status query reported perceptual job 45571074 still PENDING with no assigned node. That job was not modified or evaluated here.

A separate seed-reconstruction check on local Torch2.14.0 matched only1of12array byte hashes against PSC Torch2.10.0: largest float32 element difference1.94e-6; largest float64 difference8.33e-17. Thus seed-only cross-platform byte reproduction is not asserted. The exact preserved fabricated-input archive, whose manifest hash verifies, is the authoritative input for replay. No altered input was used to recompute a benchmark or choose a result.
