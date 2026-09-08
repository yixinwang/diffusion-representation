# Independent PSC input-readiness audit

Job 45558402; frozen source `2b58285826a74e32882a6de0af2a1d951a85ba58`.
The saved run status is `input_readiness_checks_passed`. Audit verdict: the
saved source-bound integrity and numerical-readiness results are consistent.
They are not training or generative performance results.

## Integrity and selected records

All six completion-payload hashes, all six source-file hashes against the
frozen Git commit, and the internal loader-ledger hash verified. Selected IDs
are sorted, unique, in 0..49999, disjoint between fitting and repair, and match
both saved raw-little-endian-int64 hashes. Counts are exactly 4,000/1,000, with
native 3x32x32 shape and 3,072 declared coordinates. The loader reports canonical
verification, fixture bypass disabled, and precisely the five expected training
file names/paths/hashes. No official test path appears in its opened-file ledger.

The full 40,000-fit/5,000-repair/5,000-excluded-discovery split and fit/repair
membership hashes agree with the frozen identifiers. Source inspection confirms
those hashes are checked before hash-ranked selection and float64 dequantization.
The repaired development history is preserved; these records are not newly
untouched evaluation data.

For an independent membership check without reading pixels, I accessed only
`record_ids` and `labels` from the previously published radial-development
`scores.npz`. None of its score arrays was loaded. Its complete 5,000 repair-ID
hash matches the frozen repair hash. The selected 1,000 repair IDs are members
of that exact set, contain exactly 100/class, and match an independently
recomputed SHA-256 ranking with the frozen salt. None of the selected 4,000
fitting IDs belongs to the complete parent repair set.

The readiness payload does not contain complete fitting/discovery ID lists or
labels for the selected fitting records. Therefore fitting 400/class membership
and exclusion of discovery are verified through the frozen loader's control
flow and its bound split ledger, not independently reconstructed from raw
membership metadata in this audit. This is an audit scope limitation, not an
observed split failure. No additional pixel read was performed to conceal it.

## Fixed-chart numerical checks

| Check | Fitting 4,000 | Seen repair 1,000 | Frozen tolerance |
|---|---:|---:|---:|
| Float64 logit/sigmoid roundtrip max |1.1102230e-16|1.1102230e-16|1e-12|
| Float32-logit/float32-sigmoid roundtrip max |9.1980382e-8|9.1398306e-8|1e-6|
| Float32 logit cast max error |5.9102266e-7|6.3148769e-7|diagnostic only|
| Float64 Jacobian cancellation max |1.8189894e-12|1.8189894e-12|finite diagnostic only|
| Float32 sigmoid rounded boundary count |3|0|recorded, not clipped|

Both registered roundtrip gates pass. All saved finite/interior-input flags
pass. The Jacobian cancellation value is a separate aggregate diagnostic;
its size above 1e-12 is not failure of the pointwise float64 roundtrip gate.
The three fitting-coordinate float32 sigmoid endpoints reinforce why raw
unit-cube values cannot be cast to float32 and fed into a strict logit model.
The frozen pipeline instead computes logit in float64, casts those real-valued
logits for all methods, and retains the float64 chart Jacobian. Rounded sigmoid
outputs are not recycled as strict-unit-cube inputs or silently clipped.

These comparisons audit the saved numerical aggregates, not a new numerical
rerun over pixels. Transformed-array hashes are retained but their source
arrays were intentionally not saved; this audit does not recompute them.

## Resource and interpretation limits

The runner reports 6.784 seconds and 380.20 MiB peak RSS, below its 270-second
soft limit and 2,000-MB allocation. Provenance and summary consistently mark
training, generation, quality statistics and test access false. The separate
Pro5 proposed 20k/20k split was not executed. Payloads contain selected IDs,
ledger, numerical aggregates and source/resource provenance; no image content,
model checkpoint, sample gallery, likelihood or quality score is present.

Machine checks: `check_observed_input_results.py --output NEW_JSON_PATH` and
`psc_observed_inputs_machinecheck.json`. This audit neither contacted PSC
nor touched GPU jobs, changed repository files, or submitted a job.
