# Local execution notes

The first smoke-test caller supplied a (10001,1) state and (1,6) coefficient to
an in-place Heun update. The test caller was corrected to broadcast its input
state first. Neither Heun method was altered by this correction.

The first complete local timing invocation was terminated by the 45-second
execution timeout before its end-of-run JSON was written. No timing outcome from
that invocation was available for selection or used in a claim. The harness was
amended to append each raw timing immediately outside the timed region and was
then restarted with a sufficient execution limit. No quantile threshold, method,
source law, seed, warmup, repeat count, or NFE grid was changed.

These are local exploratory checks in a standalone implementation, not a replay
of the entire GitHub source or the PSC fitting/observation artifacts. The local
benchmark has no fit stage, no method-specific memory measurement, and only one
completed process. It is not the proposed matched-quality PSC experiment.

Correction to the planned restart described above: the full-grid restart also
hit its execution limit (150 seconds). It preserved 161 raw rows in
`local_timing_live.jsonl`; it did NOT complete. There is no completed local
full-frontier timing JSON, and no complete-full-grid claim is made.

A separately labelled focused four-call diagnostic then completed: three seeds,
both batches, four arms, three warmups and nine random-order repetitions per
arm. All 216 raw rows and all six complete condition summaries are retained.
The hybrid exact sampler lost to endpoint-cached Heun4 in all six conditions.
The full-grid incomplete rows were not substituted into this diagnostic.

`local_focused_sources.npz` contains the exact local dictionary and all six
focused source banks. Regeneration on this same environment was hash-checked
against every saved focused source hash before writing this archive. It is not
a claim of byte identity with any previous PSC or Pro8 observation banks.
