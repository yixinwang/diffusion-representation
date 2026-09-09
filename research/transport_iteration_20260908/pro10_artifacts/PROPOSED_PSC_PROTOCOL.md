# Proposed next PSC quality/cost protocol — NOT executed

This prospective protocol is informed by the local development checks. It is
not a retrospective preregistration of those checks. Freeze all implementing
bytes and tests before any new PSC measurements. No job was submitted here.

## Scope and source

Preserve e7ee793 and original Pro8 and PSC artifacts unchanged. Use the same
finite law, D=3072,c=192,m=2880, public dictionary, signs and index catalog,
32 root arrays and 256 disjoint head arrays. Fit the three original declared
seeds 2026090901/02/03; no replacements on wrong recovery or numerical failure.
Materialize and hash actual dictionary, Gaussian banks, observed arrays, fit
scores and fitted parameters once; reuse identical bytes for all arms. Do not
call a seed-only cross-platform reconstruction byte-identical. Use canonical
stored arrays for cross-machine verification. Keep the 32-condition recovery
study as existing evidence, not another empirical tiny-error-probability claim.
Keep official test/discovery/native data closed.

## Arms and numerical gates

Archive the unchanged two-branch negative and all-log/reference PSC negatives.
Time unchanged all-log exact and Heun N=4,8,16,32,64 as legacy arms.
Time hybrid exact and endpoint/time-cached Heun at every same N with the same
optimized root, summary cache, chart, dtype, allocations and validation policy.
Use the old-root/new-root cached-Heun ablation to attribute common root savings.
Retain a bitwise exact-copy decoder control and the Gaussian zero-field case.
Cache immutable coefficients and per-sample context, never evolving field state
or sources/results across newly generated samples. Record N-equivalent stages
and N-2 nontrivial kernels separately. Any additional buffer/compiler optimization
must be frozen for both families before results, with no fast-math assumptions.

Float64 primary gates: independent high-precision forward error <=2e-13 for
|z|<=50; inverse roundtrip <=1e-12; folded log-CDF identity error <=1e-11;
threshold-adjacent downward numerical jumps <=1e-13, not a claim of bitwise
monotonicity; optimized/reference Heun discrepancy <=2e-13 on the declared
source/adversarial bank; full-source roundtrip <=1e-9; no nonfinite accepted
output; exact-copy bitwise equality. Cover e=.4,.6, gamma=+/-.5, zero, signed
zero, nextafter neighbors around -8,0,8, dense e/z grids, Gaussian arrays and
mixed central/extreme-tail arrays. Reject invalid/unrepresentable inputs; never
clip or silently hide invalid tail branch results. Save raw outputs before gates.
Numeric tests do not prove implemented floating-output KL.

## Quality certificate and matching

Rerun the original interval median certificate, retaining all five enclosures.
Independently compute J for the discrete Heun map and certify the compact
forward-KL integral plus the explicit Gaussian tail bound in MATH_AUDIT.md.
Require full-array interval width <=2e-8. If unresolved, label the full-KL value
uncertified; do not substitute quadrature resolution agreement for enclosure.
Report correct-recovery K_N and unconditional expected-risk intervals separately.
Use the conservative failure upper bound in MATH_AUDIT.md for expected-risk
feasibility, or a separately frozen tighter proof, not omission of failures.

Freeze quality targets 1e-1,1e-2,1e-3,1e-4,1e-5,1e-6 nats/full array, while
showing the entire five-point frontier. At each target, the comparator is the
FASTEST arm whose certified expected-risk upper bound meets it. An interval
crossing the target is unresolved. If no Heun point meets a target, report
'no certified feasible comparator in the fixed grid', not an infinite speedup
or a matched-quality win against an ineligible point. Exact-copy ties remain.
Theoretical exact risk <=8.249606e-9 is an ideal continuous-law statement.

## Timing, setup, fitting and memory

CPU primary uses the same PSC CPU class/software/thread settings as the
completed adverse replication, with hardware and loaded library identities
recorded. No GPU/native compiled-kernel result is substituted for this CPU case.
Use batches 1 and 64. Per seed use three fresh process launches. Per process
perform exactly 10 warmups per arm and 30 balanced random-order timing blocks.
Retain individual values and ordering; hash and preserve partial rows on failure.
Save timing sources separately from fit streams. Record cold import/first-call,
plan construction, optional compilation, warm steady-state median/p95 and total
harness wall time. Pay all root/context/output/validation work in each arm.

Report (a) same-source input-to-complete-output latency and (b) fresh-source
end-to-end generation including RNG/allocation, using a matched RNG protocol.
No RNG or nonlinear context feature is free merely because a microbenchmark
starts after it. Record actual fitted-state/dictionary storage, temporary/cache
bytes and process resident-memory baseline/peak. Measure per-arm memory in
isolated processes; a shared process's lifetime max-RSS cannot identify each
method's peak. Record full fitting time and peak memory, including decoding the
observations, root inversion, candidate score scan and any cache preparation.
Fit once per seed and use exactly that fitted state for every sampler. Common
fit cost cancels in differences but remains in absolute deployment costs.
Source simulation, acquisition, verification and validation costs are separately
reported, never charged selectively to one inference arm.

For K requested outputs and batch b, report
T_a(K;b)=setup_a + fit_common + actual generation-batch sum,
with K in {1,64,4096,1000000}. Account explicitly for the final partial batch.
If setup_exact>setup_control and steady latency is smaller, break-even batches
are at least ceil((setup_exact-setup_control)/(tau_control-tau_exact)); include
any unequal fit cost too. If the denominator is <=0, there is no sampling-led
amortization crossover. This is not a zero-setup claim.

Keep all seed x batch x N results. Report paired timing contrasts within
process/block and independent process variability separately. Use simultaneous
intervals over the frozen comparisons/targets; timing repetitions are not extra
independent training seeds. A 10% speed-repair claim requires the upper paired
latency-ratio interval below .9 in every declared seed/batch cell against the
eligible fastest comparator, with setup/break-even and memory reported. Any
contrary cell remains a failure; no relabeling it as warmup or changing N after
results. With only three training seeds, do not imply broad population certainty.

## Stop rule

A repeat of the four-call latency loss remains an adverse cost result even if
the exact law is better. A strict-quality win can still exist at a higher
feasible N, but it must be reported with its target and setup horizon. Do not
promote this finite catalog into native image quality, representation-semantic,
all-FM, or all-latent-decoder superiority. The separate copula falsifier changes
the source and requires its own frozen experiment and learning analysis.
