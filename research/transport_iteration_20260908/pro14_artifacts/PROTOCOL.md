# Pro14 review protocol: one response-head factorial

Status: proposed for review, not launched or published to GitHub. Native source
integration, optimized baseline equivalence, and source closure are prerequisites,
not already completed work. No fit/repair/test arrays were accessed in this review.

## Decision and hypothesis

Keep rank16 frames, scalar heads, masks, analysis architecture, root architecture,
and input data fixed. Test whether inexpensive *own-innovation-dependent*
conditional means/variances improve native observable quality, beyond an
identical prefix-only head and beyond equally scheduled strong RQS. The proposed
architecture and its prefix-only control are specified completely in THEORY.md
and implemented as a standalone module in mechanism.py.

No rank/bin/root/width/anchor/mask/learning-rate sweep is authorized by this plan.
No synthetic witness is a substitute for the native gates.

## Six factorial cells plus one scalar reference

Rows: P = old block plus prefix-only response; I = old block plus innovation
response; R = strong four-layer global RQS, width32, bins8, heads4.
Columns: F = frozen A/root, 180 seconds residual fitting; J = 90 seconds residual
then 90 seconds joint A+residual, root parameters fixed with input gradients live.

There are six primary cells P_F, P_J, I_F, I_J, R_F, R_J, plus S42 as a whole-model
parameter-matched no-mixer scalar reference with frozen A/root. I and P each
have 539,120 whole-model parameters by current count algebra; S42 has 544,080;
R retains all 589,712. All counts must be verified on actual active modules.
The isolated old M/J results remain archived failures; new names do not overwrite
or retroactively pass the old pilot's 27 gates.

This is one 3-by-2 controlled experiment, not a decoder or optimizer sweep.
I-vs-P isolates the route from existing stochastic anchors. J-vs-F tests training
allocation within a decoder. R_J prevents attributing an advantage to a joint
schedule that was denied to RQS. A-only/root-only remain diagnostics, not full-
budget controls; no claim against them substitutes for R_J. There is no additional
full-budget A arm because joint RQS is included.

## Fixed data, optimizer, and charges

Use the same canonical 4,000 fit and reused 1,000 repair records, original
membership and dequantization, no official test. Seeds 77201/77202/77203 remain
three independent DEVELOPMENT fits, not new confirmation. All cells of all
seeds freeze before any repair evaluation. No repair-based selection of backend,
weights, hyperparameters, anchors, head shape, or runtime stop is permitted.

Use full 3,072-dimensional Gaussian priors and the same preserved per-seed
2,000-by-3,072 evaluation source bank. No regeneration replaces the original
source bank in a replication; verify its original hashes. A new bank would be
an explicitly separate development evaluation, not an unchanged replay.

Use the frozen analysis90/root90/residual180 standalone envelopes, Adam1e-3,
batch32, no clipping, extra optimizer tricks, retries or additional training
objectives. The response heads initialize at zero output. P and I start with
identical old-block weights and corresponding response-head parameter values.
All residual families use the same seeded fitting-index stream; different wall-
limited step counts do not imply identical final sample exposure.

For each decoder family independently, save an exact first90 residual fork with
optimizer and RNG. Its F and J tails fork from that actual state. J preserves
residual Adam state and adds A parameters with fresh Adam state. Root input
gradients remain live. Root weights are fixed. Joint steps recompute current
A/coarse/residual/prefix values: no stale fixed-A cache may enter joint losses.
Shared analysis/root prerequisites and fork preparation are charged in full to
each standalone arm, not amortized or divided by the number of descendants.

Whole likelihood is root NLL + residual NLL - analysis LD - outer logit LD.
Save all five per-example columns. Compare complete NLL across different A;
never rank different analyses by residual NLL alone. The new response determinant
must be included inside residual NLL. Auxiliary gradient/feature metrics remain
evaluation-only, not fit objectives.

## Optimized baseline and cost accounting

Use an actually optimized, globally conditioned RQS implementation and its joint
version, not a coordinate-loop substitute or an unoptimized baseline paired with
an optimized candidate. Preserve the exact four-layer architecture, all used
parameters, spline domains, masks and full-dimensional prior. Native equivalence
checks against the frozen reference must precede a new launch; this local
fabricated package does not certify those native kernels.

A bounded common backend policy for the new registration is:
1. Retain vectorized eager kernels as the checked reference. The only compiler
   candidate is one fixed-shape default compile of pure transform/conditioner
   tensor kernels, not a search over compiler modes. Keep finite validity
   decisions observable and outside compiled regions where necessary.
2. Charge compile, graph capture, benchmarking and cache construction inside the
   same applicable stage envelope for EVERY family, including shared A/root.
   Use a fixed fabricated forward/backward microcheck, no optimizer updates,
   and no native outcomes to select between eager and that one compiled backend.
   Restore/check model and RNG identities after this check; no synthetic fitting.
   A fixed decision rule is the lower projected remaining-stage time after five
   synchronized trials of each successfully checked backend. Both trials' costs
   are charged. No repeated engineering search is allowed within this study.
3. Compiler failure is saved and consumes its time. A predeclared eager fallback
   gets only the remaining envelope. Zero training updates fails the stage;
   no extra time is allocated to recover compile cost. Report genuine overruns,
   including a compile/in-flight update that crosses the deadline.

For each family, both schedule columns share its first90 backend decision/fork;
J may need a separately charged joint-graph compilation within its final90.
Same controls and failure semantics apply to all families. This policy is a
proposal; actual optimized native kernels are not included in this review.

Report charged source verification/preflight separately from fitting as in the
parent protocol. Inside fitting include model/optimizer construction, movement,
cache generation, all synchronization/validity checks, profiling overhead,
checkpoint/fork copies, serialization and setup. Record CUDA-event and wall
breakdowns for analysis, root/input-gradient path, prefix conditioning, scalar
head, QR/Cayley, response GEMMs, backward, Adam, validation and I/O. Do not infer
which component dominates from step counts. Rotate family execution order across
the three seeds with a fixed predeclared cyclic order; retain the within-family
fork chronology. No outcome-adaptive order or temperature retry.

Additional response work is two 704-by-16 projections per example per block,
small 48->32->32 heads and elementwise arithmetic: O(batch*D*r), with no new
CNN/attention pass or QR/Cayley factorization. The current frame is reused in
both response and mixer. The global summary is cached during the existing
conditioner evaluation; it is never cached across optimizer updates. This is
operation counting, not a V100 speed result.

## Numerical and provenance closure

Retain the original source closure, numerical banks, full precision receipts,
failed attempts and sigmoid endpoint counts. Do not edit the parent data or
manifest. New artifacts have their own manifest. Require actual finite source/
output/intermediate checks, dense fabricated Jacobian and parameter-gradient
checks, float32 native source inversion<=1e-3 and LD cancellation<=1e-2 per
example, and the original same-source exact-copy check. Save failing outputs
before rejecting them. No zero-row substitutions for incomplete banks, clipping,
resampling or silent numerical fallback. The inherited scalar tails remain
unchanged; report actual sigmoid zeros/ones without clipping.

The local prototype tests only a fabricated surrounding flow. Native integration
must additionally check its h-cache plumbing, masking, root input gradients,
analysis gradients, actual QR/Cayley reuse, and no-current/future-prefix leakage.
Production finite checks must include intermediate preactivations and scale
projections, not just finite values after tanh. Do not transfer a local CPU
arithmetic tolerance into a universal CUDA numerical certificate.

## Frozen native evaluation and stopping

Reuse the parent pinned Inception source/weights, preprocessing, batches, all-bank
unbiased polynomial KID, PRDC, quadrant covariance, both gradient energies,
paired normalized energy, float32 logits/float64 saved pixels, per-array likelihood
components, and generated bank integrity checks. Save raw data needed for the
independent recomputation already underway on the parent; this review does not
claim that recomputation has completed. Report every seed and every cell. No
average or significance claim rescues a failed seed.

Carry forward the old engineering requirements explicitly as *new-study*
requirements, without modifying historical receipts:
- I_F residual NLL < S42; I_F quadrant-covariance absolute error <= half S42.
- I_J complete NLL < I_F; I_J KID < S42, R_F, and R_J.
- Both I_J gradient means within10% of the original repair means.
- I_J energy <= each of S42, I_F, R_F, R_J plus .0005.

For the specific innovation-response mechanism, additionally require I_F to
beat P_F and I_J to beat P_J in both complete NLL and KID, in every seed, while
I_J is energy-noninferior to both P cells with the same .0005 tolerance. These
are demanding descriptive development orderings, not hypothesis-test guarantees.
The same candidate must pass all gates; no choosing F for one gate and J for
another. Other cells' metrics are still reported in full.

Record a fixed diagnostic of anchor-conditioned mean and energy misfit, not a
replacement success criterion. After each frozen model's base inverse, form its
u,v, standardized followers z=(v-m)exp(-s), and fixed features
Phi=[tanh(u),tanh(u)^2]. Save empirical matrices

    mean[ Phi^T (z U_f) ],
    mean[ Phi^T ((z^2-1) U_f) ]

per block on the same eligible repair records. Here the notation means the
average outer product per record (32-by-16 per matrix), not a product of sample
means. A correct conditional law makes these zero. They are restricted moment
tests, cannot certify conditional independence, and cannot select/rerun a model.
Differences between fitted coordinate systems must be labeled as model-adequacy
diagnostics, not common-coordinate covariance rankings.

Interpretation is fixed before data: equal P and I gains do not support the
innovation mechanism; only J gains suggest an optimization-allocation effect,
not unique representation superiority; R_J winning rejects an advantage claim;
NLL gains without the retained native quality gates stop this route. Do not
post-hoc increase rank, bins, root depth or time after a failure of this study.

## Efficiency and continuation gate

Generation-with-saving and evaluator-resident memory are not latency measures.
After model freezing, benchmark complete generation separately, with identical
full-source banks, batch1 and batch64, cold setup/cache compilation itemized,
warm median/p95 and peak resident plus duplicate cache bytes. Exclude file saving
and Inception from the measured generation call and charge them separately in
end-to-end operational totals. Use fixed run counts (five warmups, twenty trials)
and identical residency rules; no data-dependent best batch or timing selection.

Call this a practical efficiency win only if I_J passes every native quality gate
and is no slower at both batches and no higher in measured peak memory than BOTH
R_F and R_J, with at least one strict measured resource improvement against each.
Record all timing variation and abstain from population latency claims. If this
fails but quality succeeds, the result is an architecture-quality observation,
not an efficiency win.

Only after those native gates pass does the route justify a separately registered
comparison with a trained globally conditioned full-dimensional FM using the same
observations/analysis access, complete budget, active capacity at least as large,
and fully charged velocity NFEs/solver error. A direct full-D FM and capable
latent baseline must ultimately be evaluated on the same observable-quality/
end-to-end resource frontier before any broad superiority claim. FM numerical
likelihood is not silently labeled an exact closed-form density. No FM or latent
training is part of this review or is automatically authorized by this plan.

For video the proposed response cost stays linear in the number of retained
coordinates at fixed rank, but analysis, context processing and causal serial
depth may dominate. This is a route worth testing only after image gates pass,
not evidence of a video or latent-baseline advantage.
