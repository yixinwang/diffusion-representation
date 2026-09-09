# Reproduction and matched-quality protocol

## Frozen scientific object

Use the original Pro8 class, training sizes, public exact orthonormal dictionary,
Gaussian source law, population field, uniform Heun step sizes and complete
D=3072 chart. Do not fit, alter the law, select parameters from teacher outputs,
or use the native/dependence work. The local certifier reads no fitted artifact
or source bank. Its integration variables are the fresh scalar Gaussian source
z and the fitted tilt e, analytically uniform on [0.4,0.6] on correct recovery.

The declared fixed grid is 4,8,16,32,64 mathematical stages. Endpoint specialization
has N-2 nontrivial kernels. The certified real map agrees algebraically with
both the older all-call formula and the optimized endpoint formula. Differences
in floating arithmetic remain subject to the separate archived parity audit.

## Primary numerical certificate

Evaluate all five settings, retaining successes and failures. Primary settings:
6-node tensor Gauss-Legendre, 12th normalized derivatives, MPFR 128 bits, initial
48x4 compact partition, z in [-12,12], e in [2/5,3/5], requested full-array
quadrature error-radius allowance 5e-9, maximum adaptive subdivision depth 20.
The returned integral interval, not the requested allowance, determines success.
All five actual conditional widths are <=4.66e-10. Analytical tail error is added
outwardly. Denominator, logarithm and step-Jacobian positivity are enforced.

Compute finite-learning correction separately at 192 bits using `risk_bounds.cpp`.
Use the final exact decimal endpoints of each returned conditional enclosure.
Reject reversed/nonfinite intervals or a final full-array width exceeding 2e-8.
Do not substitute the weaker `unconditional_upper` in the initial integration
receipt for the final expected-risk enclosure.

A changed-order/changed-mesh N4 certificate is a secondary consistency check:
8 nodes, 16th derivatives, 192 bits, initial 32x3 partition. It is not used to
estimate quadrature error. Each individual run has its own rigorous remainder.

## Assurance levels

1. Analytic proof: the restricted law, discrete Jacobian, bijectivity, KL identity,
   quadrature error formula, Gaussian tail and learning-failure inequality.
2. Directed computation: MPFR interval evaluation of constants, quadrature nodes,
   weights, function values, derivative bounds and final statistical correction.
   Exported decimal interval endpoints remain outward.
3. Consistency tests: 256 checks at 192 bits plus all-domain constant checks and
   one changed-order certificate. Tests do not replace levels 1 and 2.
4. Ordinary observations: archived binary64 sampler timings/parity/roundtrips,
   ordinary floating quadrature values, local certificate runtimes and traced
   memory. These do not themselves certify KL or deployment latency.

The arithmetic trust assumptions are MPFR/GMP correctness, the C++ compiler,
valid public ABI declarations, and correct implementation of the displayed
interval recurrences. No proof assistant or verified compiler is claimed.
Prefer vendor development headers. The minimal header fallback was tested only
with Linux x86-64 and the recorded MPFR/GMP runtime; do not assume other ABIs.

Literal rounded finite-float output laws are discrete. The finite KL here is for
the underlying exact-real continuous maps, NOT continuous truth against those
discrete laws. No rounding claim about SciPy quantiles, QR, learned catalog
scores or sigmoid saturation follows from MPFR verifier accuracy.

## Decision rule

At a target tau, an algorithm is eligible only when its upper bound is <=tau.
It is excluded only when its lower bound is >tau. Otherwise it is unresolved.
Maintain separate conditional and expected-training-risk decisions. For a
nonnegative exact sampler with upper bound 8.2496059247963795e-9, the implicit
lower bound is zero; all six requested targets are eligible.

Among eligible fixed-grid Heun settings, select the smallest archived primary
median independently within each source/batch case. Do not infer ordering from
N alone; `frontier.py` explicitly minimizes those observed costs. For these
records the common selected N is 8,16,32,64,64,None over the six targets.
Use paired ratios endpoint_Heun / central_exact for the SAME seed and batch,
then report their range. Do not divide an unrelated maximum by a minimum.

At 1e-6, every fixed-grid comparator is excluded. Report no eligible grid member,
not infinity, an undefined latency ratio, or a lower bound for all methods.
For a target that straddles an interval, report unresolved. At loose quality
above approximately 0.223724714, Heun4 is eligible and its measured cost advantage
remains an adverse case for a blanket exact-sampler speed claim.

## Cost evidence and exclusions

The PSC source freeze is 9391d3ab89ad0667036d39a85f2e59409da34dc6; archived results
were inspected at 4ec7341be5b1d8a84700354ac492b854f12471eb. The retained primary
summary is hash-matched to that result manifest. This iteration recomputes paired
ratios from it but does NOT rehash all 129 PSC payloads, load all 102 output
arrays, replay the 78 gates, or resubmit the timing job. Those verifications are
reported by the inspected prior machine-check record, not newly performed here.

The primary endpoint implementation constructs its time-only cache inside each
call. The separately timed secondary schedule amortizes ALL-call cached Heun,
not endpoint Heun; do not mix its numbers into this primary frontier. The
central/tail exact threshold is 5 in the PSC benchmark; the old Pro10 standalone
threshold of 8 is not substituted. The common root/context/Haar/sigmoid work is
included in the compared primary timed calls. Setup/training is common and no
independent-fit uncertainty is obtained from three source seeds.

The user-reported traced-memory negative, exact central 16.69 MB versus endpoint4
10.42 MB, is retained as prior evidence; it was not remeasured here. Tracemalloc
is not intrinsic memory, and there is no new memory advantage claim.

## Failure handling and deliverable boundaries

`run.py` writes only into a new local directory, logs every process and enforces
per-process timeouts. Failed runs preserve existing outputs and write FAILED,
never COMPLETE. A partial `--n 4` invocation explicitly records that the full
grid was not requested. The actual primary results cover the complete grid.
No network calls, model data, fitting code, scheduler commands, repository
writes, or source-bank regeneration exist in the runner.

All new artifacts are local. No branch, commit, pull request, PSC job or native
experiment was created. The final archive includes source, receipts, logs,
proof/protocol notes and hashes, not external libraries or scientific binaries.
