# Pro14 native-pilot diagnosis and one next decision

**Recommendation:** test a small nonlinear response from existing innovations
inside each cached block, against an identical prefix-only response, crossed
with frozen versus joint analysis training. Include optimized RQS in both
training columns. Keep the current rank, frame, scalar bins, root, and masks.
Do not widen the frame or replace the native question with a synthetic result.

## Evidence status

The native pilot remains failed in all three seeds. Its supplied numbers are
provisional source-matching reports pending independent raw-feature/pixel
recomputation. This review inspected frozen source but did not repeat any native
verification, fit, data read, metric computation, or timing benchmark.

RQS has lower KID and lower complete NLL than J in all three reported seeds.
M's residual likelihood ordering and J's complete likelihood ordering show that
the mechanisms affect density, but do not establish image-quality gains. The
reported horizontal/vertical gradient energies are approximately64%-102% above
the repair reference, not close to their10% acceptance bands. J's energy-score
noninferiority is not a strict quality win. All historical failures remain valid.

## What the source distinguishes

**Conditional dependence:** the old block is a single affine mixture of independent
scalar innovations for fixed H. Its conditional covariance is diagonal plus a
rank-at-most32 correction. This is a real restriction, but neither the native
unconditional pixel covariance nor its failure identifies that restriction as
the causal bottleneck. A still more basic missing capability is direct nonlinear
within-block value-dependent modulation of other innovations. Global prefix
attention is already present; the missing feature is not simply global access.

**Scalar/head restrictions:** equal-width8 bins, bounded derivative, identity
scalar tails, bounded explicit shifts/scales and bounded mixer eigenvalues also
matter. RQS differs on several of these and on masked within-layer conditioning,
so M-vs-RQS does not isolate rank or the frame. The prefix-only response control
is included specifically to give comparable extra head and scalar flexibility
without stochastic within-block modulation.

**Frame chart:** chart boundaries and initialization can affect conditioning.
At an identity mixer, frame gradients from that mixer vanish. But the graph
chart's excluded boundary does not prove a substantial approximation gap, and
no fitted frame spectra or conditioning traces were read. A rank/chart diagnosis
is not supported by the reported covariance ratio alone.

**Analysis/root:** fixed-A M and RQS have the same analysis and root; those alone
cannot explain their fixed-A difference. They may still impose a common quality
ceiling. J changes the distributions entering its frozen root and changes its
analysis determinant, so its smaller residual NLL cannot be compared to RQS's as
a coordinate-invariant quality improvement. Only the complete ledger supports
that comparison. Joint RQS is necessary to test the analysis-allocation story.

**Cost:** the code computes QR/Cayley once per block application, not once per
sample; its inference cache is not a cache across training updates. Joint steps
recompute analysis and retain gradients through the fixed root. Model-internal
validity decisions add synchronization beyond the optimizer's three checks.
Those are plausible costs, not a measured explanation for453-458 J updates.
Training profiles and a separate complete-generation benchmark are required.
Save/evaluator-resident times cannot establish a latency advantage.

## The bounded change

Let B_H denote the complete old block. Insert F_H before it:

    F_H(u,w) = (u, m(h,u)+exp(s(h,u))*w),

where16 existing Gaussian coordinates u are fixed, evenly spaced anchors,
704 coordinates w remain independent Gaussian followers, h is the existing
16-number cached global prefix summary, and m,s use follower rows of the
existing rank16 frame. A48->32->32 head yields16 mean and16 scale coefficients.
No extra noise, CNN, attention pass, QR, or dense720-by720 matrix is introduced.
There are10,496 additional parameters, about1.99% of the old whole model.

The explicit inverse, exact determinant, normalized full-dimensional density,
KL decomposition, approximation assumptions, moment mechanism and observable
bounds are in THEORY.md. The native experiment and failure/efficiency gates are
in PROTOCOL.md. The candidate nests the old block when its output head is zero;
its comparator has identical active parameters but uses only h.

This is not claimed as a new generic type of normalizing flow. The scientific
claim at issue is a scoped finite-budget allocation of nonlinear dependence.

## Executed fabricated evidence

The final local CPU checks passed. They include a28-dimensional complete dense
Jacobian with nonconstant prefix dependence, gradients through response heads
and the reused frame chart, a3,072-dimensional fabricated flow in both precisions,
zero-head nesting, explicit normalization, and invalid-input/intermediate rejection.

Selected final results:

| Check | Result |
|---|---:|
| Worst small dense-logdet error | 3.1503e-15 |
| Full3,072 float64 round trip | 7.1054e-15 |
| Full3,072 float32 round trip | 3.8147e-6 |
| Full3,072 float32 LD cancellation | 6.5565e-6 |
| Reused-frame directional-gradient error | 1.3026e-10 |
|2D response density integral | 1.0000000000000027 |
| Exact Cov(U,V) in energy witness | 0 |
| Cov(U^2,V^2) in energy witness | .5364014664 |

The witness's .02260215-nat advantage is only over a moment-matched independent
Gaussian. There was no RQS or FM fit/comparison, image metric, or native performance
result. A correlated-follower fixed-base failure (.22314355 nats conditional TC)
is retained explicitly, not hidden or generalized into a full-model floor.

The first run failed in the fabricated scalar harness because a bin index promoted
float32 arithmetic to the global float64 default. Initial code, traceback, and
partial receipt are preserved unchanged under attempts/. The correction explicitly
casts that index to the input dtype. The final run also rejects an intentionally
missing determinant and hidden nonfinite preactivation instead of hiding them.
No failed native receipt was altered.

## Delivery and limits

This package contains the review, theory, proposed protocol, standalone response
module, fabricated harness, results, source-inspection metadata, and retained
attempts. It does not contain copied native datasets, fitted states, original
native result arrays, or the full frozen repository source closure. Source was
read through the GitHub connector; a direct raw-source download into the sandbox
failed due to DNS, so no unchanged-source byte-copy claim is made.

GitHub publication and native integration are deferred until review. There is no
new repository commit, active-branch modification, PSC job, or native inference
associated with this package. No image/video efficiency or latent-baseline
superiority is established. The next claim is rejected unless the same native
candidate passes all retained quality gates and the fully charged comparison to
both fixed and joint strong RQS.
