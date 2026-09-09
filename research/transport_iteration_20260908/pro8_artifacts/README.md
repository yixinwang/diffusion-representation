# Iteration 8: exact, cached global innovations

Review target: `yixinwang/diffusion-representation`, branch
`agent/observation-transport-audit-20260908`, commit
`a397981d13f1bbe2121b58d5dac9f13089c243a7`.

This directory is a standalone review artifact, not a repository change. No PSC
job was submitted, queried, or executed here. No native data, protected split,
external model weights, or GPU were used. Repository tests were inspected, not
rerun. The experiments below are local CPU mathematical fixtures, not native
image or video performance evidence.

## 1. Proposed next native model: cached-global-innovation flow

Keep the current invertible pre-Haar/multiscale analysis A for the first paired
mechanism test. Replace the post-analysis coarse FM and residual decoder by one
exact triangular flow. All D=3072 independent standard-Gaussian source
coordinates remain, with 192 coarse and 2880 residual coordinates.

* Root: four exact alternating 96/96 coarse coupling layers. Start with
  width-32 two-layer dense conditioners and fixed permutations, not an ODE.
  The root's actual parameters, training, and latency must be counted.
* Residual: four fixed balanced blocks of 720 coordinates, using a declared
  channel/spatial four-colouring in the existing lossless packed layout.
* For block b, form H_b=(C,R_<b). Compute a masked local width-32 feature map
  from this fixed prefix. Pool four inducing queries to a 16-dimensional global
  summary E_b(H_b). Scalar heads receive the local feature, global summary,
  fixed coordinate embedding, and block identity. They must not receive active
  or future residual coordinates. Cache these quantities once for that block.
  The global summary is 16-dimensional; the complete local-plus-global context
  is NOT claimed to be 16-dimensional. Its dimension and cost must be reported.
* Each scalar head predicts a positive piecewise-linear density g on [0,1]
  using 8 fixed bins / 9 knot heights. A convenient parameterization is
  g = eta + (1-eta) sum_i softmax(logits)_i b_i / integral(b_i), eta=0.1,
  with fixed nonnegative linear hat functions b_i. Thus g>=eta and integral g=1.
  The CDF G is piecewise quadratic and its inverse has a stable quadratic formula.
  Generate r = shift + exp(log_scale) Phi^{-1}(G^{-1}(Phi(z))). Bound log_scale
  prospectively, and include affine Jacobians. This uses 9+2 head outputs,
  versus 3*8+1=25 in the current affine/RQS layer.

Generate the root, then each innovation block, then apply A^{-1} and the shared
outer sigmoid. Encoding reverses this sequence, with prefix summaries computed
from the observed encoded prefix. The Jacobian is block triangular. The model
has a normalized full-dimensional density, a constructive inverse, and no
finite-ODE invertibility gap. Its likelihood is continuous/dequantized; it is
not automatically the exact likelihood of discrete 8-bit observations.

The scientific hypothesis is that A handles enough local dependence that a few
prefix-conditioned, globally summarized innovation blocks suffice. The new
scalar head and removal of root integration may save work. The hypothesis is
NOT that four shallow conditional-independent blocks fit every native law.
Normal-CDF/quantile kernels, serial block work, and summary extraction can erase
the proposed savings. Optimize equivalent inactive-head work in controls too.

Legal cache distinction: an FM comparator may cache its unchanged coarse context
and corresponding summaries. Its evolving state-dependent features cannot be
cached as though they were unchanged. Likewise E_b must be recomputed when the
prefix changes between blocks.

### Native error ledger

Write S_bj for the complete scalar context, including local features and the
learned global summary. For a fixed fitted A and fixed fitted summaries,

  KL(P_X || Q_X) = kappa_C + sum_b [
       TC(R_b | H_b)
       + sum_j I(R_bj ; H_b | S_bj)
       + sum_j E KL(P(R_bj | S_bj) || q_bj(. | S_bj)) ].

Here kappa_C=KL(P_C||q_C). TC is conditional total correlation. Expectations and
conditional information are under the actual data law pushed through A. This
is an exact chain-rule decomposition, not a bound assuming the summary is
sufficient. The last term contains scalar approximation, estimation and
optimization errors. Any misspecification induced by the learned analysis is
reflected in the resulting conditional laws and information terms. There is no
native bound on these terms in this review. Treating all pixels as independent
training samples would be unjustified.

## 2. Explicit nonlinear, non-Gaussian positive class

Let phi and Phi be the standard-normal density and CDF, and psi(r)=2Phi(r)-1.
For |e|<1 define

  p_e(r) = phi(r) [1 + e psi(r)],
  F_e(r) = (1-e) Phi(r) + e Phi(r)^2.

This is a smooth, positive density on all of R, because psi is uniform on
[-1,1] under a standard normal and has mean zero. Its density ratio to phi is
between 1-|e| and 1+|e|. It is non-Gaussian for e!=0. Its exact Gaussian-source
transport is

  Q_e(z) = Phi^{-1}( 2 Phi(z) /
      [1-e + sqrt((1-e)^2 + 4e Phi(z))] ).

This is precisely the one-bin subfamily of the proposed positive-linear-density
head, not an unrelated approximation theorem for a different spline family.
The implementation uses log-CDF/log-survival arithmetic; it does not clip or
truncate Gaussian sources.

### Full-dimensional model and actually learned quantities

Set c=192, m=2880 and M=16. Let O have 16 fixed orthonormal dense rows in R^192,
shared by both algorithms. Let unknown gamma be either -1/2 or +1/2. Generate
independent U~N(0,I_192), V~N(0,I_2880), and

  C_i = Q_gamma(U_i),
  h_j(C) = 2Phi((O Q_gamma^{-1}(C))_j)-1,
  e_j(C) = 0.5 + 0.1 h_j(C),
  R_l = Q_{e_j(C)}(V_l), l=1,...,2880.

The unknown j belongs to {1,...,16}. Under the true root, the h_j's are independent
Uniform[-1,1], and e_j lies in [0.4,0.6]. Apply a fixed inverse two-level orthogonal
Haar transform and sigmoid to (C,R) to obtain an open-unit-cube RGB32 array.
No source coordinates are discarded and no extra hidden randomness is used.

Both algorithms know the class, O, dimensions and fixed amplitudes. Neither is
given the true root sign or summary index. The class is a finite catalog of
smooth full-dimensional distributions, not an unrestricted neural-summary or
nonparametric-native-image class. A is fixed and known in this theorem; it is
not claimed that native learned A recovers it.

### Finite learning theorem

Use 32 independent observed arrays to choose gamma by the sign of the average
psi(C_i). Under the truth E psi(C_i)=gamma/3, and all 32*192 root coordinates
are independent. Hoeffding's bound gives

  P(root wrong) <= exp[-32*192/72] = 8.7137322468e-38.

Use a disjoint 256-array sample to select j by exact maximum likelihood over the
16 candidates, after applying the fitted root inverse. There is no optimizer
gap for this finite scan. Conditional on correct root selection, write affinity
as Aff(P,Q)=integral sqrt(pq). For scalar heads e,f in [-a,a], a=0.6,

  Aff(p_e,p_f) <= 1 - (e-f)^2/[24(1+a)].

Proof: (sqrt(1+e psi)-sqrt(1+f psi))^2 is at least
(e-f)^2 psi^2/[4(1+a)], and integral phi psi^2=1/3. Affinity is one minus half
the squared Hellinger distance. Conditional independent residual coordinates
multiply affinities. For a wrong candidate k!=j,

  P(|h_j-h_k|>=1/2)=9/16,
  rho = 1 - (9/16) [1-exp{-2880*0.1^2*0.5^2/(24*1.6)}]
      = 0.9038288789764752.

The likelihood-ratio square-root bound and union bound give

  P(summary wrong | root correct) <= 15 rho^256
                                 = 8.593339504996332e-11.

Consequently the fitted exact generator equals the full data distribution with
probability at least 1-8.594e-11. Its full joint KL is zero on this event. This
is a high-probability event statement, not unconditional zero expected risk.
The many residual replicates are legitimate ONLY because conditional independence
is an explicit assumption of this positive class. This argument cannot pool
native dependent pixels as independent replicates.

## 3. Precisely restricted, stronger plug-in FM comparator

The comparator uses the SAME fitted root and summary, the SAME complete source,
the SAME outer inverse Haar/sigmoid and the SAME training algorithm/cost. It is
given the exact Gaussian-reference population velocity of its fitted scalar
law, so it has no flow-field regression noise, optimization error, or insufficient
receptive field. It caches e(C) once. It then uses uniform Heun with actual call
counts N in {4,8,16,32,64}, as in the current strengthened-control grid.

Its restriction is algorithmic: the canonical Gaussian-reference field and that
uniform Heun solver must be used. It cannot replace the field with a solver-
compensating field, use the analytic inverse CDF, distill an endpoint map, or
switch to an unrestricted decoder. An exact-copy decoder ties. This theorem is
NOT a lower bound for every neural FM parameterization or every solver.

### Exact nonlinear Gaussian-reference field

For a=1-t, s^2=a^2+t^2, d=sqrt(2a^2+t^2), k=t/d, the variance-normalized
independent path Y=(aZ+tR)/s has marginal

  p_t(y) = phi(y) [1+e(2Phi(ky)-1)].

Indeed R conditional on the base Gaussian Y is Gaussian with mean (t/s)Y and
variance a^2/s^2. Averaging Phi(R) gives Phi(ky). Differentiating this CDF and
using the continuity equation yields

  w_t(y;e) = [2 e a/(s^2 d)] phi(ky) /
             [1+e(2Phi(ky)-1)].

Equivalently k'=2a/d^3 and w=2e k' phi(ky)/[(1+k^2)(1+e psi(ky))].
The field depends nonlinearly on the evolving y. This is not the Gaussian zero-
field special case.

### Global Heun invertibility for this comparator

A uniform bound |partial_y w|<0.55, hence <=0.6, holds for e in [0.4,0.6],
t in [0,1], y in R. Here are explicit bounds, not finite input tests.
Let q=ky, lambda=2a/(s^2 d). Then lambda*k<=3/2 and e<=0.6, so
lambda*e*k<=0.9. The inequality lambda*k<=3/2 follows, with r=t/a>=0, from

  3r^4-4r^3+r^2-4r+6 >= 0.

For r<=2 the polynomial is (r-1)^2(3r^2+2r+2)+4-2r; for r>=2 both
r^3(3r-4) and (r-2)^2+2 are nonnegative.

The derivative, apart from -lambda*e*k, is

  q phi(q)/(1+e psi(q)) + 2e phi(q)^2/(1+e psi(q))^2.

For q>=0 its magnitude is at most 1/sqrt(2*pi*Euler_e)+1.2/(2*pi), giving
less than 0.390 after multiplying by 0.9. For q<0 the two terms have opposite
signs, so bound their maximum, not their sum. The first term is at most
1/[0.4 sqrt(2*pi*Euler_e)], giving less than 0.545. For -1<=q<0 the second
term uses 2Phi(1)-1<0.683 and gives less than 0.50. For q<=-1 use
phi(q)^2<=exp(-1)/(2*pi), giving less than 0.40.

Thus for h=2/N<=1/2 the Lipschitz constant of a Heun-step perturbation of the
identity is at most hL+(hL)^2/2<=0.345<1. Each scalar step is a global increasing
bijection, with a well-defined density. The composed Heun median is H_N(0;e).
This removes density ambiguity on the comparator side as well.

### Actual endpoint lower bound, with cancellation controlled

Define d_N(e)=1/2-F_e(H_N(0;e)). The interval program encloses this quantity over
ALL e in [0.4,0.6], not just a grid of point evaluations. It propagates 65,536
parameter intervals through all Heun steps using outward-rounded binary64
arithmetic. Normal PDF/CDF values on the reached [-1,1] domain are enclosed by
19-term polynomials with explicit Taylor remainders. It checks domain and
nonzero-denominator conditions. The arithmetic assumption is IEEE-754 correctly
rounded elementary operations and sqrt plus nextafter; it is not a proof-
assistant formalization. No repository ordinary-float Lipschitz certificate is
being relabeled as an interval proof.

The comparator puts probability 1/2 below its median. The truth puts probability
1/2-d_N there. Data processing to this binary event gives

  KL(p_e || Law[H_N(Z;e)]) >= kl(1/2-d_N || 1/2) >= 2d_N^2.

Conditional product structure and the common exact root give the full joint
lower bound 2*2880*inf_e d_N(e)^2. On successful parameter recovery, exact-flow
KL=0 while every declared Heun point has strictly positive KL. The recorded
interval lower bounds in nats PER FULL ARRAY are:

| Actual calls | CDF defect lower | Joint KL lower |
|---:|---:|---:|
|4|0.00328581134203648|0.062188163570624|
|8|0.00156032554443525|0.014023387034595|
|16|0.00040409736810215|0.000940577373545|
|32|0.00010153870753965|0.000059386228582|
|64|0.00002469336064098|0.000003512229464|

Use the JSON's directed lower values rather than rounded table entries for any
formal downstream check. The separation becomes very small at high call counts.
It does not establish a meaningful perceptual gap on native images.

## 4. Nonzero root/summary error: explicit robustness

The conclusion need not depend exclusively on exact finite-catalog recovery.
Let the true conditional tilt be e(C), fitted tilt f(C), both in [0.4,0.6],
and |e(C)-f(C)|<=delta uniformly. Give both models the same fitted root q_C,
with finite kappa_C=KL(P_C||q_C), and the fixed common invertible chart.
Then chi-square dominates KL and

  KL(p_e||p_f) <= (e-f)^2/[3(1-0.6)] = (e-f)^2/1.2.

For a Heun median at fitted f, if u=Phi(H_N(0;f)),

  F_e(H_N(0;f))-F_f(H_N(0;f)) = (e-f)(u^2-u),

whose magnitude is at most delta/4. Consequently

  KL(P||Q_exact) <= kappa_C + m delta^2/1.2,
  KL(P||Q_FM,N) >= kappa_C + 2m [d_N,min-delta/4]_+^2.

These are an UPPER bound and a genuine comparator LOWER bound, not two upper
bounds. The same nonzero root error appears in absolute quality and cancels
only in the comparison. A sufficient strict separation condition is

  delta < sqrt(2)*d_N,min / [1/sqrt(1.2)+sqrt(2)/4].

At delta=0.001, the exact conditional KL upper bound is 0.0024 nats/full array;
FM's lower bounds are 0.0530850 for four calls and 0.00988965 for eight calls,
plus the same kappa_C. Thus the corresponding improvement is at least 0.05068
and 0.00748 nats/array. The condition need not hold at larger N. Since e=0.5+0.1h,
this delta corresponds to at most 0.01 uniform summary error when amplitudes
are correct. No evidence that a native neural summary achieves this uniform
accuracy is asserted. General-head and analysis misspecification must be added;
the conditional-information ledger, not this tilt-only bound, governs native data.

### Expected-risk separation including wrong selections

The high-probability statement also gives a nonvacuous unconditional expected-
KL comparison over the random training data. A wrong summary still produces
f,e in [0.4,0.6], so the conditional KL is at most
2880*(0.2)^2/1.2=96 nats. A wrong root contributes at most
192*(1)^2/[3*(1-0.5)]=128 additional nats. Therefore

  E_training KL(P||Q_exact) <= 96*(15*rho^256) + 224*exp(-32*192/72)
                           < 8.25e-9 nats/full array.

For the FM comparator nonnegative KL off the correct-selection event gives

  E_training KL(P||Q_FM,N) >= (1-p_fail)*2m*d_N,min^2.

Even at 64 calls this lower bound exceeds 3.51e-6 nats/full array, so there is
an expected-risk separation, not only a conditional-on-perfect-learning claim.
These are statistical real-arithmetic risks, not rounded floating implementation
KL estimates. `finite_risk_bounds.json` records this calculation without
refitting or retiming the reference.

## 5. Cost and executed reference

Let C_common include the root, legal shared summary cache, output transform and
allocation costs; let c_Q and c_w be per-residual-coordinate exact-head and field
costs for a specified implementation and batch size. Then

  C_exact = C_common + m*c_Q,
  C_FM(N) = C_common + N*m*c_w.

Generation is strictly cheaper when c_Q<N*c_w. A twofold full-pipeline gain
requires C_common+2*m*c_Q<=N*m*c_w. Training is identical for this comparator,
so any verified sampling savings also reduce total cost at positive sample
count. These conditions are not architecture-independent GPU-speed theorems.

`positive_class_reference.py` implements the complete known-chart 3072-dimensional
class. Training sees observed Haar/sigmoid arrays, not privileged U, gamma or j.
The script fitted the correct gamma=0.5 and j=7 from 32+256 disjoint arrays.
Common fitting time was 0.0545 seconds, fully attributed to both algorithms.
Full-source round-trip max error was 5.11e-15; scalar tail tests through |z|=50
had max error 1.42e-14; the same-weight stochastic copy was bitwise equal.
These finite tests are not arbitrary-tail floating-point guarantees; sigmoid
can saturate at sufficiently extreme floating inputs.

One local float64 CPU process, one-thread BLAS, 3 warmups/arm and 9 randomized
interleaved timing repetitions/arm, measured the following full-pipeline
batch-64 medians, including coarse generation, summary and Haar/sigmoid:

|Method|Milliseconds per batch64|
|---|---:|
|Exact|12.5263|
|FM4|13.1395|
|FM8|26.4280|
|FM16|48.2608|
|FM32|94.9656|
|FM64|189.6397|

At batch1 the respective medians were 0.2909,0.3502,0.4684,0.8556,1.4915,2.7556ms.
No training-memory or GPU advantage is measured. The four-call margin is small
and not robust performance evidence. The first implementation evaluated both
normal-quantile tail branches, discarded one, and LOST at batch64 to FM4:
16.1546ms vs 12.8978ms. Its source and result are retained as `*_two_branch.*`.
The current code computes only the selected branch; the mathematical law is
unchanged. This optimization history must not be suppressed. No repeated-
process/hardware confirmation was performed.

## 6. Prospective native experiment and training ladder

Do not change the already-frozen native PSC pilot or its original outcomes.
The following is a new proposed protocol, not an executed experiment.

### Stage A: numerical and dependency gate

Check tiny-dimensional full Jacobians, inverse source recovery INCLUDING THE
ROOT, full logdet cancellation, all-coordinate participation, tail-safe scalar
inversion, finite gradients, and no active/future-coordinate conditioner input.
Add adversarial source magnitudes separately from untruncated Gaussian rows.
An implementation may fail at finite precision even though its real-arithmetic
map is a bijection. Preserve every failure. Test that a Gaussian target and
zero correction produce no claimed strict advantage over the zero-field
Gaussian-reference control. Test the non-Gaussian positive class above and a
source-identical failure fixture with dependence deliberately omitted by the
summary. Independent reference checks should reproduce the interval audit and
compare the analytic velocity with numerical continuity-equation calculations.

### Stage B: 4000-fit native mechanism test, 360 synchronized seconds per arm

Keep the original eligible 4000 fit IDs and 1000 reused repair IDs. Partition
only those fit IDs by a predeclared hash into 2000 discovery, 1000 scalar-head
fit and 1000 internal-validation records. Historical fit/repair use means this
is research validation, NOT newly untouched confirmation. Official test and
excluded discovery remain closed.

Use 90s for A, 90s for the exact root and 180s for innovation fitting; prospectively
split the last 180s into 120s discovery summary/provisional-head fitting and 60s
head fitting on the separate head sample with summaries frozen. Every arm gets
the same permitted fitting observations; end-to-end controls may use the union
of the 3000 fitting records. Use last checkpoints, no repair-derived budget or
NFE choice. Count actual updates, exposures, overrun, fit-cache and setup costs.

Root attribution: retain the original learned root-FM output as an ablation,
with identical source rows and both conditional decoders; do not alter the old
published pilot. Decoder attribution: put both the candidate and the existing
four-global-RQS decoder and Gaussian-reference residual FM behind the SAME new
exact root and same A. Give FM its valid coarse-context cache. Retain all five
fixed NFE points. Keep the ordinary-path result as a legacy control, not the
sole comparator.

Summary attribution: same local conditioner and scalar head, with global-summary
sizes 0,16 and64; a full-prefix globally receptive conditioner is the capacity
reference. Freeze the three summary sizes in advance. Compare held-out per-array
log losses. Candidate16 should recover at least 90% of a positive full-context-
over-local improvement before the summary bottleneck is promoted. This is a
prospective operational gate, not a statistical upper bound on conditional MI.
If the full-context gain is nonpositive, there is no established global-summary
mechanism to promote. A wider summary winning does not validate size16.

Use conditional PITs and fixed within-block/far-pair dependence descriptors,
plus residual prediction probes with the missing prefix context, to try to
falsify summary sufficiency and within-block independence. Negative probes are
not independence certificates. Separate teacher-forced conditional quality
from unconditional generations using GENERATED root/prefix; real-context
sampling cannot establish the generator's quality.

The full standalone competition must retain the fixed 264,912-parameter full-D
ordinary/Gaussian-reference models at the entire 360s budget and a pure exact
NLL flow receiving that entire budget, not merely the 90s A-only model. Before
strong promotion, also match whole-pipeline parameter count by a shape-only
width rule. Residual-only parameter matching is not full-model matching.
All models must use the same observation/logit convention. The new full-D FM
module does not itself apply the shared logit/sigmoid. Its source packing uses
pixel order, whereas shared-analysis paths use a block permutation; explicitly
align the mathematical source via fixed permutations rather than asserting
identical semantics from seed equality.

### Stage C: only after numerical and mechanism gates

Run 3 declared seeds at the 360s budget, then 5 declared seeds at 4x that budget
if the model survives. Keep all failed seeds and NFE points. Use the existing
pinned perceptual KID/PRDC evaluation, fixed sample banks, nearest-neighbour
checks, original energy, and fixed dependence descriptors. Interleave warmed
batch1/64 timings; report median/p95, allocated/reserved peak memory, host/cache
storage, all learned parameters, training, preprocessing and checkpoint costs.

A proposed strong-promotion requirement is at least 2x full-pipeline generation
speed at both batch1 and64 against a matched-quality control, at equal standalone
training budget, together with no more than 0.02 absolute PRDC precision or
recall loss and KID not worse by more than 5% of the control's positive estimated
KID. Use confidence intervals that respect image/video and training-seed units,
report simultaneous comparisons to the fixed frontier, and do not declare a
pass when small-sample uncertainty cannot resolve these margins. These are
prospective engineering margins, not claims that the small repair study can
confirm superiority. Retain every original metric even when an appendix favours
a method. Lower NLL alone does not imply perceptual improvement. Do not open
protected 40k/5k/5k or official test allocations under this review; larger-data
promotion needs a separately frozen, permitted protocol.

### Real-video mechanism test, after image gates

Use a separately frozen manifest from UCF101 official split-1 TRAINING videos
only, grouped by (action, published group ID) before any discovery/head/validation
assignment. No group labels enter the generator; grouping only prevents related
videos crossing partitions. A candidate small roster is up to 256 discovery
videos,128 head-fit,128 development, one four-consecutive-frame32x32 RGB clip per
video, selected deterministically before decoding/scoring. Do not silently fill
short groups by leakage across partitions. This roster has D=4*3*32*32=12,288;
all temporal and channel coordinates stay. Fixed lossless spatiotemporal packing
or Haar is allowed; interpreting RGB channels as time is not.

Preserve frame order/timestamps. Compare joint video samples with independent-
frame and repeated-frame controls, fixed lag1/lag2 and spatiotemporal-gradient
descriptors and a temporal energy score, as well as frame appearance. Frame-only
KID cannot establish temporal generation quality. A model that wins only frame
metrics but fails native temporal dependence stops at the video gate. Charge
all temporal state, cache and output costs. This proposal has not accessed UCF
files or evaluated generated video.

## 7. Prior work and novelty scope

Primary sources checked during this review:

- Müller, McWilliams, Rousselle, Gross, Novak (2019), Neural Importance Sampling.
  Piecewise-polynomial coupling transforms and analytic inversion are established
  ingredients, not a new priority claim here. Disney Research publication page.
- Yu, Derpanis, Brubaker (2020/2021), Wavelet Flow, arXiv:2010.13821.
  Conditional multiscale exact flows predate this construction.
- Lee et al. (2019), Set Transformer, PMLR97:3744-3753.
  Inducing-point attention supplies a precedent for cheaper global aggregation;
  it does not prove native summary sufficiency.
- Zhai et al., Normalizing Flows are Capable Generative Models, arXiv:2412.06329,
  ICML2025 (TarFlow). Pixel-space/no-VAE flow generation is already established.
  Its augmentation, post-denoising and guidance must not be silently folded into
  a claim that the final postprocessed sampler is a single exact bijection.
- Chen et al. (2026), Normalizing Flows with Iterative Denoising,
  arXiv:2604.20041 (iTARFlow), further limits broad novelty claims.
- Official UCF101 dataset page: published groups can share background/viewpoint;
  hence grouped video splitting is required for the proposed diagnostic.

The prospective contribution is the particular exact-root, cached-global-summary,
analytic-innovation allocation and an attributable quality/cost result if native
experiments support it. The finite-class and plug-in-Heun separation is a
constructive scoped mechanism witness. It does not prove superiority to native
full-D learned FM, all stochastic latent models, or optimized general flows.
No meaningful semantic representation advantage is established by invertibility
or Gaussian likelihood alone; that requires separate same-information probes.

## 8. Reproduction and files

Run in a separate local directory; these commands do not contact PSC or GitHub:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python positive_class_reference.py \
  --seed 77181 --repeats 9 --output my_reference_results.json
OPENBLAS_NUM_THREADS=1 python interval_tilt_check.py \
  --cells 65536 --output my_interval_results.json
```

The reference needs NumPy and SciPy; the interval program needs only NumPy.
The saved environment versions appear in the result JSON. Interval enclosures
are arithmetic-assumption-dependent; runtime is hardware/process-dependent.
The native neural model is a specification, NOT implemented by these scripts.
`positive_class_reference_two_branch.py` and its JSON retain the initial slower
implementation. `SHA256SUMS` records final artifact hashes.
