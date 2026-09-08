# Diffusion representation: independent iteration-5 audit

Reviewed source: `ffec36e35c325d36bc9cdfc73b04305f6f485205`, parent
`adc1699357676c3d5015c5258ff3edd226a5c95c`. This is a prospective research
specification and independent deterministic algebra check. No model was fitted,
no real data were loaded, no PSC job was submitted, and no official test was
opened by this review. The two supplied PSC job statuses are not live-verified.

## Decision

Do not claim one proved-and-competitive method yet. There are three distinct
estimators: the add-one interpolated histogram, the proposed convex bounded
linear-density spline, and the trained neural rational-quadratic coupling flow.
Their learning results are not interchangeable. The existing two frozen pilots
must remain unchanged. Their completion would establish at most their registered
synthetic conclusions, not realistic image/video advantage.

For the next proof-connected empirical candidate, use ONE complete staged
transport: a learned invertible analysis, followed by a learned conditional CDF
law with frozen neural feature gates and convex response heads. Train every
part from permitted observations, include all Gaussian coordinates, and charge
all stages. Do not deploy a tensor-product table over image patches. Do not run
another family of hand-designed synthetic worlds after correctness. The next
scientific decision is a bounded CIFAR-32 development comparison against a
learned-analysis latent FM with a globally dependent stochastic decoder.

This specification is not an implemented real-data runner. The accompanying
Python contains independent bound, inverse, memory, and provenance checks only.
The root's separate convex implementation can use the finite-feature extension
below; no existing model or frozen experiment is modified here.

## What was verified in the pinned source

`qalt/src/qalt/conditional_cdf.py` fits one observed parent per coordinate using
add-one counts. It is not a learned-context neural model. Its encode/decode
implementation rejects Gaussian-CDF saturation instead of silently clipping.
The code iterates through coordinates even for a parallelizable graph: a
log-depth graph is not a measured log-depth implementation.

`qalt/src/qalt/multiscale_flow.py` learns both the coarse and detail distributions,
uses every Gaussian coordinate, and supplies a copied stochastic-latent wrapper.
Its checkerboard neural rational-quadratic couplings are not the convex
linear-density spline sieve. In unit-interval mode it uses logit, Haar, neural
couplings, and a Gaussian source; the compact-density assumptions are not
inherited from that architecture.

`qalt/src/qalt/flow_matching.py` contains small three-convolution velocity fields,
fixed Haar analysis, and Heun integration. A full model uses `2*steps` velocity
calls; the hierarchical model uses `2*steps*(levels+1)` calls at different sizes.
These are useful pilot controls, not the requested strong learned-analysis
latent baseline. Calls alone are not a runtime or FLOP comparison.

The source records the covariance failure and the same-array Gaussian-oracle
headroom. They remain negative development findings. There is no reason to
reopen that covariance mechanism or reinterpret it as new confirmation.

## 1. Scalar-spline algebra: valid with important scope conditions

For a conditional density

    q_a(t|c) = sum_v B_v(c) sum_k a_vk H_k(t),

nonnegative partitions of unity and the constraints

    ell <= a_vk <= U,    sum_k w_k a_vk = 1

ensure normalization and bounds. Negative log likelihood is convex in `a`.
The Frank-Wolfe gap certifies empirical global suboptimality only with a valid
linear minimization oracle over the actual feasible set. It does not certify
population error, learned context sufficiency, or an earlier neural optimizer.

For the ordinary open-uniform quadratic B-spline construction, use Greville
abscissae so the context basis reproduces affine functions. Sampling an
arbitrary context location per basis function does not have that property.
For a C2 density with Hessian operator bound H, the response trapezoid
normalizer differs from one by at most H/(12 b^2). Thus `b^2 >= H/6` makes the
normalizer at least 1/2. The supplied conservative approximation constant

    A_s = H(9s+1)/2 + M H/6

and squared-density-to-KL argument are consistent with
`KL <= A_s^2/(ell b^4)`. Boundary knot multiplicities, tensor-product coordinate
count, and row normalization must match this construction.

Inside an interval of width h, endpoint densities a and a_right give slope
`d=(a_right-a)/h`. If v is probability mass measured from its left endpoint,

    t = 2v / (a + sqrt(a^2 + 2dv))

returns DISTANCE from that endpoint, not a unit-bin fraction. The flat-density
case is automatically `v/a`; positive endpoints keep the mathematical
radicand positive, including decreasing pieces. A bin-width error changes the
law even if an implementation still appears monotone. Encode and decode must
use the same canonical endpoint masses. A residual CDF error e implies scalar
inverse error at most e/ell. Finite-precision tail rejection and numerical
residuals must be counted, not clipped away and called exact sampling.

## 2. Independent-array fast excess-risk bound

The convex excess-risk argument survives arbitrary within-array dependence and
misspecification. Let L_a(X) be the average conditional negative log likelihood
across all D scalar factors of ONE array. Let a* minimize its population risk in
the same convex coefficient set. Define r_a=L_a-L_a* and R_a=E r_a. First-order
optimality and curvature of -log give

    R_a >= E[ D^-1 sum_i (q_ai-q_a*i)^2 ] / (2U^2).

Jensen across coordinates and log's Lipschitz constant on [ell,U] give

    E r_a^2 <= (2U^2/ell^2) R_a,
    |r_a| <= log(U/ell).

Bernstein plus a feasible coefficient net therefore supplies the stated kind
of `P log(n)/n` excess bound, with n independent COMPLETE arrays. A feasible
net is required: independently rounding coefficients can violate row
normalization or derivative constraints. For completeness, choose coefficient-net radius ell/(4n). Each log loss
changes by at most 1/(4n), and the net has size at most
`[1+8(U-ell)n/ell]^P`. With V=2U^2/ell^2 and B=log(U/ell), Bernstein and
`sqrt(2 V R t/n) <= R/2 + V t/n` give an estimation coefficient
`2V+4B/3`, plus `2tau+3/(4n)`, after lifting from the net. The supplied
`8V+8B/3` and `1/n` are conservative. A union over G predetermined fits
adds log G to t. The stated conservative kappa is not the problem. The
problem is its size and what comparator it controls.

P must include all separately fitted coefficient groups. Sharing a single
conditional head across sites reduces P but introduces a common-law
approximation restriction. Pooling sites does not replace n by nD. The same
convex excess argument still works for tied heads; its comparator is the best
tied head, not necessarily every true site-conditional law.

## 3. The polynomial dependence gap and finite constants

Let h(x)=[3(2x-1)-(2x-1)^3]/2. Exact rational integration gives

    integral h = 0,     integral h^2 = 17/35.

For `p_rho(x,y)=1+rho h(x)h(y)`, the two marginals are uniform. Hence the best
product density has joint KL equal to their mutual information. Taylor's
integral inequality for `(1+z)log(1+z)-z` proves

    I >= rho^2 (17/35)^2 / [2(1+abs(rho))].

This is a fixed-coordinate PRODUCT-decoder lower bound. It is not a lower
bound for a learned invertible analysis or a globally dependent decoder.
One offending pair gives this JOINT gap, not this gap at every image coordinate.
With D coordinates its normalized contribution is I/D. Repeated disjoint pairs
add only when that repeated target and the relevant decoder restriction are
actually part of the experiment.

Use rho=1/2. Then m=1/2, M=3/2, and H=21rho=10.5 is a conservative valid
Hessian bound: diagonal entries are bounded by 12rho and cross entries by
9rho. For s=1, ell=1/4 and U=3, the supplied constants give

    I_lower = 0.01965986394557823 nats per dependent pair,
    A_s = 55.125,
    kappa = 2310.626417732768.

At n=40,000, delta=.05, one table/catalogue, tau=0 and b=2, P=12. The
estimation term alone is 10.6221707136 nats. Optimizing the displayed total
bound over integer b=2,...,511 gives b=5 and 56.1930088644 nats. Even granting
all other factors for free, it does not certify the pair gap. At n=10^10 the
same integer search gives b=39 and 0.0156769307 nats. This illustrates the
certificate's scale; it is NOT a necessary sample size or a minimax lower
bound on learning this easy target. The same bound is still worse at 300
independent video units. Real head fitting after an internal split has fewer
than 40,000 independent fitting arrays.

## 4. W2 transport: what must be added

A valid learned influence matrix C must be nonnegative, uniformly bound
quantile changes from preceding coordinate changes, and include the context
map's derivatives. The required condition is an operator-norm bound, not a
spectral-radius assertion or a typical training-Jacobian measurement. Every
strictly triangular C has spectral radius zero. A star with eight children and
edge influence .75 has row sums below one but norm `.75 sqrt(8)>1`.

Coupling true and fitted full conditionals by common uniforms gives

    W2(P_Y,Q_Y) <= ||(I-C)^-1||_2 sqrt(KL(P_Y||Q_Y))/(sqrt(2) ell).

Indeed, the local quantile error is bounded by conditional TV/ell, Pinsker
bounds its squared expectation by conditional KL/(2ell^2), the KL chain rule
sums over coordinates, and the influence inequality propagates those local
errors. `||C||_2 <= kappa < 1` yields the stated `1/(1-kappa)` factor. Omitted
context contributes to the full joint KL; it is not removed by this coupling.
Derivative constraints can only retain the original approximation bound when
the approximating target has an appropriate strict feasibility margin.

For X=S(Y), an L-Lipschitz synthesis adds a factor L. KL is invariant under an
invertible chart; W2 is not. In particular Gaussianizing and inverse-Gaussian
CDF charts do not automatically have finite global Lipschitz constants.
Without such a synthesis bound, but when both observation laws live in
[0,1]^D, a separate valid fallback is

    W2(P_X,Q_X)/sqrt(D) <= min(1, (KL(P_X||Q_X)/2)^(1/4)).

This follows from maximal coupling and the cube's squared diameter D. It is
usually weak and does not recover the favorable dimension-normalized bound.
Neither W2 result proves an FID, KID, perceptual-recall, or temporal-quality win.

## 5. The single proof-connected algorithm and its unresolved remainder

Let A be a learned full-dimensional invertible analysis, trained on an internal
discovery part of the existing fit set. All of its coarse/detail laws are
learned. Set Y=Phi(A(X)) coordinatewise. Learn causal feature gates on that same
discovery part, then freeze both A and the gates before fitting response heads
on the remaining independent arrays. A previously trained analysis that used the head-fit arrays cannot be reused
and called independent: retrain it on discovery only for this theorem. A gate
can inspect all allowed earlier coordinates through a masked CNN/attention network; it never sees a future
response while scoring or generating that factor.

For group g, use R nonnegative gates summing to one and b+1 linear response hats:

    q_a,i(t|Y_<i) = sum_r w_eta,i,r(Y_<i) sum_k a_g(i),r,k H_k(t).

Enforce the same bounded, normalized coefficient constraints and fit the heads
by convex NLL to a recorded Frank-Wolfe tolerance. Include a uniform-head
identity option. Draw precisely D independent Gaussian inputs epsilon, convert
them to uniforms, generate every Y_i by the fitted conditional quantile, and
return `A^-1(Phi^-1(Y))`. The full density includes every change-of-variables
term. This is one normalized, full-dimensional, no-VAE, staged learned
transport. The Gaussian source cannot be supplemented with uncounted decoder
noise or observed coarse images at generation time.

Conditional on the frozen discovery output, the argument in section 2 applies
with `P = number_of_groups * R * (b+1)`. This replaces exponential tables by a
finite learned feature dictionary WITHOUT claiming that context approximation
is free. Assuming finite KL to a feasible head, it proves an oracle inequality
for the full generator relative to the best heads for that learned analysis and
feature dictionary. Conditioning then integrating over discovery gives the
same probability guarantee; it does not prove that discovery found a good
representation.

The oracle remainder includes omitted-context conditional mutual information,
site-sharing mismatch, bounded-density/model-support mismatch, and finite-feature
approximation. For a simple selected context C_i, the exact decomposition is

    KL(P_Y||Q) = sum_i I(Y_i;Y_<i | C_i)
                + sum_i E KL(p(Y_i|C_i)||q_i(Y_i|C_i)).

For nonlinear gates the second term also contains their dictionary restriction.
A learned low-dimensional bottleneck can depend globally yet discard the
critical variable. No theorem here gives it the correct unknown context for
free. An observed richer-context improvement falsifies adequacy of a fitted
small model; without additional realizability it is NOT an identified estimate
or an upper bound on the omitted-context mutual information.

Learning the gates jointly with heads on the head-fitting arrays forfeits the
conditional convex guarantee. One needs an independent discovery split or a
uniform bound for the joint neural/context class. Continuous context search
requires its actual covering complexity and optimization error. A public
catalogue of G fixed contexts costs log G, but data-dependent context proposals
are not made public/fixed merely by giving their realized list a name.

For genuinely independent validation of G fitted heads in the SAME frozen
chart, bounded array log loss yields the selection penalty

    2 log(U/ell) sqrt(log(2G/delta)/(2 n_val)).

Different learned charts also require control of their full loss/Jacobian
terms. Repeatedly inspected repair data are not independent fresh validation
for a newly proposed catalogue. All real development conclusions here are
therefore exploratory; that does not prevent using them to stop a method.

The attractive `n^(-4/(s+5))` sieve rate is NOT inherited by an arbitrary neural
feature dictionary. Recovering its approximation proof may require R growing
like b^s, restoring the memory problem. The finite-feature oracle bound and the
unknown approximation remainder are the honest theorem for this algorithm.

## 6. Memory, locality, and measurable assumptions

At b=8 a quadratic-context/linear-response tensor needs `(b+2)^s*(b+1)` raw
coefficients per group: 90, 900, 90,000 and 900,000,000 at s=1,2,4,8. At s=8
this is 3.35 GiB in float32 for ONE group before CDFs, gradients, or optimizer
state. Separate CIFAR-coordinate tables would require about 10.06 TiB of raw
coefficients. Sparsity of evaluation does not eliminate storage: as many as
`2*3^s` density products can be active. Tensor tables over realistic patches
are not a credible competitive architecture without extra low-complexity
structure. Frozen neural gates address storage; they do not prove that such
structure exists in the target.

A model density floor and normalization can be enforced. Target lower bounds,
global Hessian bounds, population context sufficiency, and uniform contraction
cannot be established from a finite set of successful sample/Jacobian checks.
Ordinary uniform dequantization does not itself produce a globally C2 positive
conditional density. Any extra smoothing changes the target and must be common
to all methods. Instrument/noise assumptions, analytical bounds, or explicitly
qualified approximation assumptions are required for a theorem application.

Useful development diagnostics are held-out per-array NLL, larger-context
challenge losses, floor/ceiling saturation, fit-size sensitivity, decode/encode
residuals, tail failures, learned-code dependence, actual conditioning depth,
and measured encoder/head/decode time and peak memory. They can reject a
premise, not prove a global regularity constant. All source arrays, pretrained
features, latent caches, and context-search costs must be available equally to
the relevant control and included in its deployable cost.

## 7. Fair latent baseline and exact-copy boundary

The baseline must learn an invertible analysis A_phi and use a globally
dependent nonlinear stochastic decoder. For example, split the SAME D Gaussian
inputs into coarse and decoder coordinates, learn a coarse latent FM, and use
an invertible conditional global residual transport followed by A_phi^-1.
The residual decoder may use global coupling/attention and all residual noise;
it must not be restricted to independent scalar details. A shared learned
analysis is the cheapest attribution comparison, but a positive result must
also survive an independently optimized analysis with the same data and budget.
Include the zero-velocity/identity-prior case so unnecessary FM is not forced
on an already Gaussianized analysis.

For equality control, simply copy the candidate's analysis, heads, graph, and
computation into this latent decoder. The same Gaussian input must produce
bitwise-identical samples and log densities using the same computation, with
identical inherited cost. This proves no strict separation from the containing
latent class. Only comparisons with explicitly independently fitted resource-
bounded procedures can show an empirical advantage. Do not apply an ambient-D
histogram minimax lower bound to neural FM, diffusion, or this learned decoder.

## 8. Smallest real-data development decision

See `DEVELOPMENT_PROTOCOL.json`. First finish the frozen synthetic correctness
checks. Build the strong baseline and verify its data/prior/copy/resource
contract before allocating a real-data campaign. Then run CIFAR-32 only, not
UCF or a new toy sweep. Preserve the existing 40k fit / 5k seen repair / 5k
excluded-discovery identities and the official-test exclusion. An internal
20k/20k split of the existing fit IDs supplies the intended within-algorithm discovery/head separation. It does
not erase prior adaptive use of these observations in research decisions:
applying a fresh-sample probability statement to this reused campaign would
need a separate justification. These arrays are not newly collected data.

The first one-seed screen is a kill test. Positive development evidence must
survive two predeclared additional training seeds, the independent-analysis
control, a competent full-image FM, and a solver-cost frontier. Inference uses
complete unconditional samples, not observed parent images. Measure all
training stages, not just the final convex solve, and all decoding work, not
just the scalar inverse. FID at 5k is a 5k descriptive statistic, not a 50k
published benchmark reproduction. Bootstrap and seed stability on reused
repair images are descriptive and do not restore confirmation status.

Only a development pass justifies the next UCF development allocation. Preserve
300 training clips, 30 validation clips, and 75 sealed test clips with ORIGINAL
source-group disjointness. If multiple clips share a source group, n is the
number of independent groups, not the clip count. Thirty validation groups
cannot support a strong video-quality claim from a plug-in FVD number alone.
Do not open either official image tests or the 75 video tests as a response to
an inconclusive or failed development result.

## Local deterministic-check provenance

Thirteen independent unit tests passed. An initial inverse fixture evaluated a
right-endpoint mass using a differently rounded polynomial expression, causing
the strict range check to reject it. The fixture now uses the same canonical
trapezoidal endpoint mass. The inverse implementation was not changed to clip
out-of-range probabilities. This was a deterministic test-fixture correction,
not a model fit, experiment, or changed scientific threshold.

## Sources inspected

Pinned repository paths: `research/transport_iteration_20260908/README.md`,
`conditional_learnability.md`, `complete_pilot_record.md`,
`qalt/src/qalt/conditional_cdf.py`, `multiscale_flow.py`, `flow_matching.py`, and
`qalt/experiments/conditional_cdf_validation/PROTOCOL.md` (in the commit diff).
The fuller Pro4 constants audited above were supplied in the user request;
this review did not independently retrieve that private conversation.

Prior architecture attribution: Yu, Derpanis & Brubaker, Wavelet Flow,
arXiv:2010.13821; Durkan et al., Neural Spline Flows, arXiv:1906.04032;
Dao et al., Flow Matching in Latent Space, arXiv:2307.08698. These sources do
not prove novelty, the finite constants here, or a current experimental win.
