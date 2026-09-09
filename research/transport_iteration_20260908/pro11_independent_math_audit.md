> Archival independent derivation written before source delivery. The subsequently inspected fixture explicitly sets C_1 uniform, and its sampler uses a continuous CDF interpolant. Those two pending dependencies below are resolved for the delivered fixture in pro11_independent_artifact_audit.md; generic qualifications still apply. No saved-state empirical verification has occurred.

# Independent audit: Pro11 unknown matching and context class

This is a proof and ordinary scalar arithmetic audit of the supplied formulas. It has not inspected delivered artifacts, fitted a model, accessed observations, verified empirical numbers, or run a PSC job. The independent arithmetic script and JSON are `work/pro11_matching_bound_check.py/.json`.

## Precise sufficient data assumptions

There are192 observed root coordinates C and four observed residual blocks with720 coordinates each. Root coordinates are independent, each with an unknown categorical histogram on eight *known aligned equal-width* subintervals of[0,1], with density uniform inside its cells. The density of the first coordinate C_1 is at least1/2. The32 equal-width regression bins refine those eight histogram cells, so C_1 conditional on any regression bin is uniform. These facts are stronger than a density lower bound alone.

Each residual block has an unknown fixed perfect matching, with360 pairs. Conditional on C, all pairs are independent (both within and across blocks). In blockg, each pair has density

p_theta(u,v)=1+theta_g(C_1) psi(u) psi(v), psi(u)=sqrt2 cos(2pi u).

All360 pairs in a block share theta_g. Assume |theta_g|<=kappa=.45, Lipschitz constant<=L=.5, and |E theta_g(C_1)|>=a=.35 for *each* block. Pair marginals are uniform and independent of C; this supplies the null cross-edge moments. The graph is fixed across histories. Arbitrary conditional residual distributions or graphs changing with C are not covered.

Use2,000 whole independent arrays for graph estimation and a separate2,000 arrays for conditional function estimation. Within-array pairs are conditionally independent, not unconditionally independent: their shared context remains a cluster effect. All4,000 arrays may estimate the root histograms. The learner observes C and residual coordinates and knows the declared coordinate/block partition and feature dictionary. This structural information must be available to every comparator; it is not a learned semantic representation guarantee.

## Structure recovery and the split

For an edge(i,j), define X=psi(R_i)psi(R_j). Its range is[-2,2]. For a true edge, E[X|C]=theta_g(C_1); for a false edge, E[X|C]=0 because coordinates belong to different conditionally independent pairs with zero feature means. In both cases E X^2=1: for true edges the extra term contains the zero third cosine moment, and for false edges the pair marginals have unit second feature moment. Thus Var X<=1 and |X-E X|<=2+kappa.

Bernstein with thresholdt=a/2 yields

P(|sample_mean-E X|>=a/2) <=2 exp[-n_s a^2/{8(1+(2+kappa)a/6)}].

There are E=4*choose(720,2)=1,035,360 edges. A union bound establishes the stated delta_s. It does not assume edges are independent. On its complementary strict event, thresholding absolute empirical moments at a/2 retains every true edge and rejects every false edge. Every block then has exactly degreeone, hence the correct matching. Boundary equality has probabilityzero for these continuous nondegenerate samples; alternatively define the good event with strict inequalities. The graph acceptance rule must reject any non-perfect matching; it may still accept a wrong perfect matching on the failure event.

The graph-success event depends only on the first2,000 arrays. Conditional on any successful structure-training outcome, the second2,000 arrays still have their original independent-array law. This justifies applying the regression calculation on the successful event and multiplying its bound by a probability at mostone. Conditioning on a graph selected using those same regression arrays would invalidate that argument without an additional uniform-selection proof. Using all arrays for the separate root estimator causes no problem: add its unconditional expected KL bound by linearity rather than assert independence from graph success.

## Conditional regression with random bin counts

In a function-training array, average X over the m=360 true pairs of one block. Given C, this response has mean theta_g(C_1) and conditional variance(1-theta_g(C_1)^2)/m<=1/m. Do not replace2,000 arrays by720,000 independent context observations.

Let bin probabilities be p_k. The density lower bound gives p_k>=p_min=.5/K. Because the aligned histogram assumption gives a uniform C_1 within each bin,

Var(theta_g(C_1)|bin k) <=L^2 Var(C_1|bin k)=L^2/(12K^2)=v.

The Lipschitz variance inequality follows by an independent-copy argument. If density is merely bounded below but not uniform within the32 cells, the safe generic range bound is v=L^2/(4K^2); the claimed1/12 constant does not follow.

Conditional on bin count N_k>0, average the pooled responses, then clip to[-kappa,kappa]. Clipping cannot increase squared error to the true theta_g(C_1). The integrated bin error is at most

v+(1/m+v)/N_k.

For an empty bin the zero estimate has error at most kappa^2. After multiplying by m and summing G=4 blocks, the expected total squared parameter error is bounded by

M v+(G+M v) A_n+M kappa^2 exp(-n p_min).

Here M=1440 and the harmonic-binomial bound is valid. For integer N>=1,

1/N <=1/(N+1)+3/[(N+1)(N+2)].

The usual binomial reciprocal expectations give

E[1_{N>0}/N] <=1/[(n+1)p]+3/[(n+1)(n+2)p^2].

Multiply by p_k, sum over K bins and use p_k>=p_min to obtain exactly

A_n=K/(n+1) [1+3/((n+2)p_min)].

The empty-bin contribution uses sum_k p_k(1-p_k)^n<=exp(-n p_min). These steps explicitly handle random counts; they do not assume the observed bin counts equal n/K or that the root distribution is globally uniform.

## Converting parameter error to forward KL

Write h(u,v)=psi(u)psi(v). Under uniform measure, h is symmetric, |h|<=2, E h=0 and E h^2=1. For the negative-entropy function F(theta)=integral(1+theta h)log(1+theta h),

F''(theta)=integral h^2/(1+theta h)=integral h^2/(1-theta^2 h^2).

Therefore1<=F''(theta)<=1/(1-4kappa^2). Forward KL(p_theta||p_eta) is the Bregman remainder F(theta)-F(eta)-F'(eta)(theta-eta), hence

(theta-eta)^2/2 <=KL(p_theta||p_eta)<=B_kappa(theta-eta)^2,

B_kappa=1/[2(1-4kappa^2)].

Apply this conditional on the observed context and sum over independent pairs. This establishes the regression term of the proposed KL bound.

On a structure-failure event the accepted matching may be wrong, but every fitted pair factor remains in[1-2kappa,1+2kappa]. Comparing any true product of M pair factors to any accepted fitted matching, or to the product fallback, gives pointwise log density ratio <=M log[(1+2kappa)/(1-2kappa)]. Thus the bad-event contribution is at mostdelta_s times that constant. No claim that every bad graph is detected is required.

For a root histogram with r=8 cells and N=4,000 arrays, add-one probabilities have expected forward KL <=(r-1)/(N+1). One proof bounds KL by chi-square and uses E[1/(count+1)] <=1/[(N+1)p]. Multiply by192 roots. The root density must truly be constant inside the modeled bins; an unknown within-bin shape adds approximation error absent from the displayed theorem.

## The grid term needs an actual density construction

For one pair, first generate U uniformly and then generate V conditionally. The exact conditional CDF is

F(v|u)=v+theta psi(u) sqrt2 sin(2pi v)/(2pi).

It is strictly increasing, with density between1-2kappa and1+2kappa. A *continuous piecewise-linear CDF interpolant* on a uniform grid of J cells defines a normalized density equal to the exact average conditional density in each cell. The conditional density's derivative in v is bounded by4pi kappa. The maximum deviation between its value and its cell average is at most2pi kappa/J. Since both densities are at least1-2kappa, the log-density difference is at most

2pi kappa/[J(1-2kappa)].

This establishes the added M-pair term, provided the implemented sampler inverts this continuously interpolated CDF, and the density in the theorem is the same interpolant. It is NOT a consequence of coordinate error alone. Returning bisection midpoints or rounding the inverse to grid points gives a discrete numerical law; mathematical forward KL from a continuous true density to that law is infinite. Ordinary floating implementations of continuous models conventionally approximate their ideal maps, but cannot turn a bisection point-error estimate into this density-KL proof. Delivered sampler and likelihood code must be inspected before applying the grid term. Exact real-arithmetic CDF values at knots are assumed by this interpolation bound; finite sin evaluations require a separate numerical-error analysis if claimed certified.

## Independent arithmetic and comparison limits

The supplied formula evaluates in ordinary arithmetic to0.61910165641061, agreeing with0.619102. Components are:

| Component | Expected joint KL bound |
|---|---:|
|Root estimation|0.335916020995|
|Within-bin context approximation|0.077097039474|
|Function-estimation variance|0.185832260667|
|Empty bins|2.06e-11|
|Structure failure|0.020246855546|
|Continuous CDF interpolation, J=2^32|0.000009479709|

The structure failure bound isdelta_s=4.7752106431e-6. If within-bin uniformity is removed and the generic1/4 variance constant is used, the bound rises to0.775998094941. These are numerical evaluations of symbolic bounds, not interval-certified decimals.

For the restricted product residual baseline with the same fixed observed coordinates and root information, uniform true scalar marginals imply its optimal conditional scalar-product density is uniform. Conditional pair KL is at leasttheta_g(C_1)^2/2. Jensen and |E theta_g|>=a give the floor M a^2/2=88.2. This separates that ablation; a stochastic conditional copy of the proposed pair model ties it. General global FM or RQS decoders are not excluded from learning the same pair dependence. An unrestricted learned analysis could also absorb the matching/coupling, so the comparison must fix or equally charge that analysis.

No uniform separation from a context-*constant pair* baseline follows from the assumptions: Lipschitz<=.5 permits constant theta. To prove such a separation one needs a positive lower bound on Var(theta_g(C_1)), in which case the same Bregman lower bound applies to the best constant parameter. The supplied empirical~0.21 vs~3.08 figures have not been verified here.

Removing the nonzero intercept can make E theta_g=0 while conditional dependence remains. Then the signed unconditional edge statistic is not identifiable by the registered threshold, and the |E theta_g|>=.35 assumption fails. Product fallback and a substantial adverse KL are plausible, but the reported2.028 is unverified. Such a failure must be retained, not reported as a contradiction of this explicitly restricted theorem or repaired without a new registration.

Generation uses the complete source dimension and can process a known matching in linear dimension time apart from scalar inversion/grid search. Learning the unknown matching requires O(n_s G*720^2) Gram-matrix work (about2.07billion unique-edge sample products) and must be charged. The theorem does not prove lower whole-system complexity than latent diffusion/FM, a learned summary guarantee for images/videos, or superior realistic generative performance.

## Additional pending fixture membership check

An unknown nonuniform histogram for C_1 need not have E sin(2pi C_1)=0. Consequently, the positive fixture theta=.35+.075 sin(2pi C_1) need not satisfy E theta>=.35, and removing the .35 intercept need not cancel the unconditional structure signal. The delivered fixture must either establish a uniform C_1, explicitly center its function under the actual histogram with appropriate amplitude/range/mean checks, or compute and prove the required mean bound directly from the histogram masses. None of these conditions has yet been assumed or verified here. The empirical positive/failure labels cannot establish class membership by themselves.
