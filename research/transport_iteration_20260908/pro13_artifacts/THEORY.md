# Pro13 — declared tent copula, exact inversion, and packed-sign discovery

## Scope and decision

This is one candidate: keep the observed-only split estimator, replace the declared cosine copula by the specified tent copula, and discover its unknown matching using packed signs and exact Hamming counts. The graph algorithm is implemented in Python and in a single-thread compiled popcount kernel. It is a word-parallel computational improvement, not a subquadratic-in-dimension algorithm, new copula-family claim, or global-FM/VAE superiority theorem.

All statements below concern the new law. They do not verify, replace, or regenerate Pro11's pending artifacts. No native data, PSC job, active-branch edit, or FM fit was used here.

## 1. Class and information

Write X=(C,U), with c=192 independent root coordinates and G=4 residual groups of d=720 coordinates. Each group has an arbitrary unknown perfect matching of m=d/2=360 pairs. Thus D=c+Gd=3072 and M=Gm=1440. Given C, all pairs are independent. Within group g every pair has density

    p_theta(u,v | C) = 1 + theta_g(C_1) psi(u) psi(v),  0<u,v<1.

Root densities are unknown histograms with positive bin probabilities on k0=8 equal bins. The C_1 density is at least f_min=1/2. The functions theta_g are unknown, bounded by kappa=.45, L-Lipschitz with L=.5, and satisfy |E theta_g(C_1)| >= a=.35 in the positive class. The graph, signs, root probabilities, and theta functions are not supplied to learners. C_1's index, grouping, chart, psi, all bounds, and bin grids are public to every arm.

There are N=4000 iid *arrays*, not 4000M iid arrays. The first n_s=2000 arrays discover the graph, the last n_r=2000 fit theta. All N arrays fit the roots with add-one/Laplace probabilities. Shared-pair averaging below uses conditional pair independence, not an assumption of marginally independent sites.

## 2. Normalization, symmetry, covariance, and regularity

Let A=sqrt(3/2), knots t=(0,1/8,3/8,5/8,7/8,1), and values A(1,1,-1,-1,1,1). Let psi interpolate linearly. If V is uniform, psi(V) has law

    (1/4) delta_{-A} + (1/2) Uniform[-A,A] + (1/4) delta_A.

Consequently all odd moments vanish, and

    E psi(V)^(2r) = A^(2r) (r+1)/(2r+1), r>=0.

In particular E psi=0, E psi^2=1, E psi^3=0, E psi^4=27/20. The function satisfies psi(1-u)=psi(u); |psi(u)psi(v)|<=b=3/2. The pair density is normalized, has uniform one-coordinate marginals, and lies between 1-b*kappa=.325 and 1+b*kappa=1.675.

For any integrable functions f,h, conditional covariance is

    Cov(f(U),h(V)|C)=theta(C_1) [integral f psi] [integral h psi].

Both f(u)=u-1/2 and f(u)=Phi^{-1}(u) are antisymmetric around 1/2, while psi is symmetric. Hence ordinary and Gaussianized cross-covariance are zero. Nonmatched coordinates are independent conditional on C; each residual marginal is uniform independently of C. Thus all residual off-diagonal ordinary/Gaussianized covariances vanish even though E[psi(U)psi(V)|C]=theta(C_1). This is NOT independence and does not make arbitrary nonlinear observables independent.

**C1 correction.** psi is continuous and Lipschitz, but not differentiable at its internal slope-change knots: at 1/8 its left derivative is 0 and its right derivative is -8A. The conditional pair density is continuous in u,v, not globally C1. Its conditional CDF is C1 in the response coordinate, but the full triangular map is not globally C1 in the parent coordinate. Root histogram densities also jump. The true Lipschitz-theta transport is continuous and bijective, piecewise differentiable/a.e. differentiable; it is not a global C1 diffeomorphism. The *fitted* 32-cell step function theta generally introduces discontinuities across context-cell boundaries, so the learned full transport is only a measurable triangular bijection with a.e. Jacobian, not even a globally continuous homeomorphism. Normalization and the density change of variables hold by conditional substitution. Smoothing theta to restore regularity would define a different estimator and needs a different proof.

## 3. Exact conditional inversion and full-dimensional transport

Let Psi(v)=integral_0^v psi(s)ds. Its knot values are

    A*(0,1/8,1/8,-1/8,-1/8,0).

For alpha=theta(C_1)psi(u), the conditional CDF is H_alpha(v)=v+alpha Psi(v). H_alpha(0)=0, H_alpha(1)=1 and H'_alpha(v)=1+alpha psi(v) in [.325,1.675]. Therefore it has a unique inverse on [0,1]. Four internal-boundary comparisons choose among five segments.

On a selected segment starting at t_j, with psi slope s_j, define

    Delta=q-H_alpha(t_j), d0=1+alpha psi(t_j), beta=alpha s_j.

Then the proposed expression is correct:

    xi = 2 Delta / [ d0 + sqrt(d0^2 + 2 beta Delta) ],  v=t_j+xi.

It includes the limit beta=0, xi=Delta/d0. The discriminant equals the square of the response density at the solution, so it is at least .325^2=.105625; the denominator is at least .65. The segment probabilities on the two ramps are exactly 1/4. The smallest possible segment probability is (1-b*kappa)/8=.040625. There is no iterative inverse or CDF-grid approximation in this ideal map. The vectorized reference code evaluates a square root before selecting its affine result; do not count it as skipping every flat-segment square root in actual compute.

For a Gaussian prior in all D coordinates, use learned root inverse histograms on Phi(Z_root); for each learned matched pair retain U=Phi(Z_i) and set V=H^{-1}_{theta(C_1)psi(U)}(Phi(Z_j)). Fallback groups keep all independent Gaussian-to-uniform coordinates. No Gaussian source coordinate is discarded. Fix one deterministic orientation per recovered edge and a right-cell boundary convention. Root and pair inverses recover every source coordinate, including when the fitted theta is discontinuous in context. The learned density is the product of learned root histogram factors and learned pair factors; all factors remain positive. Gaussianizing observed coordinates multiplies by standard normal Jacobians and leaves KL invariant.

For fixed parent/context, the path H_{t,alpha}(v)=v+t alpha Psi(v) is strictly increasing for 0<=t<=1. Its inverse path obeys dv/dt=-alpha Psi(v)/(1+t alpha psi(v)). This gives an exact conditional flow interpretation. It does not turn the fitted discontinuous-context map into a globally Lipschitz continuous normalizing-flow field.

## 4. Finite precision: supported conclusions and exclusions

The algebraic discriminant/denominator margins rule out cancellation in the rationalized root formula. Neighboring exact pieces agree at boundaries. A rounded comparison near a boundary may choose the adjacent piece; the implementation rejects large segment excursions and explicitly counts any tiny roundoff clamps. No clamp occurred in the recorded checks. Zero theta and flat segments take the affine value.

For an *exact* residual certificate |H_alpha(v_hat)-q|<=epsilon,

    |v_hat-v_exact| <= epsilon / (1-|alpha|A).

If alpha and q are perturbed, add |Delta q| + (3A/16)|Delta alpha| to the residual numerator: max|Psi|=3A/16. These are deterministic inverse-stability bounds. A floating residual computed with the same polynomial is not an outward-rounded certificate. The report's original residual/.325 field is a nominal numerical diagnostic; `precision_addendum.json` uses independently integrated 80-digit polynomials and the actual rounded-alpha minimum density. Neither is an interval proof. A point-location bound alone does not bound a continuous density or KL.

H_alpha(1-v)=1-H_alpha(v). For Gaussianized outputs, use q=Phi(-|z|) and reflect, retaining log q. In the outer affine piece, log v=log Phi(-|z|)-log(1+alpha A), then use an inverse log-normal-CDF routine. The implementation uses log_ndtr/ndtri_exp, not ndtri(exp(log_q)) in an underflowed tail. This handles the recorded +/-1000 scalar tests, with deliberately extreme round-trip error about 1.22e-9; through |z|<=40 the recorded maximum is 7.11e-15. It is not a proof for every floating input.

The full unit-cube decoder rejects Gaussian-CDF saturation or output endpoints; it does NOT silently clip/resample. Its Gaussianized scalar-tail routine demonstrates a tail-safe representation, not a globally tail-safe full-D open-cube decoder. A finite floating grid cannot represent every point in an open cube. A globally robust full Gaussianized root decoder would need the analogous root-tail treatment. No such full-D tail implementation is claimed here.

Floating generated arrays are atomic. The KL theorem is for ideal continuous laws defined by fitted coefficients, NOT continuous-density KL of atomic stored outputs. Root add-one masses are rational counts/(N+k0); the stored doubles permit exact recovery of the original integer counts in all six cases, checked separately. Rounded probability sums have ordinary floating error, not exact symbolic normalization.

## 5. Verify the proposed KL and concentration constants

Let Z=psi(U)psi(V) under uniform independent U,V, and

    h(t)=E[(1+tZ) log(1+tZ)].

Z has a symmetric distribution, E Z^2=1, and |Z|<=b. Thus

    h''(t)=E[Z^2/(1+tZ)]=E[Z^2/(1-t^2 Z^2)]
    1 <= h''(t) <= 1/(1-kappa^2 b^2).

The forward KL is its Bregman divergence:

    KL(p_theta || p_eta)=h(theta)-h(eta)-h'(eta)(theta-eta).

Therefore

    (theta-eta)^2/2 <= KL(p_theta || p_eta)
                       <= B_kappa (theta-eta)^2,
    B_kappa=1/[2(1-kappa^2 b^2)]=0.918484500574053.

All proposed constants are valid for this declared law. The original real-valued graph statistic Z has E[Z|C]=theta(C_1), E[Z^2|C]=1 because integral psi^3=0, unconditional variance at most 1, and centered magnitude at most b+kappa=1.95. Bernstein with this centered bound is valid. A bad learned matching/fallback configuration has conditional joint KL bounded by

    J = M log[(1+b*kappa)/(1-b*kappa)] = 2361.230297178379,

since the true density is at most (1+b*kappa)^M and any clipped fitted density is at least (1-b*kappa)^M (a fallback factor is 1). This is a coarse upper envelope, not the typical error of one wrong edge.

For a product law in this fixed chart, even allowing arbitrary root dependence and context-dependent one-coordinate marginals, the best residual factors are uniform. Its KL is at least

    sum_g m E h(theta_g(C_1)) >= M a^2/2 = 88.2.

This lower bound does not apply to analytic pair models, arbitrary flows, learned charts, or copied candidates.

## 6. One lower-cost discovery algorithm: exact packed-sign correlations

Let s(u)=sign psi(u), with a fixed positive convention at zero. Under the continuous law the zero points have probability zero. Then s is balanced Rademacher, integral s psi=integral|psi|=3A/4, and for a true edge

    E[s(U)s(V)] = r E theta(C_1),  r=(3A/4)^2=27/32.

For a nonedge it is zero. The minimum true absolute signal is gamma=r*a=.2953125. Use threshold tau=.159 on the empirical sign-product mean, computed EXACTLY as (n_s-2*HammingDistance)/n_s from observed signs. This is not a random-projection approximation. Accept a group only if every vertex has degree exactly one in the threshold graph; otherwise fall back for the entire group. No oracle graph repair or forced best matching is allowed.

There are H=G*d*(d-1)/2=1,035,360 unordered tested pairs. A union bound and the one-sided Hoeffding inequality for [-1,1] variables give

    delta_graph <= 2(H-M) exp(-n_s*tau^2/2)
                   + M exp(-n_s*(gamma-tau)^2/2)
                = 3.394869410068532e-5.

The two terms are 2.1682938350123623e-5 and 1.2265755750561697e-5. All rows are iid; dependence across tested edges is harmless for this union bound. Threshold .159 was fixed from the public constants before local fits, not selected on observed recovery results.

Packing costs O(n_s Gd) elementary work and 720,000 raw bytes, or 737,280 bytes after uint64 column padding. Exhaustive comparison uses H*ceil(n_s/64)=33,131,520 XOR/popcount word visits, plus accumulation/threshold work. A full GEMM Gram uses 4,147,200,000 multiply-adds. A symmetry-aware Gram still uses 2,076,480,000 multiply-adds including diagonals. These different operation types must NOT be interpreted as a 125x or 63x latency guarantee. Complexity remains quadratic in d. CPython integers use 30-bit digits on the recorded platform, not 64-bit words; its separate cost record is retained.

The sign statistic loses signal. Under the same new law, the dense-psi Bernstein bound gives delta_dense=2.367236698632463e-6 and the tighter joint bound 0.4864887381938313 below. Packed discovery is a computation/statistical-signal tradeoff, not a uniformly better estimator.

## 7. Explicit finite-N learning theorem

Condition on correct graph recovery. For each regression array i and group g, form

    T_ig=(1/m) sum_pairs psi(U_ij)psi(V_ij).

Conditional on C_i, E T_ig=theta_g(C_i1), Var(T_ig|C_i)<=1/m. Use K=32 equal context cells, the within-cell sample mean clipped to [-kappa,kappa], and zero for an empty cell.

The 32-cell partition refines the 8-bin root partition, so C_1 conditional on any cell is uniform. Hence Var(theta(C_1)|cell)<=L^2/(12K^2)=A_K, by the independent-copy variance identity and Lipschitz continuity. Without alignment/within-cell uniformity, this 1/12 constant is not justified. A generic interval bound is L^2/(4K^2).

Each cell has probability p_b>=f_min/K. For a binomial count N_b,

    E[1{N_b>0}/N_b] <= 2/[(n_r+1)p_b],
    P(N_b=0) <= exp(-n_r p_b).

Decomposing within-cell approximation, regression variance and empty cells, and using contraction under clipping, yields for each group

    R_theta := E_train,C (theta(C_1)-theta_hat(C_1))^2
      <= A_K + [2K/(n_r+1)](1/m + A_K)
             + kappa^2 exp(-n_r f_min/K)
       = 0.00010984023505283588.

The graph-good event depends only on the first 2000 arrays, so the regression risk calculation is unchanged after conditioning on that event. Root fitting may use all 4000 arrays; no independence from graph fitting is needed for the following additive expected-risk bound.

For each add-one k0-category root histogram, Jensen and the binomial inverse-count identity give

    E KL(p_root || p_hat_root) <= log[(N+k0)/(N+1)].

Summing c independent-root factors and using the graph-error envelope gives

    E_train KL(P || Q_hat)
      <= c log[(N+k0)/(N+1)] + M B_kappa R_theta + J delta_graph
       = 0.33562251023339346 + 0.14527663694703438 + 0.08016068506017908
       = 0.5610598322406068 nats / complete D3072 array.

A safe rounded statement is <0.562 nats/array. This is an expectation over fitted samples, not a uniform per-fit, per-context, or high-probability quality certificate. The difference from Pro11's reported cosine bound is not a head-to-head improvement: the law and discovery statistic changed.

## 8. Masked observable prediction

For any fixed mask O,H (or a mask independent of the array and training data), the exact chain rule gives

    E_{P_O} KL(P_{H|O} || Q_hat_{H|O}) <= KL(P || Q_hat).

Thus the training-averaged masked conditional KL is at most the joint bound. For an observable h in [0,1], integrated squared conditional-mean error is at most joint_KL/2; for h in [-1,1], at most 2*joint_KL, by Pinsker. This statement requires the actual Q_hat conditional, not an arbitrary clamped sampler.

A more useful bound uses the structure. Require C_1 observed and masks independent of values. For a hidden member V of a pair with U observed, put h(V)=psi(V)/A. Its Bayes mean is

    E[h(V)|C,U] = theta(C_1) psi(U)/A.

If both pair members are hidden, the individual h mean is zero. On graph recovery, the integrated squared prediction error is at most (1/A^2)R_theta=(2/3)R_theta, uniformly over fixed coordinate masks. On graph error both true and fitted conditional means lie in [-kappa,kappa], giving error at most 4*kappa^2. Therefore each hidden residual coordinate, or their fixed average, has

    E excess squared prediction loss
      <= (2/3)R_theta + 4*kappa^2 delta_graph
       = 0.00010072526559011236.

The experiment's random half-mask can gain a further factor 1/2 in the good-graph term; the stated bound is conservative. An observable event is also available: Pr(psi(V)>=0|C,U)=1/2+(3A/8)theta(C_1)psi(U). This gives a corresponding binary-Brier mean-error bound by the same calculation.

These are observable-coordinate nonlinear predictions, not semantic labels. Ordinary U-mean and Gaussianized-coordinate mean are constant for this family and do not benefit from dependence learning. For masks hiding C_1, the general KL chain rule still holds, but obtaining the correct conditional requires context integration/posterior inference; it is not covered by the simple pairwise prediction formula or its inference-cost measurement.

## 9. Zero-mean conditional-dependence falsifier

For an arbitrary nonuniform root histogram define, in the *fixture only*,

    theta_g(c) = [.5/(2pi)] [sin(2pi c+phase_g) - E sin(2pi C_1+phase_g)].

It has Lipschitz constant <=.5 and absolute value <=1/(2pi)<.45, and exactly zero mean. Its conditional dependence is nonzero. Both unconditional real-psi and packed-sign graph statistics have zero population signal. The probability that any packed threshold edge appears is at most

    2H exp(-n_s*tau^2/2).

With high probability all groups fall back, incurring conditional KL sum_g m E h(theta_g(C_1)) >= (m/2)sum_g E theta_g(C_1)^2. A raw uncentered sine under a nonuniform C_1 distribution does NOT establish this failure. A sin mean computed from hidden fixture probabilities is allowed only in data generation, never in learning.

Observed C_1 still makes conditional discovery possible for other estimators; this is a failure of the chosen unconditional discovery statistic, not a general identifiability impossibility. No zero-mean rescue is proposed in this package.

The six new fabricated cases recovered all 1440 pairs in each positive fit and zero pairs/all four fallbacks in each centered-zero-mean fit. The failure is retained and is outside the positive theorem class, not a counterexample to its assumptions.

## 10. Obstruction to native-image/video claims

The useful theorem assumes a known informative scalar feature, known context/grouping, a sparse disjoint-pair graph, uniform residual marginals, and conditional parameter sharing. Real-image/video use would need an observed-only, independently frozen representation establishing approximation error for those assumptions and charging its discovery/training cost. This package supplies no such representation theorem or native evidence. Pairwise zero covariance and a successful masked nonlinear-coordinate score do not establish object identity, temporal coherence, or semantic usefulness. Quantized native observations may also put nonzero mass on feature-zero/tie boundaries.

Known piecewise-polynomial coupling transforms already provide analytic inverses; the inspected repository already implements the same rationalized quadratic formula. An analytic pair decoder can use this very discovery routine and inverse, and an exact stochastic-latent copy must tie. The defensible new result here is the finite-learning/computational construction and its falsifier, not a universal advantage over an equally informed model class.
