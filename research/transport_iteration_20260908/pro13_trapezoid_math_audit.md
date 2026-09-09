# Trapezoid-feature variant: independent constructive audit

This is a new declared data/model law, not an acceleration that preserves the earlier cosine law. It uses the same rank-one separable copula construction with a different public feature. No novelty of the copula family, universal representation advantage, latent-FM dominance, fitting result, or native image/video claim is made. Only exact symbolic moments, ordinary bound arithmetic and fabricated scalar checks were performed. No fits or PSC work occurred.

## Feature and dependence structure

Let A=sqrt(3/2), knots t=(0,1/8,3/8,5/8,7/8,1), and heights a=A(1,1,-1,-1,1,1). Interpolate psi linearly. Its distribution under uniform input is symmetric: a shift by1/2 moduloone flips its sign. Also psi(1-u)=psi(u). Direct exact polynomial integration gives, before multiplying by powers of A,

integral psi/A=0; integral(psi/A)^2=2/3; integral(psi/A)^3=0; integral(psi/A)^4=3/5.

Thus integral psi=0, integral psi^2=1 and integral psi^3=0; |psi|<=A and |psi(u)psi(v)|<=A^2=3/2. These assertions are independently reproduced using rational polynomial arithmetic in `work/pro13_trapezoid_bound_check.py/.json`.

For p_theta(u,v)=1+theta psi(u)psi(v), |theta|<=.45 gives density in[.325,1.675]. Both marginal densities are uniform. Conditional on root C, retain the same unknown perfect matching and shared per-block theta_g(C_1) assumptions as Pro11: four blocks,360 pairs/block, conditional independence of pairs, theta Lipschitz<=.5, |E theta_g|>=.35, and the same observed coordinate/feature information for every comparator.

## Direct forward/inverse map and determinant

Let Psi(v)=integral_0^v psi(s)ds. At the six knots,

Psi(t)=A(0,1/8,1/8,-1/8,-1/8,0).

Given u and theta, set b=theta psi(u). The conditional CDF is F(v)=v+b Psi(v), with F(0)=0,F(1)=1 and derivative1+b psi(v) in[.325,1.675]. This CDF is globally C1 in v, with a piecewise-quadratic interior. It is not generally C2 at the knots.

Its six output knots are q_i=t_i+b Psi(t_i), strictly increasing. Select the one of five segments containing a uniform source p. For that segment let Delta_i=t_{i+1}-t_i, s_i=(a_{i+1}-a_i)/Delta_i, d_i=1+b a_i, beta_i=b s_i, and q=p-q_i. Then the inverse offset is

delta=2q/[d_i+sqrt(d_i^2+2 beta_i q)], and v=t_i+delta.

This formula is valid for both ramps and flat segments. On flat segments beta=0, giving delta=q/d_i without any division by a vanishing beta. At q=0 the same expression giveszero. The discriminant equals (1+b psi(v))^2 and is at least .325^2 in exact arithmetic. There is one square root and a fixed five-segment selection, not a32-step CDF search or a large lookup table. The maximum |beta| on a ramp is5.4. Density positivity bounds inverse slope between1/1.675 and1/.325.

For independent Gaussian sources z_1,z_2, take u=Phi(z_1), p=Phi(z_2), v=F^(-1)(p|u,theta(C_1)). The inverse is z_1=Phi^(-1)(u), z_2=Phi^(-1)(F(v|u,theta(C_1))). The full pair map is triangular, retaining both Gaussian coordinates. Its forward log absolute determinant is

log phi(z_1)+log phi(z_2)-log[1+theta psi(u)psi(v)].

The inverse determinant is the negative of this expression at the recovered source. For Gaussianized observed outputs x=Phi^(-1)(u), y=Phi^(-1)(v), x=z_1 and the forward log determinant simplifies to log phi(z_2)-log phi(y)-log[1+theta psi(Phi(x))psi(Phi(y))]. Known-root histogram inverse maps can be composed with all192 remaining source coordinates, charging their likelihood/determinants separately.

The scalar conditional CDF's C1 statement does not imply the entire map is C1 in u or root context: psi' jumps at its knots, and theta is only assumed Lipschitz. A.e. derivatives and triangular change-of-variable determinants remain appropriate. Do not advertise global smoothness stronger than these assumptions.

## Zero covariance and what masked prediction can miss

Symmetry about1/2 gives integral (u-1/2)psi(u)du=0. Hence pair covariance of ordinary unit-interval coordinates iszero for every theta, despite the nonlinear dependence whenever theta!=0. The stronger identity is E[V|U=u,C]=1/2. It follows that optimal squared-error prediction of any residual coordinate from all other observed coordinates and root context is the same constant1/2; marginalizing unavailable context preserves this identity. This statement concerns residual targets, not prediction of root C_1 from residual evidence.

Similarly, Phi^(-1)(u) is odd about1/2, whereas psi is even. Therefore integral Phi^(-1)(u)psi(u)du=0. Gaussianized residuals have standard Gaussian marginals, zero off-diagonal covariance and conditional meanzero, but their joint density is non-Gaussian when theta!=0. An ordinary covariance diagnostic or Gaussianized squared-error masked predictor can completely miss this class.

Nonlinear feature prediction detects it: E[psi(V)|U=u,C]=theta(C_1)psi(u), so E[psi(U)psi(V)|C]=theta(C_1). Higher moments also change. Exact integration gives integral u^2 psi(u)du=11A/192, hence E[V^2|U=u,C]=1/3+theta(C_1)psi(u)11A/192. No claim of equal conditional distributions follows from equal conditional means.

## Finite-sample KL bound for the new law

All Pro11 random-count and independent-array split arguments remain valid, with the same2000 structure /2000 function arrays, all4000 root estimation arrays, K=32 aligned context bins and p_min=.5/K. Shared360-pair responses have conditional variance<=1/360 because integral psi^2=1 and integral psi^3=0. False-edge second moments are likewiseone.

Only the range-dependent constants improve. Edge features lie in[-1.5,1.5], so centered Bernstein increments are bounded by1.5+kappa. With E=1,035,360 edges,

delta_s=min(1,2E exp[-n_s a^2/{8(1+(1.5+kappa)a/6)}]).

The symmetric-feature entropy Hessian obeys

1<=F''(theta)<=1/[1-kappa^2(1.5)^2],

so B_kappa=1/{2[1-kappa^2(1.5)^2]}=0.918484500574... . The root, context approximation, harmonic random-bin-count and empty-bin terms are otherwise unchanged. An arbitrary wrong accepted matching or fallback has conditional log-ratio at most M log[(1+1.5kappa)/(1-1.5kappa)]. There is no grid term for the exact real-arithmetic quadratic inverse just derived.

Using v=L^2/(12K^2) and A_n=K/(n+1)[1+3/((n+2)p_min)], the expected full joint forward-KL upper bound is

192*7/4001 +B_kappa[Mv+(G+Mv)A_n+M kappa^2 exp(-n p_min)]
 +delta_s M log[(1+1.5kappa)/(1-1.5kappa)].

Independent ordinary arithmetic yields:

| Term | Bound |
|---|---:|
|Root learning|0.335916020995|
|Context approximation|0.026908725603|
|Function sampling|0.064859939429|
|Empty bins|7.18e-12|
|Structure failure|0.005589591013|
|Total|0.433274277047|

Here delta_s=2.3672366986e-6. The restricted independent-coordinate residual baseline still has population KL floor M a^2/2=88.2 because the Hessian lower bound remainsone. These decimal evaluations are not interval certificates, and the comparison still fixes the same coordinate partition/structural information. An expressive conditional stochastic-copy baseline ties the construction. A context-constant pair learner is not uniformly separated without a lower bound on context variation.

## Fixture membership and numerical hazards

The root assumption needed for v's1/12 constant is a known aligned eight-bin histogram family with uniform density inside each bin, refined by the32 regression bins. Density>=.5 alone is insufficient. Neither a nonuniform histogram nor its lower bound implies E sin(2pi C_1)=0. A positive theta=.35+.075sin fixture meets the mean condition if C_1 is uniform; otherwise calculate its actual mean, range and Lipschitz constant from the declared root masses before claiming membership. Its zero-intercept counterpart need not have cancelled unconditional signal under a nonuniform root. A centered function must also have its amplitude/range checked after centering. The additive warning has been recorded in the earlier Pro11 audit.

For the direct inverse, validate monotonically increasing finite CDF knots, positive discriminants and recovered offsets within their own segments. Never silently clip a negative discriminant or output to force a pass. Use the rationalized quadratic form above, including beta=0, and compute log density with log1p(theta psi(u)psi(v)). Endpoint selection at exact ties must be consistent; the exact map is continuous across segment boundaries.

Extreme Gaussian Phi values can round tozero/one. Reject unsupported saturation or use an explicitly tail-stable implementation; do not clip source coordinates or claim arbitrary-tail correctness from central tests. On the first/last plateau, the exact relations F(v)=(1+bA)v and 1-F(v)=(1+bA)(1-v) allow stable log-tail calculations followed by inverse-normal log-tail evaluation for Gaussianized outputs. This can preserve extreme-tail information, but needs its own implementation audit. The exact real-arithmetic KL bound does not by itself certify the distribution represented by finite-precision output rounding.

The independent scalar checker `work/pro13_trapezoid_scalar_check.py/.json` verifies nonidentity Gaussian-source roundtrip to6.67e-16, finite-difference pair logdet to3.32e-10 and boundary inverse-CDF consistency to1.12e-16 on fabricated cases. It is not an interval or exhaustive numerical certificate.

Replacing the cosine feature removes iterative conditional inversion for this new law but leaves structure-fitting work O(n_s G*720^2) and root/function learning costs. These must be charged in any future end-to-end comparison. The direct inverse is a useful tractable positive class, not a proof of superior realistic image/video model efficiency.
