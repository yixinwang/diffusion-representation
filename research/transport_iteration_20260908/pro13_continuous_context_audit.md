# Continuous context-interpolation variant: independent bound

This is a distinct declared variant of the trapezoid model. Preserve the earlier piecewise-constant context estimator, scalar prototype and0.4332742770 theorem unchanged. The new evaluation rule linearly interpolates the same32 fitted clipped context-bin estimates at bin centers, using constant extrapolation between the first/last center and the boundary. It uses no additional fit or observations. No fits, prototype edits or PSC work were performed for this note.

Let h=1/K and centers c_j=(j+1/2)h. For c in[c_j,c_{j+1}], set r=(c-c_j)/h and

theta_tilde(c)=(1-r)theta_hat_j+r theta_hat_{j+1}.

Outside the extreme centers use the nearest fitted estimate. All estimates are clipped to[-kappa,kappa] and an empty-bin estimate iszero. Therefore the interpolation is continuous, bounded bykappa, and piecewise linear. No value clipping at generation is needed.

## Proof of the proposed regression bound

Assume the original aligned root-histogram condition: C_1 is uniform inside each32-bin regression cell and its cell probability p_j is at least p_min=.5/K. Let mu_j=E[theta(C_1)|cell j]. Lipschitzness gives, for any evaluation contextc,

|mu_j-theta(c)|<=L E[|C_1-c||cell j]<=L(|c-c_j|+h/4).

Each whole-array pooled pair-feature response has conditional variance at most1/m given the exact context; its variance within a bin is at most1/m+v, where v=L^2/(12K^2). Given a nonzero count N_j, the raw bin mean is unbiased for mu_j and has variance at most(1/m+v)/N_j. Clipping cannot increase its squared distance to the true theta(c), which is in[-kappa,kappa]. Thus its mean-square error relative to theta(c), on N_j>0, is at most this variance plus L^2(|c-c_j|+h/4)^2. For N_j=0, the zero estimate has error at mostkappa^2.

Weighted Jensen applies pointwise to the interpolated errors even though neighboring counts and estimates are dependent. There is no assumption that the two bin estimates are independent. Between centers, the weighted squared center distances and distances satisfy

(1-r)|c-c_j|^2+r|c-c_{j+1}|^2=h^2 r(1-r)<=h^2/4,

(1-r)|c-c_j|+r|c-c_{j+1}|=2h r(1-r)<=h/2.

Consequently the weighted squared bias bound is

L^2[h^2/4+(h/2)(h/2)+h^2/16]=9L^2 h^2/16.

In the two endpoint extrapolation regions, distance to the retained center is at mosth/2, yielding the same bound L^2(h/2+h/4)^2. This proves the proposed uniform approximation term. It is deliberately conservative; no tighter constant is needed here.

For every cell, the same binomial reciprocal inequality used before gives

E[1_{N_j>0}/N_j]<=1/[(n+1)p_j]+3/[(n+1)(n+2)p_j^2]
 <=H=1/[(n+1)p_min] [1+3/((n+2)p_min)].

Unlike the previous piecewise-constant estimator proof, the interpolation weights do not match the training-bin masses, so one cannot simply reuse A_n after averaging. The uniform H bound is valid for every evaluation context and every interpolating weight. For p_min=.5/K, H=2A_n exactly. The weighted empty-bin probability is at mostexp(-n p_min), since interpolation weights sum toone.

Summing over G blocks and M=Gm pairs gives the claimed total expected squared parameter error:

M*9L^2/(16K^2)+(G+Mv)H+M kappa^2 exp(-n p_min).

This expectation concerns independent new evaluation arrays and the independently reserved function-fitting half. The original structure-success conditioning and root-learning arguments still apply unchanged. Interpolation keeps every fitted conditional factor bounded within[1-1.5kappa,1+1.5kappa], so the bad-graph contribution remains valid too.

## Numerical bound

With kappa=.45,L=.5,K=32,n=2000,p_min=.5/32,G=4,M=1440, the same trapezoid constant B=1/[2(1-2.25kappa^2)] applies. Add the same192*7/4001 root term and the same structure-failure term. Independent ordinary arithmetic gives:

| Term | Bound |
|---|---:|
|Root learning|0.335916020995|
|Continuous interpolation bias|0.181633897819|
|Function sampling|0.129719878858|
|Empty cells|7.18e-12|
|Structure failure|0.005589591013|
|Total expected full joint KL|0.652859388692|

The restricted independent-coordinate residual baseline floor remains88.2. There is no CDF discretization term for the ideal direct quadratic inverse. Arithmetic code/results are `work/pro13_continuous_context_bound.py/.json`; the displayed decimals are ordinary numerical evaluations of a symbolic bound, not interval-certified endpoints.

## Continuity of the complete map

A fitted add-one root histogram has strictly positive bin densities. Its CDF is continuous and strictly increasing, and its inverse is continuous and piecewise linear. Gaussian-CDF input conversion is continuous. The new theta_tilde is continuous; psi and its integral are continuous; and the conditional pair CDF is continuous in both inputs and its parameter, with transformed-coordinate derivative uniformly at least.325. Its inverse therefore depends continuously on all these quantities. Composing the root map, fixed matching permutation and conditional pair maps yields a continuous invertible full-dimensional map, with continuous inverse on the open cube (or after a Gaussianized output chart).

It is piecewise C1, not globally C1: root histogram slopes, context interpolation slopes and psi slopes have finite corner boundaries. The normalized-density/Jacobian identities hold a.e. The fitted interpolation's slope need not be bounded by the true L=.5; adjacent clipped bin values can differ by2kappa, giving an upper bound2kappa/h=28.8. Continuity alone is not a small-condition-number or robust numerical-stability guarantee. All endpoint, tail and determinant checks remain necessary.

This removes the previous discontinuity in the input-context dependence without training a new estimator, while weakening the conservative risk bound from0.4333 to0.6529. It does not establish universal latent-flow dominance, a learned semantic representation, or realistic image/video superiority. The earlier warning remains: unknown nonuniform C_1 histograms do not automatically center sine fixtures or establish their required mean-signal margin.
