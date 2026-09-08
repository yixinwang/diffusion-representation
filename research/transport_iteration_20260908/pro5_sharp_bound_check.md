# Independent check of Pro5 log-loss sharpening

Reviewed `research/transport_iteration5_20260908/LOGLOSS_SHARPENING.md`, the accompanying calculator, and the scope assumptions in `AUDIT_AND_DECISION.md`. Verdict: the sharpening and displayed constants are valid under the stated fixed-feature convex-class assumptions. No frozen source or experiment was changed; the numerical values below were recomputed directly from the formulas, without importing the companion calculator or fitting a model.

## Central condition: valid under misspecification and within-array dependence

Write a_t=(1-t)a*+ta. Convex feasibility and population optimality imply the right directional derivative of E L_a_t at t=0 is nonnegative. Because every density is bounded between ell and U and is affine in a, differentiation under expectation is justified by a bounded derivative. Therefore

    0 <= E[-D^-1 sum_i (q_ai-q_a*i)/q_a*i],
    E[D^-1 sum_i q_ai/q_a*i] <= 1.

Pointwise AM–GM on the D positive ratios then gives E exp(-r_a)<=1. The AM–GM step is inside each complete array, so it assumes no coordinate/site independence. It uses no truth-in-class assumption: the comparator is the population optimum of the same actual feasible convex set. Replacing a* by an empirical fitted head would not justify this condition.

This remains valid with shared heads and frozen nonlinear context gates: the density must remain affine in the fitted coefficients. It does not automatically hold for jointly trained nonlinear gates, neural coupling parameters, or mixtures whose weights are being trained nonlinearly on the same head-fitting arrays.

## Bernstein constant and coefficient

For r in [-B,B], the Taylor integral is bounded below by its value at r=B because exp(-tr)>=exp(-tB) for t>=0. Thus

    exp(-r)-1+r >= r^2*(B-1+exp(-B))/B^2.

Taking expectations and using the central condition gives E r^2<=V_B R, with V_B=B^2/(B-1+exp(-B)) and continuous V_0=2. R>=0 follows from population optimality. Consequently Var(r)<=V_B R.

For n independent complete arrays, apply one-sided Bernstein to Z=R-r. Its mean is zero and Z<=R+B<=2B. The standard bounded-variable form gives, for one net point,

    R-P_n r <= sqrt(2 V_B R t/n) + 2B t/(3n).

Young's inequality sqrt(2 V_B R t/n)<=R/2+V_B t/n therefore yields

    R <= 2P_n r + (2V_B+4B/3)t/n.

The factor 4B/3 is consistent with the centered bound 2B; using the uncentered B directly here would need an additional argument. This derivation supports the displayed coefficient exactly.

## Feasible net, transfer and parameter bookkeeping

With nonnegative context/response partitions of unity, coefficient sup-norm distance e changes every density by at most e and every average-array log loss by at most e/ell. A maximal separated set inside the actual feasible coefficient set gives a feasible cover; independently rounded coefficient rows are not a valid substitute. The stated box-packing upper bound [1+8(U-ell)n/ell]^P at e=ell/(4n) is conservative and valid.

For an empirical tau-optimizer a_hat and a feasible net representative a_bar, let h=1/(4n). Then

    R_hat <= R_bar+h,
    P_n r_bar <= P_n r_hat+h <= tau+h.

Substitution produces exactly 2tau+3h = 2tau+3/(4n). No extra approximation of a* is needed: a* is fixed in every excess-loss definition. A union over a predetermined catalogue of G candidate classes gives log(G/delta); charts/features may also be conditioned on genuinely independent discovery data before this argument.

P is the TOTAL number of separately fitted scalar coefficients, not the number of groups or a per-head count. For group-dependent dictionaries it is P=sum_g R_g*(b_g+1); with common settings it is number_of_groups*R*(b+1). Counting constrained coefficients rather than affine dimension is conservative. Group sharing reduces P but changes the comparator class; it does not increase n from arrays to pooled sites. A full-model tau is the appropriately site-weighted per-coordinate empirical optimization gap, not an arbitrarily small requested tolerance that a capped optimizer failed to attain.

The numerical helper explicitly handles ONE conditional table with P=(b+2)^s*(b+1). Its numerical output must not be reused as a whole-model bound with multiple separately fitted groups without replacing P and accounting for all approximation terms. G denotes catalogue multiplicity, not an automatic substitute for omitted parameter groups.

## Independent numerical check and units

For ell=.25, U=3, B=log(12): V_B=3.9373827504835957 and 2V_B+4B/3=11.187974367351192. Direct integer search over b=2,...,999 reproduces:

| Independent arrays n | Best b | One-table P | Approximation term | Excess term | Total |
|---:|---:|---:|---:|---:|---:|
| 40,000 | 13 | 210 | .4255825251 | .8862607593 | 1.3118432845 |
| 10,000,000 | 32 | 1,122 | .0115919709 | .0258567069 | .0374486779 |
| 100,000,000 | 46 | 2,256 | .0027147252 | .0057798209 | .0084945460 |

These use the calculator's conservative 1/n transfer remainder. The displayed pair gap .01965986395 is joint KL for one restricted product-decoder pair. The general theorem concerns average-array excess; multiplying by D converts it to joint excess. For the favorable numerical illustration one conditional pair-head is charged and other factors are granted for free. That does not certify an entire high-dimensional model. Choosing b from deterministic n/regularity-bound calculations before observing data does not itself require a union across that search; selecting b from empirical scores would be a different selection procedure.

## Remaining uncontrolled term

The result bounds estimation/optimization excess over the best feasible heads for a fixed analysis and feature dictionary. It does not bound the unknown neural-context approximation remainder, omitted informative predecessors, site-sharing mismatch, support/floor mismatch, or quality of a learned analysis. The strengthened constant therefore does not establish natural-image/video applicability or superiority over a globally dependent stochastic latent decoder. The same containing decoder can still copy the full construction and tie it.

The conclusion that this sufficient one-table calculation fails to certify the restricted pair gap at 40k arrays survives the sharper constant. It is not a necessary-sample-size result. For reused real-data development, conditioning on a nominal internal split does not erase previous adaptive reuse; population coverage still needs its own justification.
