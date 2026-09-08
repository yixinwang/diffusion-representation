# Supplement: sharper independent-array log-loss excess constant

This supplement improves the conservative coefficient in `AUDIT_AND_DECISION.md`.
It does not change the estimator, data splits, frozen pilots, or development rule.
The original displayed constants are evaluated correctly in that audit, but are
not the strongest elementary bound available for its convex log-loss class.
No experiment or model fitting is used here.

## Central condition at the misspecified convex optimum

Keep the same feasible convex coefficient set, fixed chart and feature gates,
and densities in [ell,U]. For one complete array X define

    L_a(X) = -D^-1 sum_i log q_ai(X),
    r_a(X) = L_a(X)-L_a*(X),
    R_a = E r_a(X),    B = log(U/ell).

Here a* minimizes the population risk over the actual feasible convex set.
All q_ai are affine functions of a; their contexts are fixed functions of X.
First-order optimality along the segment from a* to a gives

    E[D^-1 sum_i q_ai(X)/q_a*i(X)] <= 1.

The arithmetic-geometric mean inequality, applied WITHIN each array, implies

    E exp(-r_a) = E product_i(q_ai/q_a*i)^(1/D) <= 1.

This does not require independence among coordinates or a correctly specified
model. It uses population optimality, not empirical optimality of a fitted head.

## A better Bernstein constant

For any r in [-B,B], Taylor's integral identity yields

    exp(-r)-1+r = r^2 integral_0^1 (1-t)exp(-tr) dt
                >= r^2 [exp(-B)-1+B]/B^2.

Therefore, with the continuous value V_0=2 at B=0,

    E r_a^2 <= V_B E[exp(-r_a)-1+r_a] <= V_B R_a,
    V_B = B^2/[B-1+exp(-B)].

Replacing the earlier V=2U^2/ell^2 by V_B is valid in the same independent-array
Bernstein/net proof. A feasible coefficient net of radius ell/(4n) has size
at most [1+8(U-ell)n/ell]^P and changes each array loss by at most 1/(4n).
One-sided Bernstein and Young's inequality give, simultaneously on the net,

    R_a <= 2 empirical_mean(r_a) + (2V_B+4B/3)t/n,
    t = P log[1+8(U-ell)n/ell] + log(G/delta).

Lifting an empirical tau-optimizer to the net gives the full bound

    excess <= (2V_B+4B/3)t/n + 2tau + 3/(4n).

Using 1/n in place of 3/(4n) is a harmless conservative simplification, used by
the companion calculator. P counts every separately fitted coefficient group;
n is still the independent COMPLETE-array count. Frozen independent discovery,
feasible net points, the actual convex constraint set, and a predetermined
catalogue remain necessary. The bound controls excess over the best class
member, not the unknown learned-context approximation error.

## Finite numerical consequences

For the same rho=1/2 pair example, ell=1/4, U=3, s=1, H=10.5,
A_s=55.125, delta=.05, G=1, tau=0:

    B = 2.4849066497880004,
    V_B = 3.9373827504835957,
    2V_B+4B/3 = 11.187974367351192.

The supplied conservative coefficient was 2310.626417732768. Optimizing the
sharpened total approximation-plus-excess bound over integers b=2,...,999 gives
b=13 and 1.3118432844697976 nats at n=40,000. The same restricted product-decoder
joint gap remains 0.01965986394557823 nats. Thus even this substantial constant
improvement does not certify that gap with CIFAR-sized independent-array counts.
At n=10^7 the sharpened bound is about .03745; at n=10^8 it is about .00849.
These are sufficient-bound calculations, NOT necessary sample sizes and NOT
empirical learning performance. A pair gap does not lower-bound a globally
dependent decoder or a learned invertible-analysis latent model.

This closes an important audit loophole: the practical recommendation does not
rest solely on choosing the unnecessarily large ell^-2 curvature constant.
No novelty or optimal-constant claim is made for this elementary refinement.
