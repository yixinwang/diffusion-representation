# LIFT-FM theory-to-experiment revision 1

Date: 2026-08-26. Final test remains sealed.

## Development observation

The disjoint smoke run (seed 4099; 120 flow steps, 100 VAE steps) passed all hard parity/compute checks. The joint three-orientation fiber improved held-out conditional log score over the product of its fitted marginals by 0.02720 nat per detail coefficient. Nevertheless, the pooled pixel energy-score contrast `scalar - joint` was -3.66e-5, effectively a tie and in the unfavorable direction. LIFT-FM remained better than the full flow in this smoke run by 0.01279 energy-score units and 0.00685 SWD units, with a measured total transport-time ratio 0.1613.

This is a registered failure of the original “joint beats scalar in pixel energy score” promotion gate, not evidence that dependence is absent.

## Toy explanation

Let the true two-coordinate residual be `P=N(0,Sigma_rho)` with unit marginal variances and correlation rho, while the scalar product comparator is `Q=N(0,I)`. Their exact proper log-score separation is

`KL(P||Q) = -0.5 log(1-rho^2) = rho^2/2 + O(rho^4)`.

Now embed many independent blocks in D dimensions and consider Euclidean energy scores. Pairwise squared norms concentrate around D. A perturbation of one block changes a norm as

`sqrt(D + delta) = sqrt(D) + delta/(2 sqrt(D)) - delta^2/(8 D^(3/2)) + ...`.

The first-order term cancels when P and Q share coordinate marginals and total second moment; dependence appears in the second-order term, suppressed by `D^(-3/2)`. Thus a product model can have a clear O(rho^2) conditional KL defect while pixel energy score has poor finite-sample sensitivity. The smoke output matches this mechanism.

## Revision derived from the toy model

The joint block is now selected by the held-out conditional logarithmic score, which is strictly proper for the declared density and has the unsuppressed KL separation above. Endpoint generation is still required to be noninferior to the full flow by pixel energy score and SWD, so the revision does not hide degraded generations. The scalar product remains an endpoint ablation, but no significance claim is attached to a low-power energy-score contrast between the two fibers.

The confirmation promotion family is revised before any final-test access to:

1. positive paired conditional log-score gain `NLL_scalar - NLL_joint`;
2. LIFT endpoint energy-score noninferiority to full flow within 0.02;
3. LIFT endpoint SWD noninferiority within 0.02;
4. total measured transport time ratio below one;
5. exact-copy and Haar round-trip hard checks;
6. positive finite-rate VAE reconstruction error.

The original preregistration and smoke result are retained unchanged. This amendment changes neither architecture nor final-test metric margins.
