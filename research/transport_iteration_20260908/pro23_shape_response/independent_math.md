# Review23 independent mathematical audit — proposed SAS response

Scope: the supplied root description of Review23, not a read/authentication of its full conversation or delivered code. Current production reference is `qalt/src/qalt/innovation_response.py`. No data, checkpoints, fits or jobs were accessed. This is a new response proposal, not a demonstrated correction of current native quality failures.

## Exact map and density

For fixed earlier history H and unchanged Gaussian anchors a, define finite m,s,tau,kappa as functions of (h(H),a), with delta=exp(kappa)>0. Each follower transforms independent z~N(0,1) by

    v = asinh(z), q = sinh((v+tau)/delta), e = m+exp(s)q.
    z = sinh(delta asinh((e-m)exp(-s))-tau).
    log(de/dz) = s-kappa+logcosh((asinh(z)+tau)exp(-kappa))
                -.5 log(1+z²).

Every scalar derivative is strictly positive, the map is onto the real line, and the displayed inverse and log determinant are correct in exact arithmetic. The full anchor/follower Jacobian is block triangular: anchor derivatives of m,s,tau,kappa do not enter its determinant. Compose with the old invertible B_H and preceding blocks in generation order to obtain a normalized full-Gaussian-source law; inverse first applies B_H^-1 then this response inverse. Actual density evaluation still requires all component density/inverse assumptions. Shared/fitted functions are conditioned on as fixed when making this statement.

The exact-representation premise is restrictive: the correct fixed B must pull the true law back to Gaussian anchors with conditionally independent followers having precisely the SAS family with coefficient functions representable by the frozen summary/frame and head. Arbitrary nonlinear, non-Gaussian conditionals do not satisfy this premise. A frozen learned B need not equal that true B. No finite learning rate or native quality conclusion follows from normalized representability. A stochastic decoder implementing the same map with the same source ties exactly.

SAS is an extension of location/scale: tau=kappa=0 nests the old response, while m=s=tau=kappa=0 nests identity. Delta in[.5,2] and |tau|<=1 follow the proposed tanh restrictions, but m remains unbounded as a function of an unbounded history/head input. For each fixed finite context, transformed Gaussian followers have finite polynomial moments; there is no unconditional uniform moment bound without constraints on the context-dependent mean. SAS responses need not have globally bounded derivative, so no uniform bi-Lipschitz/cost/approximation theorem is supplied.

## Counts and controls

Current default response input is48->32 with bias:1568 parameters; output32->32 adds1056, totaling2624/block. Expanding to32->64 changes output count to2112:3680/block, increment1056/block or4224 over four blocks. Therefore current complete539120 becomes543344 if all other modules are identical. P-SAS and I-SAS use the same dimensions/operations/count; only the prescribed anchor-derived versus prefix-derived features differ. This does not make their information identical: that feature difference is exactly the intended ablation. I-LS has fewer response parameters and is explicitly a nested smaller class, not a parameter-matched competitor. Freeze B and frames identically for the three response arms and preserve response gradients through anchor inputs without unintentionally training frozen B.

## Stable identity proposal and its limits

Let v=asinh(z), Delta=v*expm1(-kappa)+tau*exp(-kappa), r=hypot(1,z). Then

    q = z + 2z sinh²(Delta/2) + r sinh(Delta),
    log(dq/dz) = -kappa + log1p(2sinh²(Delta/2)+(z/r)sinh(Delta)).

These follow from sinh(v+Delta), cosh(v)=r and cosh(Delta)-1=2sinh²(Delta/2). They are exact real identities. With ordinary finite z and zero tau,kappa, Delta=0, q=z and logdet=0 exactly in ordinary arithmetic where intermediate operations remain finite. No hard identity branch is necessary; such a branch could wrongly zero parameter gradients. At zero shape,

    dq/dtau = r; dq/dkappa = -v*r;
    d log(dq/dz)/dtau = z/r;
    d log(dq/dz)/dkappa = -1-v*z/r.

These are generally nonzero, so identity initialization can still learn shape. Together with e=m+exp(s)q, de/dm=1 and de/ds=z at identity. A zero final head means earlier hidden-layer gradients may initially be zero, as in the existing response; that is initialization algebra, not proof of a broken graph.

The proposed algebra is not a universal floating-point fix. When Delta opposes a large v, large terms can cancel and the log1p argument can approach-1; rounding can lose positivity even though the exact derivative is positive. sinh/squared sinh can overflow; z² can overflow in the original log1p expression; exp(s), mean subtraction and inverse sinh have their own ranges. Computing logcosh stably as |x|+log1p(exp(-2|x|))-log2 and loghypot rather than log(1+z²)/2 may help but needs value/gradient qualification and branch agreement. Do not clip a bad root, log argument, response or parameter to hide failure. Preserve all nonfinite intermediates and use legitimate equivalent formulas or explicit failure.

`work/pro23-formula-check/check.py` evaluates28 fabricated scalar settings, including exact identity, near identity, shape bounds and |z|=1e8, against80-digit mpmath. The saved report is ordinary floating error evidence only, not an interval certificate, full inverse test, automatic-gradient test or production numerical gate. Analytic derivatives above have been independently derived; no production source changed.

## Hermite and likelihood gates

Under standard normal innovations, h3=(u³-3u)/sqrt6 and h4=(u⁴-6u²+3)/sqrt24 have zero mean, unit variance and are orthogonal to lower Hermite orders. Cross covariance of projected follower features with tanh(anchor) or tanh²(anchor) is a diagnostic of selected dependence, not a complete test of conditional independence, SAS representability or summary sufficiency. Define whether Hermites are formed from inverse Gaussianized innovations or raw B-pulled followers: raw followers under nonidentity LS/SAS are not standard Gaussian, so their Hermite moments lack that null interpretation. For exact inverse-Gaussianized innovations under the correct conditional model, E[h3(u)|H,A]=E[h4(u)|H,A]=0. Consequently their marginal covariance with any integrable bounded anchor feature is zero even when H and anchors are dependent: apply iterated expectation. Shared-history confounding instead applies to raw/misstandardized followers or conditional independence without conditional standard normality. Label those raw statistics as marginal heuristic diagnostics.

Specify projection matrix/normalization, pooling across blocks, covariance centering, norm and denominator before evaluation. Rank16 projection loses directions; finite Hermite orders and32 anchor features (tanh and squared tanh for each of16 anchors) lose dependencies. A law may have zero diagnostic while remaining strongly dependent, and a well-fitted density may retain nonzero raw-feature covariance. Estimating h4 covariance variance requires appropriate eighth moments (bounded anchor factors help but do not bound follower tails). Sampling error, repeated images/sites and checkpoint comparisons must be handled at the independent-image level. No automatic concentration follows from feature orthogonality.

A25% reduction requires a declared positive baseline norm and treatment of near-zero/noisy norms. A25% closure of an LS-minus-RQS positive NLL gap means, for a common per-dimension likelihood evaluation,

    L_LS - L_SAS >= .25*(L_LS-L_RQS), with L_LS-L_RQS>0.

If the initial gap is nonpositive, the closure gate is undefined/inapplicable, not automatically passed. Requiring I-SAS lower NLL than I-LS and P-SAS in every one of three seeds is a descriptive engineering gate, not confirmatory significance after many proposals. These gates cannot falsify all SAS responses or prove a universal architecture failure. A stop decision is a budget decision about this registered implementation.

## Prospective fitting scope needing exact declaration

A3200/800 FIT-only split should sort record IDs by SHA256 of a canonical, unambiguous encoding of salt `iter23-sas-v1` and record ID, with a deterministic tie-break, then take exact counts. Thresholding hashes does not guarantee exact3200/800. All learned A/root/B/response parameters must consume only the3200 training IDs;800 IDs are development validation. The unseen official test remains sealed. Shared A90/root90, identity-response B120, frozen-B response120 for each LS/I-SAS/P-SAS, and RQS240 on same A/root imply nominal420 per standalone arm before measured setup/cache/IO. Sharing B cost across response arms does not make it free. State whether RQS starts fresh and receives its entire240, and avoid describing the different optimizers/classes as identical parameter budgets.

Use fresh78301/78302/78303 initialization streams. No old I-LS weight continuation should be implied by the phrase stop current I-LS: preserve current results and explicitly register the new3200/800 experiment. This experimental proposal is currently unimplemented and unlaunched; the isolated kernel preparation is separate from fitting. There is presently no proved coherent end-to-end learned improvement over competent latent FM or image-generation baselines.

## Scoped population approximation-risk separation

Fix the fitted invertible analysis and B, common anchor law, and history representation; condition on any training used to choose those fixed functions. Suppose true followers E_j are conditionally independent given (H,A) and exactly follow the proposed representable conditional SAS laws. Assume conditional variances are positive and finite and the differential entropies, cross entropies and displayed expectations are integrable. Against the larger comparator containing *arbitrary* conditional independent Gaussian followers, the optimal Gaussian for each conditional law has its true mean and variance. Therefore the approximation-risk gap is

    Delta = E_{H,A} sum_j { .5 log(2*pi*e*Var(E_j|H,A))
                            - h(E_j|H,A) }.

The identity follows by minimizing Gaussian cross entropy over mean and positive variance; each summand equals KL(true conditional || its moment-matched Gaussian). Exact SAS has zero conditional approximation KL under the representability premise. A restricted learned LS head can do no better than the arbitrary Gaussian comparator, so its approximation gap is at least Delta. Summing blocks under their true histories and using change-of-variables invariance gives the corresponding complete-risk difference Delta/3072 if all other components match. This is a population approximation theorem, not a guarantee about fitted weights or finite budgets.

For fixed finite m,s, SAS is Gaussian only at tau=0,delta=1. Indeed the monotone quantile map from a standard Gaussian to any nondegenerate Gaussian must be affine. For q(z)=sinh((asinh(z)+tau)/delta), as z tends to positive infinity q(z) grows as a positive constant times z^(1/delta); affine growth forces delta=1. Then q(z)=cosh(tau)z+sinh(tau)sqrt(1+z²), whose second derivative is sinh(tau)/(1+z²)^(3/2), so affinity forces tau=0. Location/positive scale cannot change that conclusion. Thus Delta>0 if nonidentity shape occurs on a set of positive probability, subject to the integrability assumptions.

The standardized Gaussian negentropy depends only on tau,delta, not m,s. On a compact positive-delta shape domain, Gaussian polynomial-tail domination yields continuity of its moments and entropy; entropy can also be evaluated from Gaussian entropy plus the expected scalar log derivative. Hence a compact subset bounded away from (0,1) has a positive minimum negentropy. If shape parameters land in that subset with probability at least p>0, that subset contributes at least p times that minimum per applicable follower. Neither p nor the minimum is supplied for real images; this is not an explicit quantitative CIFAR bound.

This separation is only versus scalar conditional Gaussian followers in the same fixed coordinates. It proves no advantage over RQS, latent FM, a different learned analysis, or an exact stochastic copy, which can tie the same SAS map. There is no finite-sample learning bound here. The sinh-arcsinh family is established prior work, not a newly invented distribution: Jones and Pewsey (2009), [Sinh-arcsinh distributions](https://doi.org/10.1093/biomet/asp053). The conditional response placement is the proposed application.
