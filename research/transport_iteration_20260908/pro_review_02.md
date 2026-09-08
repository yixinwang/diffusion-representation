# Completed Pro review: output-space exclusion

Conversation: https://chatgpt.com/c/6aa05d79-8a20-83e9-9e12-b6e9c4e52a25

The response completed with a displayed duration of 14 minutes 54 seconds. It was read through the ChatGPT page. The proposed downloadable package could not be retrieved because ChatGPT displayed another temporary request limit. No external code from that package was executed. The mathematical statements below were checked independently in this thread.

## Material conclusions

Let S(Q;P) be expected Euclidean distance from a generated sample to an independent target sample, minus half the expected distance between two independent generated samples. Assume finite first moments. For two frozen generators under a shared Gaussian source, let m be their expected paired displacement, and let A0 and A1 be their respective expected within-law pair distances. The candidate's possible improvement over the parent is bounded by

S(Q0;P)-S(Q1;P) <= B = m + (A1-A0)/2 <= 2m.

This retains the signed diversity change and can strengthen the absolute displacement bound. It does not require target examples. For two independent coupled draws, write their displacements as d0,d1 and the corresponding within-law distances spread0,spread1. Then

H = (d0+d1+spread1-spread0)/2

has expectation B. The triangle inequality gives 0 <= H <= d0+d1. With an analytic displacement limit b, H is in [0,2b]. Independent disjoint pairs permit the usual bounded-variable confidence calculation. Reused source pairs require a different dependence calculation. This extension is a checked derivation; the current committed screen implements only the simpler absolute bound.

The factor two is attained in the limit by moving a small mixture mass from a point at one toward the target at zero. Smoothing the laws and using monotone quantile transports extends sharpness to smooth positive distributions with Gaussian sources. Invertibility and non-Gaussianity do not by themselves reduce the universal constant.

A parent-relative repair must also close the parent's deficit to the strong latent comparator. If that deficit is g, the candidate's improvement over the latent comparator is bounded above by B-g. A required gain eta requires B >= g+eta. A measured historical point gap alone cannot supply a population lower confidence bound for g.

An unrestricted stochastic decoder can split a full-dimensional Gaussian source, reparameterize its active part by a nonlinear bijection, and call the same full generator. Its output law and computation can tie the original flow. Strict superiority requires specified learning and resource restrictions. Total cost must include training, preprocessing, tuning, transforms, decoders, output materialization, and inference. Memory and latency require separate measurements.

A nontrivial radius-independent angular warp is positively homogeneous. Differentiability at the origin would force it to be linear, by taking its directional derivative. Almost-everywhere normalization does not establish a globally smooth extension. This finding concerns radius-independent angular warps; a smooth radius-dependent twist can behave differently at zero.

## Repository reconciliation

The returned R22 branch hash matched the local Git reference exactly: `eed5ca696a6ea39c4dc5b1d7b7f2c9a1d5a7e9a1`.

The R20 result quoted in the original supplied Pro task and the separate GitHub milestone describe different recorded source strata and sample counts. The supplied task reported gain 0.00009155 with 101 generated identities and 61 references. The GitHub milestone reported 0.00010105 with 64 identities and 128 references and an additional class subset. Neither record supports the required 0.005 gain. They must remain separate records; their effects and intervals cannot be combined. The original sampler commits in those Pro tasks were not available for direct reproduction through the reviewed branch.

The review recommends measuring coupled output displacement before further angular repairs. It does not establish that any R20--R22 candidate satisfies a sufficiently small displacement bound. The synthetic check in this branch is explicitly separate from those historical candidates.
