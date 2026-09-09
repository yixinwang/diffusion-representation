# Review20: scoped risk-gap theorem and corrected diagnostic

This is an independent elementary derivation for the **current** innovation response, not Review19's different tied-spline flow. No protected observations, checkpoint, or job was accessed.

## Fixed-common-chart population theorem

Fix the same invertible analysis and prefix-dependent base transformation B for both competitors, the same frames W, source ordering, root/anchor distributions, and observed-history variables H_b. In this common pulled-back coordinate system let A_b denote the current16 anchors and F_b the704 followers. History excludes the current anchors. Define

    Gamma = sum_b I_P(F_b ; A_b | H_b).

A comparator is anchor-blind only when q_0(F_b|H_b) cannot access A_b directly or through its summary/cache. It may otherwise be arbitrarily expressive and jointly dependent across followers. Let

    alpha_I = inf_{q in Q_I} sum_b E_{P(H_b,A_b)}
                  KL(P(F_b|H_b,A_b) || q(F_b|H_b,A_b)).

This includes every restriction of the actual response:16-dimensional summary, conditional-product Gaussian followers, rank-limited means/scales, finite head, and frame chart. It is not assumed zero for native data. If all relevant densities and finite log-score expectations exist, the optimal blind-family loss is at least conditional follower entropy given H. The anchored-class infimum equals entropy given(H,A) plus alpha_I. Shared Jacobians/root/anchor terms cancel. Therefore, for normalized population negative log score R=E[-log q(Y)]/3072,

    inf_{Q_0} R - inf_{Q_I} R >= (Gamma-alpha_I)/3072.

For any fitted blind q_0 and a fitted q_I satisfying sup_{Q_I}|Rhat-R|<=epsilon and Rhat(q_I)<=inf_{Q_I}Rhat+xi,

    R(q_0)-R(q_I) >= (Gamma-alpha_I)/3072 - 2epsilon - xi.

The2epsilon term follows by transferring empirical q_I risk to population and back at its class infimum. No empirical optimality of q_0 is required for this lower bound. Epsilon and xi are in normalized-risk units here; if stated in full-array nats, divide them by3072 too.

These assumptions are material. A finite-data neural uniform deviation bound has not been supplied; unbounded Gaussian log scores and unbounded summaries require tail/complexity control. Conditioning on a chart fitted using the same samples does not magically restore iid samples for a fixed-class bound. Use an independent chart-fitting split or a justified uniform bound over the entire joint data-dependent class. The identity holds pointwise for each fixed fitted chart, but learnability does not follow from it. In the current empirical protocol, separately learned frames/base transformations or joint analyses can differ across arms, so a common-chart theorem is not automatically the explanation of their measured difference.

## Exact anchor-permutation identity

Under the true law, sample(A,F)|H, then independently sample A' from **the true P(A|H)**. Let q_a(f)=q(f|a,H), p_a(f)=P(f|a,H), and p(f)=P(f|H). The proposed statistic is

    T_q = E[log q_A(F) - log q_A'(F)].

At each H, expand cross entropies to obtain exactly

    T_q = I_P(F;A|H)
          + E_A KL(p || q_A)
          - E_A KL(p_A || q_A).

Average over H and sum blocks if desired. In particular, if q_a=p_a,

    T_p = I_P(F;A|H) + E_A KL(P(F|H) || P(F|A,H)).

The second term is generally positive. T_p is thus not MI or a lower bound on MI. For arbitrary fitted q, the final subtraction destroys even that simple ordering; an anchor-blind q gives T_q=0 regardless of the true MI. Permuting anchors across unequal continuous histories also does not generate P(A|H), so it changes the statistic again. The model's independent Gaussian anchor prior is not evidence that the true pulled-back anchors have that law conditional on history.

A lower confidence bound failing to exceed a margin is failure to establish that margin, **not falsification**. Even a correctly covered interval would concern T_q, not Gamma. A high permutation contrast may be model overdependence/misfit; a low one may be model underfit. A calibrated upper bound below a margin can reject the corresponding predicted-score contrast, but cannot in general reject a MI premise without additional model/error/resampling guarantees.

### Strict Gaussian witness

For standard normal A,F with correlation rho and q=p, MI=-.5 log(1-rho²), while direct Gaussian cross-entropy calculation gives

    T_p = rho²/(1-rho²).

At rho=.5 these are about.143841 and.333333 nats. This is already a strict discrepancy, not a small estimation error.

### Nonlinear, non-Gaussian witness inside the current16/704 response

Let all16 anchors be independent standard Gaussian. Set one follower

    F_1 = gamma*[tanh(A_1)^2-k] + E_1,
    k = E[tanh(A_1)^2],   E_1 independent standard Gaussian,

and leave other703 followers independent standard Gaussian. This joint law is normalized, nonlinear and non-Gaussian, with Cov(A_1,F_1)=0 by symmetry. The conditional mean is nonconstant when gamma!=0, so dependence/MI is positive.

It is representable within the actual fixed graph-frame chart: choose bottom=0 and Cayley rotation=0, so W=[I16;0]. Coordinate0 is an anchor, coordinate1 is a follower for the declared endpoint-spaced anchor rule; use the frame column supported on coordinate1. Set all scale coefficients to0 and make its mean coefficient gamma*(tanh(A_1)^2-k). Two SiLU hidden units can realize the required linear dependence on the available squared-tanh feature exactly using SiLU(x)-SiLU(-x)=x; an output bias supplies the constant. Other coefficients vanish. Set the common B to its identity controls. Embed other blocks/root through any fixed common invertible chart. This is an exact-real representability example, not a claim that native training learns it.

Writing mu(A)=gamma*(tanh(A)^2-k), conditional variance1 gives

    T_p = Var(mu(A)),
    I(F;A) = h(F)-h(E) <= .5 log(1+Var(mu(A))) < Var(mu(A))

for nonzero variance. The entropy upper bound suffices for the strict comparison. It also shows why zero covariance and nonlinear anchor use do not make permutation contrast a calibrated information estimator.

## One operational diagnostic with an honest target

Freeze a common-chart blind model and an anchored model **before** scoring an independent validation bank. For each independent whole observation i, record

    Delta_i = [sum_b log q_I(F_ib|A_ib,H_ib)
                         - log q_0(F_ib|H_ib)]/3072.

This estimates the fitted models' achieved normalized follower log-score difference, not MI. All groups within an image form one statistical unit; no n-times-block-count sample-size fiction. Save per-image scores, model/chart hashes, input identifiers and every failure. Check common-chart assumptions or instead evaluate each model's complete observation-space log density, including all Jacobians and root terms.

Report the mean and uncertainty with an explicitly justified sampling/tail model. If scores are known a priori to satisfy |Delta_i|<=B, Hoeffding gives a two-sided radius B*sqrt(2 log(2/delta)/n); otherwise this particular guarantee is unavailable. A predeclared bounded/truncated diagnostic would target the truncated score, not the full KL gap, unless its tail error is calibrated. Do not silently truncate model likelihoods. Previously reused repair data provide descriptive development evidence only.

A valid lower bound above a declared margin supports **predictive improvement of these fitted models**. A lower bound not crossing it is inconclusive. Neither outcome falsifies the MI premise without separate control of approximation/estimation error. This diagnostic avoids the uncalibrated conditional-anchor resampling requirement altogether.

## Counts and computation: coefficients are not the pipeline cost

The four response heads have4*[32*(16+2*16)+32+32*32+32]=10496 parameters. Four frame charts have4*[(720-16)*16+16*15/2]=45536 parameters; together56032, with complete model539120. These are not the sole model parameters or a proof of lower complexity.

As implemented by OrthonormalFrame.matrix in cached_global_innovation.py:121–183, each block training transform forms QR([I;bottom]) at O(m r²), a rank-sized Cayley solve at O(r³), and the frame rotation. The graph chart excludes singular leading minors and the Cayley chart excludes rotations with eigenvalue-1. Fixed inference caching removes repeated factorization only after charged preparation; cache validation, copies and memory still count.

Per sample/block, the response head costs O(32*(16+2r)+64r), follower expansion O((m-r)r), plus scalar nonlinearities and writing outputs. The complete prefix conditioner, B transform, coarse/root and analysis are additional costs. Producing a D-dimensional output already costs Omega(D) for every generator. Small coefficient count does not establish an asymptotic or measured end-to-end advantage over latent models that also produce full outputs. Exact stochastic copies remain ties.
