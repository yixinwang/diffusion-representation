# Covariance log-score advantage under a fixed nonlinear chart

Date: 2026-09-08. Author: theory_audit agent. Status: derivations and numerical constants checked by author and independently reviewed by the root agent. This note supplies mathematical interpretation for the frozen covariance experiment. It introduces no implementation or new empirical result.

## 1. Variables, model family, and assumptions

Let X be an observed detail block, let W be its conditioning context, and let T_W be a fixed invertible differentiable B4 chart. The chart output is Y=T_W(X) in R^d, with d=9 for the frozen study. The context and chart are shared by all compared methods and remain fixed after training. Write S=E[YY^T] for the uncentered population second moment, where the expectation averages over the prespecified target distribution of images, sites, and contexts. Assume S is finite and positive definite.

The matrix S may differ from the centered covariance: E[Y] may be nonzero. The frozen procedure fits no location parameter. Calling it second-moment fitting avoids asserting that this layer estimates or removes the mean. Positive definiteness is a separate assumption; finite second moments alone do not ensure it.

For any positive definite matrix C, the Gaussian model q_C is N(0,C). Its log density is

log q_C(y) = -(d/2) log(2 pi) -(1/2) logdet(C) -(1/2) y^T C^(-1) y.

The inverse map C^(1/2) applied to a standard Gaussian gives this density. Equivalently, the forward whitening map C^(-1/2) contributes log determinant -(1/2)logdet(C). No assumption that the true Y is Gaussian enters the following expected log-score formulas.

Define L(C)=E[log q_C(Y)]/d, measured in nats per detail coordinate. Finite S makes this expectation finite. A fitted random C is treated as fixed when evaluating L on fresh data, conditional on the training outcome.

## 2. Exact expected gain and optimum

The identity E[Y^T A Y]=tr(A S) for a fixed matrix A gives

L(C) = -(1/2)log(2 pi) -(logdet C + tr(C^(-1)S))/(2d).

The gain relative to the standard Gaussian is

g(C;S) = L(C)-L(I)
       = [tr S - tr(C^(-1)S) - logdet C]/(2d).

To compare with C=S, define A=S^(-1/2) C S^(-1/2), and let lambda_1,...,lambda_d be its positive eigenvalues. Cyclic invariance of trace gives tr(C^(-1)S)=tr(A^(-1)); determinant multiplication gives logdet C-logdet S=logdet A. The regret relative to the population optimum is

R(C;S) = L(S)-L(C)
       = (1/(2d)) sum_j [lambda_j^(-1)-1+log(lambda_j)].

For f(lambda)=lambda^(-1)-1+log(lambda), its derivative is (lambda-1)/lambda^2. It decreases up to one and increases after one, with f(1)=0. Every summand is nonnegative. Equality holds only when A=I, or C=S. This proves the unique population optimum in the zero-mean full Gaussian family for arbitrary true Y with the stated moments.

The optimal gain against the identity is

g(S;S) = [tr S-d-logdet S]/(2d) >= 0.

It is strictly positive exactly when S differs from I. The result concerns Gaussian expected log score. It does not assert that Gaussian residuals represent the full nonlinear or non-Gaussian distribution.

## 3. Finite perturbation regret

Suppose the fitted second moment satisfies

||S^(-1/2) C S^(-1/2)-I||_op <= epsilon < 1,

where ||.||_op is the spectral operator norm. Every lambda_j lies in [1-epsilon,1+epsilon]. The maximum of f on this interval is at one of the endpoints. The difference between its values at the lower and upper endpoints is

f(1-epsilon)-f(1+epsilon)
= 2 epsilon/(1-epsilon^2) + log((1-epsilon)/(1+epsilon)).

This difference is zero at epsilon=0, and its derivative with respect to epsilon is 4 epsilon^2/(1-epsilon^2)^2 >= 0. The lower endpoint has the larger value. It follows that

0 <= R(C;S) <= r(epsilon),
r(epsilon) = (1/2)[epsilon/(1-epsilon) + log(1-epsilon)].

This uniform bound is sharp: C=(1-epsilon)S attains it. Its expansion is r(epsilon)=epsilon^2/4+O(epsilon^3). An algebraic upper bound, obtained by integrating f'(1-t), is

r(epsilon) <= epsilon^2/[4(1-epsilon)^2].

The logarithmic expression should be used for the numerical thresholds below. The elementary quadratic bound is more conservative.

This result holds for the matrix used to score samples. Eigenvalue floors, shrinkage, truncation, or conditioning rejection rules can change C and must be included when evaluating the premise. Passing a numerical condition-number threshold alone does not prove closeness to the unknown S.

## 4. Advantage over the optimal block or diagonal Gaussian

Fix a partition of the d coordinates into blocks. Let B be the block diagonal matrix consisting of the corresponding principal blocks of S. Every principal block is positive definite. Among zero-mean Gaussians with this block structure, B uniquely maximizes L, because the expected quadratic form for a block diagonal inverse depends only on those principal blocks and each block has the optimum established in Section 2.

In particular tr(B^(-1)S)=d. The population full-versus-block advantage is

Delta_block(S) = L(S)-L(B)
               = [logdet B-logdet S]/(2d).

For completeness, K=B^(-1/2) S B^(-1/2) has trace d. The scalar inequality log t<=t-1 applied to its eigenvalues proves logdet K<=0, with equality only when K=I. This proves Delta_block(S)>=0, strictly positive exactly when S has a nonzero cross-block second moment. Singleton blocks give the optimal diagonal comparator.

For the fitted full matrix C, the exact comparison with optimal B is

L(C)-L(B) = Delta_block(S)-R(C;S)
          >= Delta_block(S)-r(epsilon).

If a fitted comparator D has the required block structure, then L(D)<=L(B). The same lower bound applies to L(C)-L(D), even if D was selected or fitted by another procedure. More explicitly,

L(C)-L(D) = Delta_block(S)-R(C;S)+[L(B)-L(D)].

A sufficient condition to retain a predeclared margin eta>0 against every block Gaussian is Delta_block(S)-r(epsilon)>eta. This is a population score guarantee conditional on the perturbation premise. It is not a statement that a finite training set satisfies that premise.

## 5. Frozen rho=0.3 example and its finite-error margin

Take d=9 with three consecutive coordinate blocks of size three, matching the frozen block3 comparison. Consider

S_rho = [[I_3, rho I_3, 0],
         [rho I_3, I_3, 0],
         [0, 0, I_3]], for |rho|<1.

The optimal block and diagonal matrices both equal I_9. Each of the three coordinate pairs spanning the first two blocks has a two-by-two second-moment matrix with determinant 1-rho^2; the last block has zero cross-moments with the first two blocks. It follows that det S_rho=(1-rho^2)^3 and

Delta_block(S_rho) = -log(1-rho^2)/6.

At rho=0.3 this is 0.01571844657854022 nats per detail coordinate. The spectral perturbation threshold that merely preserves a positive gain is epsilon<0.21391552866179164. To retain the declared eta=0.01 gain, the sufficient threshold is stricter:

epsilon < 0.13715714975359652.

At epsilon=0.13715714975359652 r(epsilon)=0.00571844657854022, so strict inequality yields a gain strictly above 0.01. These decimals are numerical solutions of the displayed monotone scalar equation. They use no experiment data.

For example, epsilon=0.10 gives r(epsilon)=0.002875297726642405 and a guaranteed gain of at least 0.012843148851897815. Epsilon=0.20 gives r(epsilon)=0.013428224342895118, leaving only 0.002290222235645102; that bound does not retain the 0.01 margin.

This exact score result has a non-Gaussian realization. Let U,V,Z be independent three-dimensional random vectors with independent standardized non-Gaussian coordinates, and put Y=(U, rho U+sqrt(1-rho^2)V, Z). The resulting second moment is S_rho. For a smooth nonlinear Gaussian-input realization, each standardized coordinate can be a scalar sinh transform of a standard Gaussian divided by its standard deviation. In that case the coordinates have mean zero, variance one, finite moments, and non-Gaussian distributions. Applying any fixed invertible nonlinear B4 synthesis map transfers the same density-ratio gain to observed detail coordinates. This construction does not prove that actual image-derived B4 coordinates have S_rho.

## 6. Direct nonlinear transfer and its limits

For fixed context w, the normalized observed density corresponding to q_C is

p_C(x|w)=q_C(T_w(x)) |det J_(T_w)(x)|.

For two matrices C and D, the chart Jacobian cancels pointwise:

log p_C(x|w)-log p_D(x|w)
= log q_C(T_w(x))-log q_D(T_w(x)).

Averaging over the actual joint distribution of contexts and observations proves that the log-score comparison in Sections 2-5 holds directly for the observed conditional density, even when the chart and observations are nonlinear and non-Gaussian. If both full joint models use the same context density, it cancels as well.

This equality does not automatically apply after marginalizing out context, changing the chart for one comparator, or applying different dequantization or observation preprocessing. For a whole image, sum the local log-density differences under the registered conditional factorization and then use the declared normalization. A margin per nine-coordinate detail block should not be relabeled as a margin per entire-image coordinate without the appropriate count conversion.

The invariance is specific to log-density ratios. It does not prove a gain in energy score, FID, visual quality, or representation quality. It also does not remove misspecification shared by all Gaussian models in the same chart.

## 7. Cancellation across strata

Let H index strata with fixed target weights pi_h>=0 summing to one, and let S_h=E[YY^T|H=h]. A single global C is scored using only S=sum_h pi_h S_h. The gain and optimum above depend on this aggregate matrix.

For a decisive failure example, take two equally weighted strata with S_1=S_rho and S_2=S_(-rho). Their aggregate is I_9. Each stratum has conditional full-versus-block gap -log(1-rho^2)/6, but the shared global Gaussian optimum is C=I_9 and the global full-versus-block gain is exactly zero. Increasing sample size cannot recover a positive global covariance gain in this example. Non-Gaussian dependence can remain even though second moments cancel.

Separate population-optimal full matrices for the strata would increase the gain against I by

[logdet(sum_h pi_h S_h)-sum_h pi_h logdet S_h]/(2d) >= 0,

using concavity of log determinant; the trace terms cancel. This identifies information lost by pooling. It does not authorize a stratum-conditioned repair to the frozen method or show that strata are valid generative conditioning inputs. If strata depend on variables not available in generative order, a new normalized model would require a separate proof.

## 8. Finite image-cluster inference obligation

None of the population formulas assumes that sites within an image are independent. They concern the target expectation S. Statistical guarantees for an empirical C require a sampling argument matched to image dependence and the chosen weighting.

For illustration, suppose there are n independent image clusters. Within cluster i, sites may be arbitrarily dependent. Let M_i be the average of Y Y^T over the prespecified sites of that image. If the target gives equal weight to images and equal weight to those sites within each image, then S=E[M_i] and C=(1/n)sum_i M_i estimates the correct target. A pooled sum over unequal numbers of sites instead has different weighting and must be analyzed for its own target.

Conditional on a chart trained on independent data, suppose additionally that the matrices M_i are identically distributed and that a finite known bound v satisfies

E[||S^(-1/2) M_i S^(-1/2)-I||_F^2] <= v,

where ||.||_F is the Frobenius norm. Independence across images, centering, and expansion of the squared norm give

E[||S^(-1/2) C S^(-1/2)-I||_F^2] <= v/n.

Since the operator norm is bounded by the Frobenius norm, Markov's inequality proves

Pr(||S^(-1/2) C S^(-1/2)-I||_op > epsilon) <= v/(n epsilon^2).

This deliberately conservative finite-sample bound demonstrates the required unit of independence: images. It does not replace n by the number of spatial sites. Its additional fourth-moment-type premise is not implied by finite second moments of Y. The unknown S and v also prevent treating it as an immediately computable empirical certificate.

With only finite second moments, an independent-image law of large numbers can justify eventual consistency under the appropriate integrability and weighting, but it supplies no distribution-free finite sample rate of the stated form. A finite-image guarantee requires justified tails or cluster-moment bounds, or another validated inference procedure. Reusing the same images to train the nonlinear chart also introduces adaptation: a conditional independence proof, separate fitting data, or a different statistical argument is required. Image-cluster bootstrap intervals can summarize the registered held-out score comparison, but do not on their own verify the spectral perturbation premise.

For the frozen eta=0.01 example, the unresolved empirical-theory bridge is an image-level justification that epsilon<0.13715714975359652, together with a population lower bound on the actual Delta_block(S) near the toy value. Neither should be inferred from millions of correlated sites or from the toy covariance alone.

## 9. The Student comparator remains a separate empirical comparison

The trace/log-determinant formula compares Gaussian densities. A fixed Student density has a different log-score dependence, involving expectations of logarithms of quadratic forms or coordinatewise squared values. Its expected score is not determined by S alone. The argument above proves no dominance over that comparator.

A counterexample can be constructed by taking the true chart distribution to equal the fixed Student model with finite second moments. Under the integrability conditions for its log score, that model strictly outperforms every distinct Gaussian in expected log score, by nonnegativity of KL divergence. If the fixed comparator is a product of Student densities, use that product as the true chart distribution. A finite second moment and a nonlinear chart do not remove this counterexample.

The frozen experiment must report its Student comparison separately. A significant covariance gain over the diagonal or block Gaussian cannot substitute for that result.

## 10. Addendum: verified directional energy-score upper bound

Let Q_0 be the original generator law and Q_1 the candidate law on a Euclidean observation space. For a target P, define ES(Q;P)=E||X-Y||-(1/2)E||X-X'||, with independent draws for each term. All norms below use the same optional pixel normalization as the score.

Take any coupling (X_0,X_1), let (X_0',X_1') be an independent copy, and let Y~P be independent of both. Define m=E||X_0-X_1|| and intrinsic spreads A_j=E||X_j-X_j'|| for j in {0,1}. The target-distance difference satisfies

E||X_0-Y||-E||X_1-Y|| <= m.

Consequently the one-direction improvement is bounded by

ES(Q_0;P)-ES(Q_1;P) <= B,
B=m+(1/2)(A_1-A_0).

The two-copy triangle inequality gives |A_1-A_0|<=2m, proving 0<=B<=2m. This refines the symmetric displacement bound for a fixed direction. It is nonnegative even if the true improvement is negative.

For one pair of independent coupling draws define

d_0=||X_0-X_1||, d_1=||X_0'-X_1'||,
s_0=||X_0-X_0'||, s_1=||X_1-X_1'||,
H=(1/2)(d_0+d_1+s_1-s_0).

The reverse triangle inequality gives -(d_0+d_1)<=s_1-s_0<=d_0+d_1. It follows that 0<=H<=d_0+d_1 and E[H]=B. If the normalized paired displacement has known almost-sure bound b, then H lies in [0,2b]. For N independent, disjoint seed pairs, Hoeffding's inequality gives with probability at least 1-alpha

B <= min{2b, mean(H_i)+2b sqrt(log(1/alpha)/(2N))}.

An upper bound below the declared energy margin rejects the candidate for that one-direction population improvement margin. Reusing seed pairs in a full U-statistic does not yield N independent terms; it requires a concentration result for that dependence structure. Candidate freezing, fresh randomness, and multiplicity conditions remain as stated in the displacement proof. This diagnostic uses no target observations.

This directional bound may be tighter in expectation than 2m, but its larger bounded range and use of seed pairs affect confidence width. It should be assessed at matched computational and confidence budgets. No implementation change is made in this note.

## Verification record

The exact scalar regret and endpoint comparison were derived explicitly. The rho=0.3 gain and both epsilon thresholds were recomputed with bisection in double precision. The directional energy bound was checked separately from the symmetric proof using its two-copy triangle inequality. Application review must verify determinant normalization, the image target weighting, the actual chart's conditional generative order, and interpretation of the frozen experiment's score units. The note makes no algorithmic novelty or real-data improvement claim.
