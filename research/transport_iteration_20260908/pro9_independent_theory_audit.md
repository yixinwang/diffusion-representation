# Independent Pro9 theory audit

This is a mathematical audit of the supplied construction, not an audit of delivered code, an experiment, or a claim of superiority to flow matching. All fitted maps below are fixed, or the argument is conditional on the complete training sigma-field and applied to an independent new array. No independence of pixels or arrays within a training cluster is assumed by these identities.

## Conditional KL decomposition

Let an invertible fitted analysis map A send X to (C,R_1,...,R_B), let H_b=(C,R_<b), and define the fixed invertible conditional map

T_H(r)=M(H)^(-1) D(H)^(-1)(r-mu(H)),

where M=I+U diag(exp(alpha)-1) U^T and U^T U=I. Put V_b=T_H(R_b). The model conditional is the pushforward under T_H^(-1) of the product of scalar densities q_bi(v_i | h_bi(H)). Each summary h_bi must be a deterministic function of this entire observed prefix. Then

KL(P_X || Q_X) = KL(P_C || q_C)
 + sum_b E_H TC(P_{V_b|H})
 + sum_b sum_i I(V_bi; H | h_bi(H))
 + sum_b sum_i E_h KL(P_{V_bi|h} || q_bi(.|h)).

Here all P distributions are induced by the *same fitted* A, mu, D, M, and summaries. In particular V is not a fixed raw residual independent of those parameters. Proof: use invariance under A, the chain rule over blocks, and invariance of each conditional KL under T_H; insert the product of the true conditional scalar marginals, then the true summary-conditional marginals inside each log density ratio. Conditional expectation of the resulting three terms gives TC, conditional mutual information, and scalar fitting KL. The equalities hold in the extended sense under the usual absolute-continuity conditions; finite-KL statements require their integrability.

No Jacobian term is missing: the model conditional density in r includes -log det D -log det M, which cancels in the conditional change of coordinates. The history dependence of T_H causes no additional determinant: the complete map over blocks is triangular. Its diagonal block determinants are precisely the conditional determinants. This requires H to consist only of preceding blocks, never the block currently transformed or future blocks.

The covariance separation discussed below concerns dependence *before* inverse mixing, not TC of independent true innovations after their correct inverse mixing. Confusing these two distributions would invalidate the proposed separation argument.

When A or mixing changes during training, all these induced true distributions change too. This decomposition does not prove their learnability, summary sufficiency, optimization convergence, or small population error. Good held-out likelihood alone does not bound any one term without controlling the unknown entropy/reference risk. Conditioning on fitted functions avoids a false treatment of training observations as fresh independent evaluation samples. A randomized summary is different: adding its random seed gives an augmented-law decomposition, whereas marginalizing that seed generally yields only a KL upper bound by data processing and can induce scalar dependence.

## Low-rank mixing and scalar map

For orthonormal U with k=16 columns, M has eigenvalues exp(alpha_j) on span(U), and one on its orthogonal complement. Thus M is positive definite, M^(-1)=I+U diag(exp(-alpha)-1)U^T, and log det M=sum_j alpha_j. Under |alpha_j|, |ell_i| <= log 2, singular values of D M are between 1/4 and 4; log det(DM)=sum_i ell_i+sum_j alpha_j. These statements require genuine column orthonormality and residual dimension at least 16. An unconstrained learned U does not satisfy them merely because it has 16 columns.

Take knots x_j=-4+j, j=0,...,8. Set heights a_0=a_8=1 and a_j=0.1+6.3 softmax(theta)_j for j=1,...,7. Define g(-4)=-4, interpolate g' linearly between neighboring heights, integrate on [-4,4], and set g(z)=z outside. Since sum_{j=1}^7 a_j=7, the trapezoidal integral is 1+7=8, so g(4)=4. The map and derivative match identity at both boundaries. It is globally C1, strictly increasing, with 0.1 <= g' <= 6.4, and is a normalized scalar Gaussian pushforward. It is generally not C2 at knots.

Within bin j, write t=z-x_j, beta=a_{j+1}-a_j, and d=y-g(x_j). Then d=a_j t+beta t^2/2 and

t=2d / (a_j+sqrt(a_j^2+2 beta d)).

This stable expression includes beta=0 and d=0 without dividing by beta. The denominator is positive; on the valid interval the square root equals a_j+beta t. Density evaluation uses log phi(z)-log g'(z), and the inverse log determinant has the opposite sign. Exact interval selection and positive radicands need numerical implementation checks; a mathematical formula does not certify floating-point cancellation, out-of-bin roots, or gradient correctness.

The derivative g'(z) is continuous at the fixed input knots, but its derivative with respect to z has jumps. Branch-based inverse and shape derivatives are valid almost everywhere; arbitrary autodifferentiation at moving output knots is not established by that claim. Symmetric innovations are NOT automatic: sufficient symmetry is a_j=a_{8-j}, making g odd. Unrestricted seven-way softmax does not enforce that condition. Conditional identical variance also requires the corresponding innovation laws to agree, or it must be replaced by their actual diagonal covariance matrix.

## Covariance separation: valid restriction and a necessary correction

For rank one M=I+(exp(alpha)-1)u u^T, ||u||=1, assume conditionally independent, identically distributed, zero-mean symmetric innovations of variance v(H). Then Cov(E|H)=v I and Cov(ME|H)=v[I+(exp(2 alpha)-1)u u^T]. Consequently, for i != j,

Cov(R_i,R_j | H)=exp(ell_i+ell_j) v(H) (exp(2 alpha)-1) u_i u_j.

There is also the diagonal v exp(2 ell_i) term when i=j. Conditional symmetry ensures zero innovation means; shifts mu do not affect covariance. Gaussianity is not needed. If variances differ, the correct expression is D M diag(v_i) M^T D and the displayed simplification fails.

Nonzero off-diagonal covariance proves conditional nonindependence and hence positive conditional TC whenever TC is defined. Any scalar-product conditional decoder in these same residual coordinates has best possible KL equal to this TC; adding restricted summaries can only increase its error. This compares a restricted ablation. A decoder with mixing, a sufficiently expressive RQS coupling flow, or general flow matching is not separated by this argument. If analysis A itself is unrestricted and can absorb the mixing, even the scalar-product ablation may tie; the separation must fix the same analysis/coordinates and admissible context for both models.

The numerical claim “TC >= 2 covariance^2” requires bounded feature ranges. For f_i,f_j taking values in [0,1], compare the joint law with the product of its marginals. The difference in expectations of f_i f_j is their covariance, and this test has range [0,1]. Therefore |covariance| <= TV and Pinsker gives TC >= 2 covariance^2 (natural logarithms). The same claim is false for general [-1,1] features: symmetric correlated Rademacher variables with correlation r have TC=[(1+r)log(1+r)+(1-r)log(1-r)]/2 = r^2/2+O(r^4). For [-1,1] features the direct guaranteed bound is TC >= covariance^2/2. For arbitrary unbounded raw coordinates neither bound follows. Thus one cannot insert the raw covariance formula above directly into the bounded-feature bound.

A rigorous bridge exists using clipping, but changes the numerical margin. For centered symmetric W_i,W_j with covariance c != 0, let f_i=(clip(W_i,-B,B)+B)/(2B). Dominated convergence gives Cov(clip W_i,clip W_j) -> c as B grows, so some finite B gives nonzero feature covariance and strict TC. Quantitatively, if E W_i^2,E W_j^2 <= V and E W_i^4,E W_j^4 <= K, Cauchy-Schwarz and the clipping tail bound give |c-Cov(clip W_i,clip W_j)| <= 2 sqrt(VK)/B. Symmetry makes the clipped means zero. Taking B >= 4 sqrt(VK)/|c| gives |Cov(f_i,f_j)| >= |c|/(8 B^2), hence TC >= c^2/(32 B^4). This conservative bound is not the raw-covariance claimed constant. The proposed scalar identity tails give finite moments, so such a finite bound exists. A useful conditional population margin additionally requires uniform moment/parameter restrictions and a positive-probability set of histories with covariance bounded away from zero; covariance nonzero at an isolated history is insufficient.

## Joint training implications to check against delivered code

A frozen root density's weights may have requires_grad=False, but evaluating its log density on C=A(X) must retain input gradients when A is trained. Wrapping that evaluation in no_grad or detaching C removes part of the joint objective. Freezing root parameters is distinct from freezing root inputs.

If A changes, cached C, R, or prefixes computed from an earlier A are stale and do not evaluate the current joint likelihood. Joint training must recompute them from the current A on each batch, including the analysis log determinant and every conditional mixing/scalar log determinant. A cache is valid only after A is frozen, or if recomputed consistently after every update. The root KL can worsen as A changes even when root parameters remain fixed.

At generation, compute each prefix from generated preceding blocks and generate every innovation from its own source coordinate. Teacher prefixes at generation, reuse of a scalar noise coordinate, or omitting a residual source dimension changes the claimed model. Root exact-density assumptions remain separate from a finite-Heun coarse sampler's invertibility; an exact conditional decoder does not by itself turn an unverified coarse numerical sampler into an evaluable full density.

The decomposition and integrated-slope scalar construction are mathematically sound under the stated conditions. The unconditional raw covariance-to-TC constant, automatic symmetry, and a separation that permits unrestricted analysis or expressive competing decoders would be unjustified. No empirical or universal quality/efficiency conclusion follows from these identities.
