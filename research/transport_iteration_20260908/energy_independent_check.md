# Independent adversarial review of `theory-audit.md`

Reviewed 2026-09-08 by the repo_audit agent, independently of the note author. No observed data or experiment outcomes were inspected. Verdict: the mathematical claims in sections A–D and the smooth extension are correct under their stated assumptions. The exact asymptotic coefficient includes the sigma and factor-of-two scaling shown below; it must not be shortened to (5/32)m^(-7/2) without that scaling.

## Energy definition and coupling bounds

The convention ES(Q;P)=E||X-Y||-(1/2)E||X-X'|| yields the coefficient 2 in the displacement bound. Both distributions and target must have finite first moments, as explicitly assumed at the start. Taking an independent copy of the coupling bounds the self-distance change by 2E displacement; its half coefficient leaves E displacement. Together with the cross-distance change this gives 2E displacement. Infimizing over couplings is legitimate. The sharpness example gives ES(Q_epsilon;P)=(1-epsilon)^2 and W1=epsilon, giving ratio 2-epsilon. The exact finite-array inequality uses the correct unordered-pair coefficient 1/[n(n-1)], including n>=2. The nonlinear angular twist preserves radius and has an explicit inverse; smooth theta of squared radius avoids a regularity issue at zero. A global Lipschitz synthesis upper bound is sufficient, but has to hold on relevant inputs.

Hoeffding uses independent source pairs from frozen generators, a prespecified almost-sure displacement bound, and the stated family-size correction. The Cantelli expression is correct when a true variance upper bound v is supplied. The adaptive extension is valid if, conditionally on the prior history, each next generator and bound are fixed before its fresh independent source draws, with predictable allocated error budgets whose sum is bounded above by alpha. This rejection statement uses only generated source pairs. Failure to reject is not acceptance or quality evidence.

## Square angular construction

Independent exact-rational arithmetic confirms the cross moments [1,2,6,20,68,232], self moments [1,2,6,20,72,272], and differences [0,0,0,0,-4,-40]. Within P and within Q, the active distance laws agree because Q is a rotation of P. The difference ES(Q;P)-ES(P;P) is cross mean distance minus self mean distance, with no remaining factor 1/2. This is a place where a factor-of-two error would have changed the result; none occurs in the note.

The coefficient of v^4 in sqrt(r^2+v) is -5/(128 r^7). Multiplication by the moment difference -4 gives +5/(32 r^7). The fifth derivative of sqrt(t) is 105/(32 t^(9/2)); division by 5! gives 7/256. Since the fifth derivative is positive and decreasing on t>0, the nonnegative remainder bound 7v^5/(256r^9) holds for every v>=0, without a small-v assumption. The sum of the two fifth moments gives the safe remainder constant (7/256)(232+272)=441/32. This does not rely on cancellation of the remainders.

If R=sqrt(2) sigma chi_m, then E R^(-s)=(2sigma)^(-s) Gamma((m-s)/2)/Gamma(m/2), valid only for s<m. The expansion and remainder require m>9. The ratio of inverse moments is exactly 1/[2 sigma^2(m-9)]. Consequently

Delta_m = (5/32) E R^(-7) + O(E R^(-9))
        ~ (5/32) sigma^(-7) 2^(-7/2) m^(-7/2)

for fixed sigma>0. Dividing Euclidean distances by sqrt(D), with D=m+2, gives normalized Delta_m ~ (5/32) sigma^(-7) 2^(-7/2) D^(-4). The constants and exponents in the note are correct. This asymptotic does not apply unchanged if sigma varies with m.

## Smooth nonlinear-generated extension

Adding independent N(0,tau^2 I_2) noise to each active sample gives pair-difference noise W~N(0,2tau^2 I_2), independent of the Gaussian complement. Its conditional radial moments are exactly the stated polynomials. One independent derivation is the Gaussian heat semigroup: for v=||c||^2 in two dimensions, Delta v^j=4j^2 v^(j-1). This gives

E||c+W||^(2j) = exp(tau^2 Delta) v^j
 = sum_{l=0}^j (4tau^2)^l (j!/(j-l)!)^2 v^(j-l)/l!
 = sum_{l=0}^j binom(j,l)^2 l! (4tau^2)^l v^(j-l).

The leading coefficient is one. Since lower cross/self moment differences through degree three vanish, the first three differences remain zero and the degree-four difference remains -4 after smoothing. All fifth moments are finite for fixed tau; Taylor's global remainder bound and independence from R justify exchanging expectations and retain an O_tau(E R^(-9)) remainder. The leading coefficient and D^(-4) normalized rate persist for fixed tau>0 and sigma>0. The active distributions remain distinct: Gaussian convolution multiplies characteristic functions by a nonzero function and cannot erase the distinction between the original two finite vertex laws. They remain non-Gaussian for every finite tau; otherwise deconvolution would make the nondegenerate finite vertex law Gaussian, which it is not.

The conditional inverse-CDF construction defines a smooth triangular transport on finite real inputs because all marginal and conditional Gaussian-mixture densities are smooth and strictly positive. It is invertible and uses exactly two standard-normal coordinates. This specifies the mathematical transport. Its inverse CDFs generally require numerical solves; closed-form evaluation and constant cost are unproved. Non-Gaussian output forces the transport to be nonlinear. The two laws are still related to each other by a linear rotation, as the note candidly states; nonlinear generation alone does not establish the same moment cancellations for arbitrary nonlinear angular corrections.

## Boundaries to retain

- This is a proper-score dilution example and a target-independent rejection diagnostic. It provides no generative improvement algorithm.
- The asymptotic requires fixed positive sigma and, for the smoothed case, fixed finite tau. The finite inverse-moment proof uses m>9.
- The statement about the first two coordinates retaining a nonzero active score discrepancy uses Euclidean energy score's strict propriety (or can be proved directly for the discrete case). It does not authorize substitution of that projected metric for observation-space energy.
- The coefficient applies to this square-versus-rotated-square construction. It is not a generic consequence of low active dimension, non-Gaussianity, or nonlinear latent generation.
- A rejection for one predeclared energy-score improvement margin does not reject other metrics or prove that the candidate is worse.

The principal coefficient, remainder constant, estimator normalization, and smooth-extension proof pass independent mathematical review.
