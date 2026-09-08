# Observation-space energy diagnostic: proof and assumptions

Date: 2026-09-08. Author: theory_audit agent. Status: independently checked; see energy_independent_check.md. This note defines a diagnostic. It provides no new generative algorithm or empirical result. No evaluation data were inspected.

## Question and definitions

The observation space is Euclidean R^D. P is the target probability distribution. Q and Q' are two generated distributions. Every distribution below has a finite first moment. Define the population energy score

ES(Q; P) = E[||X-Y||] - (1/2) E[||X-X'||],

where X,X' are independent with law Q, Y has law P, and all three are mutually independent. The objective is to minimize the score. Normalized pixel energy divides every Euclidean distance by sqrt(D). This normalization must be applied consistently to scores, margins, displacement, and confidence bounds.

Question: can a proposed change move the observation-space score by a predeclared improvement margin eta > 0? A rejection certificate establishes that its maximum possible improvement is below eta. It does not establish that the candidate improves the score at all.

## A. Population displacement certificate

Let (X,Z) be any coupling with marginals Q,Q', and let (X',Z') be an independent copy. Let Y be independent with law P. Put delta = ||X-Z||.

A1. The reverse triangle inequality gives

| ||X-Y|| - ||Z-Y|| | <= delta.

A2. Applying the same inequality to the self-distance terms gives

| ||X-X'|| - ||Z-Z'|| | <= ||X-Z|| + ||X'-Z'||.

A3. Take expectations, multiply A2 by 1/2, and add:

|ES(Q; P) - ES(Q'; P)| <= 2 E[delta].

A4. Infimizing over couplings proves

|ES(Q; P) - ES(Q'; P)| <= 2 W_1(Q,Q'),

where W_1 is the 1-Wasserstein distance using Euclidean cost. For sampling implementations, couple both methods using the same Gaussian input U and compute X=G(U), Z=G'(U). This coupling may fail to attain W_1. Its expected output displacement is still a valid upper bound after multiplication by two.

Practical consequence: if 2 E[delta] < eta, this modification cannot improve energy score by eta against any target distribution P. No target examples are required to evaluate the displacement bound. An energy score reported without the factor 1/2 on the self-term is a different convention and changes the constant.

### The constant two is sharp

In one dimension, set P=point_mass(0), Q=point_mass(1), and Q_epsilon=(1-epsilon) point_mass(1) + epsilon point_mass(0), for 0<epsilon<1. Then W_1(Q,Q_epsilon)=epsilon. ES(Q;P)=1, whereas ES(Q_epsilon;P)=(1-epsilon)^2. The score difference divided by W_1 is 2-epsilon, approaching two as epsilon tends to zero. A uniformly reduced constant is impossible, even for bounded observations.

### Nonlinear low-rank transport

Let H be an orthogonal observation transform and write H(X)=(A,B), with A in R^k and B in R^(D-k). A modification may be any measurable nonlinear transport T on active coordinates, preserving B: Z=H^{-1}(T(A),B). Then delta=||A-T(A)|| exactly. The proof requires only the stated orthogonality. If ||A-T(A)||<=rho almost surely, the absolute normalized energy change is bounded by 2 rho/sqrt(D).

For a nonorthogonal L-Lipschitz synthesis transform, delta<=L||A-T(A)|| is sufficient. Whenever possible, measure actual output displacement, because a transform's Lipschitz upper bound can be loose. For a radius-dependent angular twist T(a)=Rot(theta(||a||^2))a on a two-dimensional active block, delta=2||a|| |sin(theta(||a||^2)/2)|. Smooth nonconstant theta gives a nonlinear invertible map; its inverse rotates by the negative angle at the same radius. This yields a directly computable diagnostic for nonlinear angular correction.

## B. A deterministic finite-sample bound

Take paired generated samples (x_i,z_i), i=1,...,n, n>=2, and arbitrary observations y_j, j=1,...,m, m>=1. Define the unbiased self-distance estimator

ES_hat(x;y) = (1/(nm)) sum_i sum_j ||x_i-y_j||
              - (1/(n(n-1))) sum_{i<l} ||x_i-x_l||.

The second coefficient equals one half times the average over ordered distinct pairs. Put delta_i=||x_i-z_i||.

B1. The change in the first term is bounded by mean(delta_i).

B2. Each delta_i appears in n-1 unordered pair bounds. The change in the second term is bounded by mean(delta_i).

B3. Consequently,

|ES_hat(x;y)-ES_hat(z;y)| <= 2 mean(delta_i).

This is an exact arithmetic inequality for every fixed array, independent of randomness, target sampling, or how candidates were selected. It also exposes implementation errors if violated beyond numerical tolerance. It does not by itself bound a population score. If score estimation uses separate unpaired generated sets, B3 is not the bound for that realized comparison.

## C. A finite-confidence population rejection certificate

Freeze both generators, preprocessing, the displacement definition, and eta before drawing fresh independent coupling seeds U_1,...,U_n. Suppose normalized displacements d_i=||G(U_i)-G'(U_i)||/sqrt(D) satisfy a known almost-sure bound 0<=d_i<=b. Hoeffding's inequality gives, with probability at least 1-alpha,

E[d] <= min{b, mean(d_i) + b sqrt(log(1/alpha)/(2n))} =: U_delta.

On the same event,

|ES_normalized(Q;P)-ES_normalized(Q';P)| <= 2 U_delta.

A valid early rejection rule is 2 U_delta < eta. Record n, alpha, b, empirical mean displacement, U_delta, eta, generator checkpoint identifiers, preprocessing, and the independent seed source. This test uses no target data, so it cannot leak target test observations. It can reject a candidate for a specified score margin without computing that score.

For outputs in [0,1]^D, take b=1. For outputs in [-1,1]^D, take b=2. These bounds apply only if these ranges hold for the actual evaluated images. Clipping must be the declared shared evaluation preprocessing; a clipped metric certificate says nothing about an unclipped metric. An empirical maximum displacement is not a valid almost-sure b.

If generators have unbounded outputs, do not apply Hoeffding using their observed range. A separately justified variance bound Var(d)<=v instead yields the one-sided Cantelli upper bound E[d] <= mean(d_i)+sqrt(v(1-alpha)/(n alpha)); without a range, tail, or moment bound, finite sample displacements alone do not furnish a distribution-free finite upper confidence bound for their population mean.

For K prespecified frozen candidates, use alpha/K per candidate for a simultaneous confidence guarantee. For an adaptive sequence, freeze each candidate before its fresh independent diagnostic seeds and allocate alpha_j with sum_j alpha_j<=alpha. Conditional Hoeffding followed by a union bound controls the whole sequence. Reusing the same seeds while optimizing candidates invalidates the stated frozen-candidate population guarantee. B3 remains a deterministic statement about those reused samples.

A failure to reject is inconclusive. The bound can exceed the true gain. It does not prove score improvement, equal treatment of the compared methods, downstream prediction accuracy, or performance on another metric. The rejection statement concerns only the chosen margin.

## D. Exact angular gain dilution with Gaussian complement

This example is stronger than the displacement bound: it shows that a real fixed active discrepancy can have a polynomially vanishing direct observation-space proper-score gain.

Let A be uniform on {(1,0),(-1,0),(0,1),(0,-1)} in R^2. Let C have the distribution of A rotated by pi/4. Let B be independent N(0,sigma^2 I_m), with sigma>0 and m>=1. Set P=law(A,B) and Q=law(C,B); independent draws of P or Q always use independent complements. Let

R = ||N(0,2 sigma^2 I_m)||,
h_m(v) = E[sqrt(R^2+v)], for v>=0.

The excess proper score, Delta_m=ES(Q;P)-ES(P;P), has the exact formula

Delta_m = (1/2) h_m(2-sqrt(2)) + (1/2) h_m(2+sqrt(2))
          - (1/4) h_m(0) - (1/2) h_m(2) - (1/4) h_m(4).

D1. For two independently drawn square vertices, squared distance takes values 0,2,4 with probabilities 1/4,1/2,1/4.

D2. Between a square and a rotated-square vertex, squared distance takes values 2-sqrt(2),2+sqrt(2), each with probability 1/2.

D3. P-P and Q-Q active distance laws coincide. Insert D1 and D2 into the definition of the score and subtract ES(P;P), obtaining the displayed exact expression.

D4. If V_cross and V_self are these two squared-distance variables, their moments satisfy:

power j       0    1    2    3    4    5
E V_cross^j   1    2    6   20   68  232
E V_self^j    1    2    6   20   72  272.

These entries follow by expanding (2-sqrt(2))^j+(2+sqrt(2))^j and dividing by two, and by evaluating (1/2)2^j+(1/4)4^j for j>=1. The first nonzero difference is -4 at j=4.

D5. For r>0 and v>=0, Taylor expansion through degree four gives

sqrt(r^2+v) = r + v/(2r) - v^2/(8r^3) + v^3/(16r^5)
              - 5v^4/(128r^7) + remainder,

with 0<=remainder<=7v^5/(256r^9). The bound follows from the fifth derivative 105/(32 (r^2+v)^(9/2)) and Taylor's integral or Lagrange remainder, divided by 5!.

D6. For m>9, inverse moments exist, and D4-D5 imply

Delta_m = (5/32) E[R^(-7)] + e_m,
|e_m| <= (441/32) E[R^(-9)].

The remainder constant is (7/256)(232+272)=441/32. It is deliberately conservative because it bounds both positive remainders separately.

D7. Direct chi-distribution integration gives, for 0<s<m,

E[R^(-s)] = (2 sigma)^(-s) Gamma((m-s)/2)/Gamma(m/2).

E[R^(-s)] is asymptotic to (sigma sqrt(2m))^(-s) as m tends to infinity with fixed sigma. The ratio E[R^(-9)]/E[R^(-7)] is exactly 1/[2 sigma^2(m-9)]. This gives

Delta_m ~ (5/32)(sigma sqrt(2m))^(-7).

With D=m+2 and distance normalization sqrt(D), the excess is Theta(D^(-4)). The two active distributions differ at every m. Correcting Q exactly to P removes that active error completely, but its observation-space normalized energy gain tends rapidly to zero. The projection onto the first two coordinates retains the same nonzero score discrepancy regardless of m. A projected metric may diagnose the active failure, but cannot substitute for the declared observation-space success criterion.

The example preserves the active mean and covariance and even the first three moments of squared cross/self distances; this explains the much stronger dilution than an O(D^(-1/2)) displacement bound. This decay law applies to the specified example.

## Smooth full-dimensional, nonlinear-generated version

For tau>0, replace A by A_tau=A+tau Z and C by C_tau=C+tau Z', where Z,Z' are independent standard Gaussian vectors in R^2. The resulting active laws are positive smooth non-Gaussian four-component Gaussian mixtures. Both have the same mean and covariance. The joint laws with B now have positive smooth densities on all of R^(m+2).

Each active law admits an explicit triangular Gaussian transport: first apply the inverse marginal CDF to Phi(U_1), then the inverse conditional CDF at the resulting first coordinate to Phi(U_2), for independent standard Gaussian U_1,U_2. The mixture conditional and marginal densities are positive and smooth, so these inverses exist and the transport is differentiable with positive triangular Jacobian. This transport is nonlinear because its output law is non-Gaussian. Multiplying the remaining Gaussian input coordinates by sigma produces the full-dimensional distribution. No VAE or discrete label input is required by this transport construction.

An exact formula still holds with h_m(v) replaced by

h_{m,tau}(v)=E[sqrt(R^2+||c+W||^2)],

where ||c||^2=v and W~N(0,2 tau^2 I_2) is independent of R. Rotational invariance makes this expectation depend only on v.

For integer j>=0, conditional E[||c+W||^(2j)] is a polynomial in v of degree j with leading coefficient one. More explicitly it is

sum_{l=0}^j binom(j,l)^2 l! (4 tau^2)^l v^(j-l).

This follows by integrating a two-dimensional noncentral Gaussian radial moment, or by expanding the independent Gaussian coordinates. The first three cross/self squared-distance moment differences remain zero and the fourth remains -4. All fifth moments are finite. Repeating D5-D7 yields the same leading constant (5/32)E[R^(-7)] and an O(E[R^(-9)]) remainder, now with a finite tau-dependent bound. The normalized Theta(D^(-4)) dilution persists for fixed tau>0 and sigma>0, with smooth non-Gaussian distributions generated by nonlinear invertible transports.

This smooth construction uses a linear rotation to relate the two active laws. It proves dilution for nonlinear-generated distributions; it does not claim that every nonlinear angular correction has the same moment cancellations. The general displacement certificate in A applies to nonlinear angular corrections without those cancellations.

## E. Implication for the current experiment

Implement only the paired-output displacement diagnostic and its valid confidence calculation. Specify the practical energy margin before observing its diagnostic draws. If the bound clears the rejection threshold, record that the candidate cannot attain that direct observation-space improvement margin under the frozen generator distributions, with the stated confidence. Preserve the candidate's other measured improvements separately. Do not describe rejection for this metric as disproving every generative or representation benefit.

The angular example motivates checking the direct score rather than assuming an improved latent dependence statistic must materially improve global energy. It is not evidence that the existing implementation has the example's exact fourth-moment symmetry or asymptotic rate.

## Verification record

- The displacement constant was independently stress-checked with the point-mass/mixture example, giving ratio 2-epsilon.
- The angular squared-distance moments through degree five were numerically checked using direct binomial values; floating-point differences at the zero entries were approximately 1e-15. The table above uses their exact algebraic values.
- All proofs use visible definitions and elementary inequalities. No literature novelty claim is made.
- Completed nonauthor review: energy_independent_check.md verifies the score estimator coefficient, Taylor remainder constant, inverse-chi scaling, moment-polynomial identity, and confidence conditions.
