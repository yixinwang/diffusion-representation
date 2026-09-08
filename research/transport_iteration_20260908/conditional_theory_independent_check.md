# Independent verification of conditional learnability derivations

Reviewed `work/conditional-learnability.md` independently of its author. No data or model outcomes were inspected. The add-one risk bound, continuous interpolation extension, context-density lower bound, omitted-context floor, and specified-histogram lower-bound constants are correct under the stated assumptions. No change to the principal formulas is required.

## Finite conditional risk

Given a context count s, Jensen applies to the random reciprocal count:
E log[pi_r(s+b)/(N_r+1)] <= log[pi_r(s+b) E(1/(N_r+1))] <= log[(s+b)/(s+1)].
The bound is categorywise, so summing with either the population cell probabilities or another fixed response probability vector is valid. Conditional on the context count, responses have the cell-averaged multinomial distribution. Integrating the constant-on-cell log ratio p_star/q_hat against the true context/response law gives the exact discrete fitting-error decomposition. Random context counts then give b^k(b-1)/(n+1), without a context-probability lower bound. The Holder approximation diameter, chi-square upper bound for KL, ceil-schedule constant 2^(k+1), sum over factors, and Markov conversion all check. Dependence among coordinates within one cluster is never used as extra sample size.

## Interpolation and conditional lower bound

For any selected coordinates ordered by index, their successive conditional densities given earlier selected coordinates are averages of the full predecessor conditionals. Each is bounded below by a. Their product gives marginal context density at least a^k. This step uses the assumed exact graph factorization, which makes each full predecessor conditional equal to its specified graph conditional.

A participating cell center is within one bin width of an interpolated context coordinate; all points in its original cell are within 3/(2b). Combining the response-bin width with k context widths gives the squared diameter (1+9k/4)/b^2. The categorywise Jensen bound above remains valid when weighted by response probabilities at the query context rather than at the table cell. Each cell has mass at least a_c b^(-k), giving the stated fitting bound after the binomial reciprocal identity. Convexity of KL in the fitted density validates the interpolation mixture. The explicit lower bound on fitted bin masses also gives strict monotonicity and continuous inversion of the conditional CDF. The construction is continuous and differentiable almost everywhere, with the stated knot exceptions; it supplies no global C1 claim.

## Omitted context

The KL decomposition into conditional mutual information plus conditional estimation risk is exact. The reconstruction-based upper bound follows from the context conditional minimizing expected KL, followed by the density lower bound and Holder bound. The distant cosine example has uniform selected local conditionals when D>k+1 and the final context excludes coordinate one. The convexity remainder of (1+z)log(1+z)-z gives theta^2/[8(1+|theta|)] after integrating z^2, whose integral is theta^2/4. The floor survives an invertible chart as a statement about KL for the same transformed graph restriction.

## Full-histogram lower bound constants

For a binomial centered count Z with variance v>=1,
E Z^4=3v^2+v[1-6pi(1-pi)] <=4v^2.
For an independent copy, E(Z-Z')^2=2v and E(Z-Z')^4<=14v^2. Paley–Zygmund at half the mean of (Z-Z')^2 gives probability at least 1/14. The shift-independent triangle inequality gives E|Z+c|>=sqrt(v)/28.

KL(pi_vector||q_vector) is at least the squared Hellinger distance using the sum-of-squares convention. Rationalizing square roots bounds that distance below by one half the sum of (pi-q)^2/(pi+q). Cauchy–Schwarz under the expectation gives the displayed ratio involving squared expected absolute error. Under the specified count regime, pi<=1/2, v>=n pi/2, n+K<=2n, and pi+E q<=3pi. Each cell consequently contributes at least
(n pi/2)/[28^2 * (2n)^2 * 6pi] = 1/(37632n).
The cellwise approximation bound is theta^2/[6(1+theta)b^2], since the integrated squared linear-bin error is theta^2/(3b^2). The exact projection decomposition permits adding these lower bounds. The schedule b approximately n^(1/(D+2)) eventually satisfies the count regime for fixed D. Combining its lower bound with the conditional upper bound establishes the stated ratio tending to zero for fixed D>1.

## Scope that must remain explicit

The comparison concerns the specified full joint histogram procedure, which receives the same graph but fails to exploit it. It is not a lower bound for optimized full flows or diffusion. A stochastic latent decoder that copies the conditional tables and graph, counts every decoder-noise coordinate, and uses the same implementation ties the distribution, computational graph, cost, and risk. Low context dimension alone does not bound graph depth. The constants can deteriorate with context size, density lower bounds, ambient dimension, or chart cost. This proof supplies no neural-spline training guarantee and no observed image/video claim.

One formal regularity clarification would improve the source: for observation-space Jacobian formulas, specify a C1 diffeomorphism, or explicit almost-everywhere change-of-variables conditions, for the fixed chart. An arbitrary differentiable bijection alone is weaker than those standard conditions. KL invariance itself only requires a measurable invertible change of variables and does not depend on a nonzero Jacobian. This clarification leaves the estimator risk calculations unchanged.
