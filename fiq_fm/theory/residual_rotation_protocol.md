# Preregistered residual-rotation pilot

This study isolates the unresolved residual gauge while freezing the exact nonlinear active sampler. It can establish learned block-subspace recovery and one-shot block-fiber efficiency. It cannot establish active-quotient discovery, image/video performance, or quality dominance over an unrestricted full-covariance fiber.

## Data-generating law

Every observation starts from the same six-dimensional prior

```text
(U, E) ~ Normal(0, I_6),  U in R^2, E in R^4.
```

The exact active sampler is the odd triangular diffeomorphism

```text
Z1 = 1.3 U1 + 0.35 tanh(U1)
Z2 = 1.1 U2 + 0.40 tanh(U1).
```

It is nonlinear and non-Gaussian, and its Jacobian determinant is strictly positive. Let

```text
A = [[1, 0], [0, -1]],    B = [[0, 1], [1, 0]],
h(v,w) = (tanh(v), tanh(w)) / sqrt(1 + tanh(v)^2 + tanh(w)^2).
```

Set `(a1,b1)=h(Z1,Z2)` and `(a2,b2)=h(0.8 Z1+0.6 Z2,-0.6 Z1+0.8 Z2)`. The two true residual blocks have

```text
Sigma_j(Z) = I_2 + 0.65 (a_j(Z) A + b_j(Z) B),
Sigma(Z) = Sigma_1(Z) direct-sum Sigma_2(Z).
```

The minimum eigenvalue exceeds `0.35`. Central symmetry makes every coefficient mean zero, so `E Sigma(Z)=I_4` and all residual marginal eigenvalues are tied. Within-block conditional covariance matrices need not commute.

Each unit exposes the same underlying data under two provisional residual charts:

- a signed-permutation rotation;
- a generic Haar rotation, accepted without rejecting easy draws.

Train, validation, and sealed test samples are independent. The two arms within a unit share the same source draws and target law.

## Methods

The exact active `Z` and the same generic nonlinear feature dictionary are supplied to every learned method.

1. `oracle_block`: true residual chart and learned `2+2` block covariance.
2. `permutation_block`: current conditional-dependence matching restricted to permutations, then the same block covariance estimator.
3. `jbd_block`: learned orthogonal joint block diagonalizer, then the same block covariance estimator.
4. `oracle_diagonal`: true residual chart and learned diagonal covariance.
5. `provisional_full`: unrefined provisional chart and learned unrestricted `4x4` covariance.
6. `bayes`: analytic conditional NLL only; this is a non-learned lower bound.

The full-covariance control must tie oracle quality in the population. A learned full-covariance loss is classified as an estimation/optimization effect, not structural dominance.

## Joint block diagonalizer

The permutation and JBD methods use the same train-fitted feature map and conditional second-moment observations. The implementation will:

1. standardize active inputs using training statistics only;
2. form a frozen generic polynomial/trigonometric/random-projection feature dictionary;
3. center residual outer products and estimate feature--second-moment contrasts on training data;
4. extract four leading centered symmetric contrast matrices by a frozen singular-value decomposition;
5. form the commutator Gram operator on the nine-dimensional space of traceless symmetric `4x4` matrices;
6. take its lowest eigenmatrix and split its two positive from two negative eigendirections;
7. abstain when the train/validation feature rank or commutant eigengap falls below a frozen threshold;
8. fit diagonal, block, or full conditional covariance from the same features and ridge penalty.

The population separator is `H*=O diag(I_2,-I_2) O^T / 2`. It is the unique traceless symmetric commutant up to sign when each block's contrast family is irreducible and the two block representations are inequivalent. One contrast, a commuting contrast family, or two equivalent noncommuting blocks cannot identify the registered partition; executable counterexamples cover all three failures.

Truth covariance, truth rotations, and test metrics are unavailable to every learned-chart and model-selection function. Evaluation matches block projectors rather than individual columns because within-block rotations and block swaps are unidentified.

## Development and confirmation

Development uses seeds `0..4` and may tune feature count, ridge, contrast rank, eigengap abstention, and numerical clipping. Development output is never promoted.

After those values are frozen and pushed, confirmation uses 30 independent units with seeds `100..129`:

- 12,000 training observations;
- 4,000 validation observations;
- 8,000 sealed test observations;
- both rotation arms in every unit;
- all methods and all units retained.

A crash remains a failed unit until the exact seed is rerun after a logged code fix. The pre-fix artifact is preserved.

## Equal information, prior, and randomness

Every method receives the same active coordinates, residual observations, feature dictionary, train/validation/test membership, and total six-dimensional standard Gaussian prior. Oracle rotations enter only oracle methods and synthetic evaluation. Learned methods receive no group labels beyond the registered two-block size.

If samples are generated, every method must accept the same stored `(U,E)` arrays rather than call method-specific random samplers. All secondary randomized metrics use common evaluator randomness.

## Primary metrics

All NLLs are test conditional NLL in nats per residual dimension. Primary unit-level metrics are:

- excess NLL over analytic Bayes;
- paired NLL difference between JBD and permutation under Haar rotation;
- JBD-block versus oracle-block equivalence;
- full-covariance versus oracle-block equivalence;
- oracle-diagonal minus oracle-block NLL;
- fraction of the wrong-chart block NLL gap closed by JBD;
- matched block-projector error and maximum principal-angle sine;
- normalized true conditional cross-block leakage;
- data-derived held-out cross-block contrast loss;
- analytic conditional Gaussian squared Wasserstein error.

Synthetic truth is used only for the Bayes and diagnostic metrics after every learned state is frozen. Timing and memory are reported separately for chart estimation, covariance fitting, and generation. Covariance-head output sizes are 4 for diagonal, 6 for `2+2` block, and 10 for full covariance; asymptotic residual transforms are `O(q)`, `O(qb)`, and `O(q^2)`.

## Inference and gates

All 30 independent units enter paired intervals. Strict comparisons use Holm-corrected 95% intervals. Equivalence uses a two-one-sided-test margin of `0.02` nat per residual dimension, fixed before confirmation.

Promote residual-gauge recovery only if every gate holds:

1. under signed permutation, permutation and JBD block methods are equivalent to oracle block;
2. under Haar rotation, JBD block is equivalent to oracle block;
3. under Haar rotation, JBD beats permutation with a positive corrected lower confidence bound;
4. provisional full covariance is equivalent to oracle block;
5. oracle diagonal has a positive NLL gap;
6. the JBD upper confidence bound is below `0.05` for normalized leakage and below `0.10` for block-projector error;
7. oracle/full validation NLL is close enough to Bayes NLL to show that the assay is not underfit.

If JBD fails to close the Haar oracle gap, stop before learned active-quotient or image scaling and revise or reject the rotation mechanism. Passing this assay permits—but does not itself validate—the subsequent learned nonlinear quotient study.
