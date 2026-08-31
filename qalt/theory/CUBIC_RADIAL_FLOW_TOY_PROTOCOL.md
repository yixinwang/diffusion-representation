# Cubic radial flow aligned-toy development protocol

Status: frozen on 2026-08-30 after the independent algebra and runner review
and before any Monte Carlo result from the runner below. The review corrected
the balanced-context standard error, replaced an invalid Gaussian Fisher
correlation interval with a vector-level sandwich interval, and required the
control command to verify a passed witness. This protocol follows the
user-requested empirical loop; it is not attributed to a Blei publication.

## Prior result and first failed layer

The normalized B1-v2 development study failed. Increasing the common EM
maximum from 200 to 1,000 made the strong `A8`, `I8`, and `D4` controls
converge, but changed `NLL(B4)-NLL(A8)` by only `1.269e-7` nat/detail. The
first failed scientific layer is therefore the conditional representation,
not the EM iteration limit. The B4 radial PIT deviation is about `0.815`, its
angular deviations are `0.092--0.097`, and its cross-band residual-energy
correlations are `0.135--0.269`. Its current Gaussian-mixture sampler is also
many-to-one and is not a reversible flow.

## One changed layer and checked claim

Keep nine standard-normal base coordinates, the three contiguous RGB detail
bands, the coarse context, and every outer multiresolution layer fixed. Change
only one local representation layer from an affine Gaussian map to

\[
    x=s(1+a\lVert z\rVert^2)z,\qquad z\sim N(0,I_9),\quad s>0,\ a\geq 0.
\]

For `a >= 0` this is a global bijection. If `r=||z||`, its Jacobian has eight
tangential eigenvalues `s(1+a r^2)` and one radial eigenvalue
`s(1+3a r^2)`. The exact log determinant is

\[
  9\log s+8\log(1+a r^2)+\log(1+3a r^2).
\]

The inverse radius solves `||x||/s = r+a r^3`; for `a>0`,

\[
r={2\over\sqrt{3a}}\sinh\!\left[{1\over3}\operatorname{asinh}
  \left({3\sqrt{3a}\over2}{\lVert x\rVert\over s}\right)\right].
\]

The independent mathematical check verified the inverse, determinant, and
all numeric predictions below. Conditional `s(c)` and `a(c)` preserve a
triangular reversible full map when the unchanged coarse variable `c` is
available in both directions and the parameters do not depend on the detail
being transformed.

## Frozen synthetic distribution and comparison

The downstream endpoint in this aligned toy is normalized held-out log
density per detail coefficient. The primary same-information comparison is
the population-optimal full-affine Gaussian, with the same nine Gaussian base
coordinates and context-specific mean and full covariance. For the spherical
target above it is `N(0, s^2 v(a) I_9)`, where

\[
v(a)=1+22a+143a^2.
\]

The primary binary contexts use `s=1`, `a(0)=0.030`, and `a(1)=0.038`, with
equal observations in each context. The exact target-minus-Gaussian expected
log-density advantages are `0.0189439` and `0.0262225` nat/coefficient. The
rigorous Jensen lower bounds are `0.0123598773928` and `0.0171573472409`.
The predicted correlations between distinct three-coordinate band energies
are `0.234760900963` and `0.268616555888`.

The null control has `a=0`, where the models tie exactly. The low-headroom
control has `a=0.0138`, expected advantage `0.00568412`, below the practical
margin, and energy correlation about `0.135`. A marginal-preserving control
independently permutes the second and third band-energy rows after sampling;
it retains each empirical band-energy marginal exactly while removing their
joint pairing. The layer leaves every sample direction unchanged, so an
angular improvement would reject the proposed energy-only mechanism.

An unrestricted nonlinear one-layer 9D flow can implement the exact same map
and tie it. This study therefore tests separation from the strongest affine
Gaussian, not superiority over every eligible nonlinear flow. The rival
explanation is that marginal heavy tails or an angular coupling, rather than
a shared radius, cause the development-data failures.

## Independent units, split, margins, and decisions

Independent nine-dimensional vectors are the scientific units. Development
uses seeds `3100,...,3104`, with 100,000 vectors per context and seed. Random
streams are `100*development_seed+stream`, where streams 0 and 1 are the two
primary contexts, streams 2 and 3 are the null and low-headroom conditions,
and streams 10 and 11 are the two permutation controls. Every actual stream
and its sufficient statistics are written to the result.
Synthetic confirmation seeds `4100,...,4129` are reserved and will not be run
in this child. The official CIFAR-10 test batch remains unopened.

The witness passes only if all of the following hold:

1. the pooled one-sided 95% lower confidence limit for the paired normalized
   log-density advantage across the two primary contexts exceeds `0.01`;
2. both empirical off-diagonal band-energy correlations differ from their
   checked values by less than `0.01`, and their one-sided 95% lower limits
   exceed `0.05`;
3. the maximum forward/inverse and forward/inverse-log-determinant errors are
   at most `1e-10`, the numerical-Jacobian unit test passes at `1e-8`, and the
   samplewise direction change is at most `1e-12`.

If the witness passes, run the already specified controls. They pass only if
the `a=0` pointwise density difference is at most `1e-12`, the upper 95%
confidence limit at `a=0.0138` is below `0.01`, every permuted off-diagonal
energy correlation has absolute value below `0.01`, and each permuted band
retains its original empirical energy values exactly. These are component
checks, not a new primary endpoint.

The equal-context log-density estimate is the mean of the two context means;
its standard error is one half the square root of the sum of their squared
independent-vector standard errors. Correlation limits use the
independent-vector influence function

\[
{(X-\bar X)(Y-\bar Y)\over s_Xs_Y}
-{\rho\over2}\left\{{(X-\bar X)^2\over s_X^2}
+{(Y-\bar Y)^2\over s_Y^2}\right\},
\]

not a bivariate-normal Fisher interval. All lower and upper decision limits
use the one-sided 95% critical value `1.6448536269514722`. No parameter is
estimated from the simulated outcomes. Failure of any required check stops
this layer without retuning. A pass permits only a prospective development
fit on real training data, with image-level uncertainty and stronger
nonlinear controls; it does not permit CIFAR confirmation or full-flow claims.

## Compute budget and stop rule

The complete witness, controls, tests, and aggregation may use at most two CPU
node-hours and no GPU. The runner must verify that its protocol,
implementation, tests, and command are committed and unchanged, and it must
record a machine-executed numerical-Jacobian test. Run the witness first and
preserve its result. The controls must take that witness file as input and
verify its pass decision, protocol hash, source hashes, source commit, seeds,
counts, dimension, and unopened-CIFAR statement before sampling. Do not
increase the sample size or change `a` after seeing results. Stop after the
controls and diagnose the first failure, or open one prospective real-data
child if all toy checks pass.
