# Observed B4 covariance-only child protocol

Status: frozen after the seed-2100 radial result and before any covariance
child holdout score. This is a second adaptive study on the already examined
repair split. Its intervals have no confirmation coverage. The procedure
follows the user-requested empirical loop and is not attributed to a Blei
publication.

## Parent failure and first unsupported layer

The source-bound radial child at commit
`a05c4ebb57d99e7e1420e17b36e761d5f7b41a5d` completed. Its first failed layer
was `fit_boundary`: the cubic coefficient selected the frozen upper limit
`a=0.1`. The interval was not widened. Cubic improved B4 by `0.3136291`
nat/detail but was worse than the registered Student rival by `0.1882589`
nat/detail. The Student fit selected `nu=2.1176545`.

The B4, cubic, and Student direction and normalized band-share statistics
were equal samplewise to within `6.67e-16`, as required by their radial form.
Their held-out angular and share deviations were `0.0929461` and `0.2205200`.
These exceed the frozen point limits of `0.03`. No global isotropic radial map
can change either quantity. The next child therefore changes only the failed
nine-coordinate density layer from an isotropic radial law to a zero-mean
linear covariance law. It does not compose with or promote the failed radial
arm. The fixed Student result remains the strongest same-information tail
rival.

## Frozen parent reproduction

Keep the exact reversible B4 coordinate repair, Haar transform, coarse-only
location, four existing coarse-energy strata, dequantization, split, and
common training-site sample. Before fitting covariance, require all of the
following parent values:

- site sample count `250000` and SHA-256
  `ff127f2f793dcb022df1fb27a0c3e6a2659d3ac00dc02375018502d06e6e3327`;
- B4 parameter SHA-256
  `33f130a399324d3f26aca9e8de57eea1fd521f43ed611c819feb162c10361564`;
- fitting-image SHA-256
  `87e33e5bc0465b68d0cd4a7469bad8f5cd51bc2c8209f70d5fdc5ecbb0fc9192`;
- repair-image SHA-256
  `3c1dbca3401fe130455a0a7b3eca9b7de8caa7e9f76f718bcd6c80f5fa80d4ad`;
- B4 holdout-score SHA-256
  `7f510b5f314e1b2731046a156cd2b5b5e62807718612c94b1d16924e1f8664e9`;
- fixed Student parameter `tau=0.4722205736227421` and Student holdout-score
  SHA-256
  `75ea4f0cb1de6bfdb8abb1cbf8015688c5f24eb535dd154b50a859939ddffbba`;
- parent `summary.json` SHA-256
  `9f83e2e8a2cdad2ef7f1409a651e51694ee22be45513bfcf96ae5de644ccc2fa`
  and `COMPLETE.json` SHA-256
  `827fd24188d45cea7a72c7eae0af2d09691d8f2cc87137f868eb33f194efc69f`.

Stop before covariance fitting if any value differs. Do not refit a radial
parameter.

## Checked covariance layer

For a repaired B4 coordinate `y in R^9`, fit the uncentered training second
moment

\[
 C={1\over n}\sum_i y_i y_i^T.
\]

The mean remains fixed at zero. Do not add shrinkage, clipping, or jitter. Let
the unique symmetric positive square root be `R=C^(1/2)`. The exact map and
inverse are

\[
 z=R^{-1}y,\qquad y=Rz.
\]

The forward log determinant is `-0.5 log det(C)`, and the density
increment relative to B4 is

\[
 \log\phi_9(z)-\log\phi_9(y)-{1\over2}\log\det C.
\]

Record the eigenvalues. Stop if the covariance has minimum eigenvalue below
`1e-4`, is not positive definite, or has condition number above `1e6`. Do not
repair a failed matrix after seeing a result.

Fit these arms on the identical 250,000 training coordinates:

1. B4 identity, with no added covariance;
2. one global diagonal covariance, with 9 fitted parameters;
3. one global block covariance containing three independent 3x3 band blocks,
   with 18 fitted parameters;
4. the proposed one global full 9D covariance, with 45 fitted parameters;
5. the fixed Student density rival, with no covariance and no refit.

The diagonal and block arms test whether marginal variance or within-band
covariance, rather than cross-band coupling, explains any gain. These are
controls of the same changed covariance layer, not additional scientific
revisions. Four stratum-specific full matrices are not fitted in this round;
they are a possible later child only after a diagnosed conditional-covariance
failure.

## Aligned toy requirement

Before loading CIFAR, test a shared covariance in all four toy strata,

\[
 S_\rho=
 \begin{pmatrix}
 I_3 & \rho I_3 & 0\\
 \rho I_3 & I_3 & 0\\
 0&0&I_3
 \end{pmatrix}.
\]

The diagonal and block population controls are `I_9`, while the global full
map is exact. Since `det(S_rho)=(1-rho^2)^3`, the population gain of full over
either control is

\[
 -{1\over6}\log(1-\rho^2)
\]

nat per coordinate. At `rho=0.30`, this is `0.01572`, above the practical
margin `0.01`; at `rho=0`, the gain is exactly zero. As a negative control,
alternate `+rho` and `-rho` across four equal strata and verify that one
global uncentered covariance is `I_9` and has zero population gain. Unit checks
must reproduce both formulas, the identity null, forward/inverse values, and
determinant cancellation within `1e-10` before the real-data run.

## Frozen data and endpoint

Use seed `2100`, the same 40,000 fitting images, the same common 250,000
fitting sites, and the same 5,000-image repair holdout with 500 images per
CIFAR class. The original image is the independent unit. The excluded 5,000
discovery images cannot enter fitting or scoring. Only
`data_batch_1,...,data_batch_5` may be deserialized. The official `test_batch`
remains untouched.

Per image, sum all 256 joint nine-coordinate site log densities and divide
paired NLL contrasts by 2,304 original detail coefficients. Evaluate one
Holm-adjusted family at `alpha=0.05`:

1. `NLL(B4)-NLL(full)` must exceed `0.01`;
2. `NLL(block)-NLL(full)` must exceed `0.01`;
3. `NLL(Student)-NLL(full)` must exceed `-0.01`, a noninferiority
   requirement against the fixed heavy-tail rival.

Use balanced-class Welch estimates and one-sided p-values. Pair arms within
the same image. For each contrast, also report the frozen two-sided 95% Welch
interval

\[
 \widehat\Delta \mathbin{\pm}
 t_{0.975,\nu}\operatorname{SE}(\widehat\Delta),
\]

with the same balanced-class Welch standard error and Satterthwaite degrees
of freedom used by the one-sided test. If the standard error is exactly zero,
report the degenerate interval `[estimate, estimate]`. These adaptive-repair
intervals are descriptive and have no confirmation coverage. No covariance
diagnostic can replace this conditional NLL endpoint.

After every holdout score is fixed, compute the rejection-only global
zero-mean Gaussian oracle from the holdout uncentered second moment `S_H`:

\[
 g_H^{oracle}={\operatorname{tr}S_H-9-\log\det S_H\over18}
\]

nat per detail coefficient. This is the maximum possible same-array gain of
any global zero-mean Gaussian covariance over B4. If it is at most `0.01`, the
practical B4 endpoint is impossible for the entire registered global
covariance class. Never use `S_H` to fit a reported arm or select a parameter.

## Dependence and calibration checks

Report every diagnostic for every arm. For the proposed global-full base,
require:

- coordinate mean: maximum absolute point deviation from zero at most `0.02`,
  with simultaneous image-cluster upper bound at most `0.03`;
- coordinate uncentered second moment `E[zz^T]`: maximum absolute point
  deviation from identity at most `0.03`, with simultaneous image-cluster
  upper bound at most `0.04`;
- coordinate mean within each of the four existing B4 strata: maximum
  absolute point deviation from zero at most `0.02`, with one simultaneous
  image-cluster upper bound across strata and coordinates at most `0.03`;
- the same uncentered second moment within each of the four existing B4
  strata: maximum absolute point deviation from identity at most `0.03`, with
  one simultaneous image-cluster upper bound across strata and matrix entries
  at most `0.04`;
- total-radius chi-square-9 PIT: maximum point deviation on
  `0.05,...,0.95` at most `0.02`, upper bound at most `0.03`;
- direction second and fourth moments: maximum point deviation at most
  `0.03`, upper bound at most `0.04`;
- normalized band-energy-share first and second moments: maximum point
  deviation at most `0.03`, upper bound at most `0.04`;
- three band-energy correlations: maximum absolute point estimate at most
  `0.05`, upper bound at most `0.07`.

Use class-stratified image-level standard errors and Bonferroni simultaneous
upper bounds within each family. If covariance calibration passes but the
fourth moments, shares, energy correlations, or radial PIT fail, retain
higher-order nonelliptical dependence as the rival explanation.

## Reversibility and CPU checks

On all 1,280,000 repair coordinates, require each covariance forward/inverse
value error and log-determinant cancellation error at most `1e-10`. Inherit
the parent all-holdout B4 inverse result only after every parent hash matches.
Also run the full B4/location/Haar inverse on all 256 sites of the ten lowest
repair record IDs per class. Run an additional B4 inverse on the 32 largest
B4-coordinate radii per stratum, with ties resolved by row order. Record the
probe IDs, class counts, site indices, hashes, and maxima. Require float64
conditional errors at most `1e-10`, unchanged inverse strata, and float32
Haar error at most `1e-6`.

Set `OMP_NUM_THREADS=1`. On the same first 10,000 finite repair coordinates,
use two warmups and nine timed repetitions. Record medians, quartiles,
vectors/second, peak resident memory, and fit/score calls. Proposed full
scoring must take at most `1.10` times B4 scoring, and its full conditional
round trip must take at most `1.10` times the repaired-B4 round trip.

This child may use at most one CPU node-hour, four allocated CPUs with one
computational thread, 8 GiB, and no GPU.

## Stop rule and interpretation

Preserve every result and stop promotion at the first failure in this order:
source/data/parent reproduction; covariance positivity; normalized density
and reversibility; paired NLL family; coordinate mean; aggregate uncentered
second moment; stratum-specific coordinate mean; stratum-specific uncentered
second moment; angular moments; band-energy shares; band-energy correlation;
cumulative radial PIT; local CPU limit. The runner may finish all registered
measurements for diagnosis but cannot tune or promote a later layer.
Before entering each registered layer, record that layer as the current
candidate failure. If an exception prevents the complete summary, the durable
failure record must export that registered `first_failed_layer`, rather than
only a coarse execution phase.

If the block control is within the practical margin, withhold cross-band
attribution; this alone does not establish a positive within-band claim. If
Student remains better, covariance may explain a dependence component but is
not a complete density successor. A stratum-specific covariance child may be
opened when the frozen stratum-specific uncentered-second-moment check
diagnoses pooling cancellation and the stratum-specific mean check passes,
whether or not global full passes NLL. If a stratum-specific mean fails, the
diagnosed layer is conditional location, which covariance whitening cannot
repair. The alternating-sign toy is the required qualitative pattern for that
diagnosis.
If all checks pass, run independent development seeds before any confirmation.
Even a pass does not establish full-image generation quality, long-range
spatial dependence, or confirmation performance.
