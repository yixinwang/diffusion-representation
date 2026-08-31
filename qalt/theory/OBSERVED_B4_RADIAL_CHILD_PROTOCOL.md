# Observed B4 reversible radial child protocol

Status: frozen on 2026-08-30 after independent mathematical review and before
the real-data runner produced a result. This is an adaptive screen on an
already examined repair split. Its intervals have no confirmation coverage.
The procedure follows the user-requested empirical loop and is not attributed
to a Blei publication.

## Prior evidence and first failed layer

The normalized B1-v2 development endpoint failed. Raising the common EM limit
from 200 to 1,000 made the strong scalar controls converge but changed the
decisive gap by only `1.269e-7` nat/detail. B4 retained radial PIT deviation
about `0.815`, angular deviations `0.092--0.097`, and cross-band normalized
energy correlations `0.135--0.269`. Its density is normalized, but its current
categorical recycling sampler is many-to-one and is not a reversible flow.

The aligned cubic-radial toy at source commit
`d6adf93b2689eb34b2163c42f2fb75e652175a4d` passed its likelihood,
dependence, reversibility, null, low-headroom, and marginal-permutation checks.
That result supports a shared-radius diagnostic on its specified target. It
does not establish transfer to Haar residuals.

## Density-preserving B4 coordinate repair

Fit only the required B4 parent through the bitwise-checked B4-only path; do
not spend the budget fitting unused legacy arms. Keep the B4 Haar
decomposition, coarse-only location, four coarse-energy
strata, determinant-one RGB shapes, mixture weights and scales, training-site
sample, data split, and dequantization unchanged. For one fitted B4 band and
stratum, whiten a residual `r` as `x=L^{-1}r`, let `rho=||x||`, and define

\[
 F_B(\rho)=\sum_k w_k F_{\chi_3}(\rho/s_k),\qquad
 t=F_{\chi_3}^{-1}\{F_B(\rho)\},\qquad y=(t/\rho)x.
\]

Use the continuous origin limit. Positive weights and scales make this a
bijection. Its inverse solves
`F_B(rho)=F_chi3(||y||)` by bracketed monotone bisection. Evaluate lower and
upper tails separately with CDF and survival functions; do not clip a
probability to `[epsilon,1-epsilon]`.

The exact forward log determinant is

\[
 \log|DG|=\log q_B(r)-\log\phi_3(y).
\]

Thus the coordinate repair changes the sampler but not the fitted B4 density.
It is a prerequisite, not the scientific intervention. Stop before fitting a
new radial law if committed unit checks, registered extreme-radius round trips,
forward/inverse log determinants, or samplewise density parity exceed `1e-10`.

## One changed density layer and rival

Concatenate the three repaired band coordinates into `y in R^9`. The proposed
layer is the covariance-normalized cubic radial map

\[
 y=v(a)^{-1/2}(1+a\lVert z\rVert^2)z,\qquad
 v(a)=1+22a+143a^2,qquad 0\leq a\leq0.1.
\]

It uses exactly nine Gaussian coordinates and has the checked closed-form
inverse and determinant. Fit one global `a`, shared across all four strata,
by maximizing only

\[
 \sum_i\{\log p_a(y_i)-\log\phi_9(y_i)\}
\]

on the common 250,000 training sites. Use a fixed 101-point grid, refine every
local maximum and both endpoints, and select the largest training value with
ties going to the smaller `a`. Do not assume the objective is unimodal. Stop
if the selected value is zero within `1e-8` or reaches `0.1` within `1e-8`;
do not enlarge the interval.

The residual density is

\[
 \log q_a(r_{1:3}\mid c)=
 \sum_b\log q_{B,b}(r_b\mid c)
 +\log p_a(y)-\log\phi_9(y).
\]

At `a=0`, it equals B4 samplewise.

Fit a same-information rival with the same nine coordinates and one global
tail parameter. Its covariance-normalized multivariate-t density is

\[
p_\nu(y)=
{\Gamma((\nu+9)/2)\over
 \Gamma(\nu/2)\{\pi(\nu-2)\}^{9/2}}
\left(1+{\lVert y\rVert^2\over\nu-2}\right)^{-(\nu+9)/2},
\qquad \nu>2.
\]

Use the exact radial quantile map, not an auxiliary chi-square coordinate.
Search `eta=log(nu-2)` on 101 fixed points from `nu=2.1` to `nu=1000`, refine
all local maxima and endpoints, and include the exact Gaussian endpoint.
Boundary selection at `nu=2.1` stops the rival without widening its range.

Both radial laws change individual repaired-band marginals as well as their
joint dependence. Both preserve the nine-dimensional direction and normalized
band-energy shares. The rival explanation is generic elliptical heavy tails;
the separate falsifier is non-Dirichlet energy shares or angular structure.
A bandwise three-radius control is specified as the next child only if every
screening decision below passes.

## Frozen real-data screen

Use development seed `2100`, 40,000 fitting images, the common maximum of
250,000 fitting sites, and the existing 5,000-image repair holdout with exactly
500 original images per CIFAR class. The original image is the independent
unit. Sites and color coordinates are not independent units. The excluded
5,000-image discovery split cannot enter fitting or scoring. Only CIFAR
training batches `data_batch_1,...,data_batch_5` may be deserialized. The
official `test_batch` remains unopened. Confirmation seeds and data remain
untouched.

Per image, sum all 256 joint nine-coordinate site log densities and divide NLL
contrasts by 2,304 original detail coefficients. The two primary paired
image-level contrasts are

1. `NLL(B4)-NLL(cubic)`, tested above the practical margin `0.01`;
2. `NLL(t)-NLL(cubic)`, tested above `-0.01`, which is cubic noninferiority to
   the parameter-matched rival.

Use balanced-class Welch summaries and one-sided p-values, then Holm-adjust
this family of two at `alpha=0.05`. Also report `NLL(B4)-NLL(t)` without using
it to select the cubic parameter. Passing the first comparison supports a
practical density gain; passing the second plus the efficiency comparison
below is required for cubic-specific attribution.

## Registered component checks

Invert each repair-holdout residual to the proposed Gaussian base. Use the
same held-out arrays for B4, cubic, and t. Report all diagnostics for every
arm, but the following cubic limits are required:

- total-radius chi-square-9 PIT: pointwise maximum deviation at the 19-point
  grid `0.05,...,0.95` at most `0.02`, and simultaneous image-cluster upper
  bound at most `0.03`;
- second and fourth direction moments: pointwise maximum deviation from the
  uniform-sphere answers at most `0.03`, and simultaneous image-cluster upper
  bound at most `0.04`;
- normalized band-energy share first and second moments: pointwise maximum
  deviation from `Dirichlet(3/2,3/2,3/2)` at most `0.03`, and simultaneous
  image-cluster upper bound at most `0.04`;
- the three latent band-energy correlations: maximum absolute point estimate
  at most `0.05`, and simultaneous image-cluster upper bound at most `0.07`.

Use class-stratified image-level standard errors. Use Bonferroni one-sided
normal critical values within each diagnostic family. Direction and share
statistics are samplewise invariant under both registered radial laws. Any
reported improvement in them is a code error. Calibration cannot substitute
for the paired conditional NLL endpoint.

Require no nonfinite score or sample, exact B4 equality at `a=0` within
`1e-10`, conditional forward/inverse and log-determinant errors at most
`1e-10`, and unchanged stratum labels in both directions. The newly added
float64 location and conditional maps must round trip within `1e-10`. The
frozen float32 dequantization/Haar implementation must round trip within
`1e-6`; this separate tolerance is fixed before execution from its numerical
precision. Do not clip reconstructed or generated pixels.

## Conditional-layer CPU check

On the current CPU node, set `OMP_NUM_THREADS=1`. Use the same first 10,000
finite repair vectors, two warmups, and nine timed repetitions. Record median,
quartiles, vectors/second, peak resident memory, and all fit/search calls.
Time scoring and forward-plus-inverse generation separately.

The command must receive and verify the exact source commit chosen at
submission. Record the exclusive ordered five-training-batch file ledger,
the fitted B4 parameter hash, its parameter count and fit trace, and hashes of
all result files. Write the completion record only after every result file is
durable; otherwise preserve a failure record with the available provenance.

The cubic repaired-B4 round trip must take at most `1.10` times the exact
repaired-B4 round trip. The cubic radial round trip must take at most `0.50`
times the t radial round trip. These are local conditional-layer measurements,
not full-flow or GPU efficiency claims.

## Budget and stop rule

This seed-2100 screen may use at most four CPU node-hours, four CPUs, 20 GiB,
and no GPU. Preserve every failure. Stop promotion at the first failed layer
in this order: source/data integrity; B4 coordinate parity; fit boundary;
normalized density and reversibility; paired NLL; radial calibration;
direction/share falsifiers; latent energy dependence; local CPU limit. The
runner may finish all already registered measurements so the failure can be
diagnosed, but it cannot tune or promote a later layer. Do not change any
bound, margin, search interval, diagnostic, or sample count after seeing the
repair result.

If every item passes, run the already required bandwise-radial control on this
same adaptive split, then submit seeds `2101,...,2104`. Even that would not
authorize official-test confirmation or full-image scaling. If density passes
but direction or energy-share checks fail, the first unsupported layer is the
one-global-radius representation. If the radial t ties or beats cubic, retain
generic radial tails as the rival and withhold cubic-specific attribution.
