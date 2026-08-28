# RMPF-R9 sketch derivation and falsifier

## Frozen model

For each unchanged R8 lifting group,

\[
r=(y-X\theta-g)/\sigma,\qquad y=X\theta+g+\sigma r,\qquad \sigma>0.
\]

Changing the estimator of \(\theta,\sigma\) cannot break invertibility or normalization. The log-Jacobian remains the R8 coarse/detail scale sum plus the unchanged exact R7 coupled-copula endpoint term.

## Source-frozen leverage sketch

For one grouped ridge regression,

\[
F(\Theta)=\|X\Theta-Y\|_F^2+\lambda\|\Theta\|_F^2,
\]

with \(d=2\), \(q\le9\), and \(\lambda=10^{-2}\). For channel ridge leverage \(\ell_{ic}\), use \(\ell_i=\max_c\ell_{ic}\) and

\[
p_i=(1-\eta)\ell_i/\sum_j\ell_j+\eta/N,\qquad \eta=.05.
\]

Because \(\sum_i\ell_i\le3d\), \(p_i\ge\beta\ell_{ic}/d\) with \(\beta=(1-\eta)/3\). The frozen sufficient width is

\[
m\ge\frac{8d}{3\beta\varepsilon^2}\log\frac{2dq}{\delta}=13,792
\]

for \(\varepsilon=.10\), \(\delta=.01\); execution uses 16,384 rows.

An \(\varepsilon\)-embedding gives

\[
F(\widetilde\Theta)\le\frac{1+\varepsilon}{1-\varepsilon}F(\Theta_\star),
\]

and an absolute ridge bound

\[
\|\widetilde\Theta-\Theta_\star\|_F
\le
\sqrt{\frac{2\varepsilon}{1-\varepsilon}\frac{F(\Theta_\star)}{\lambda}}.
\]

It does **not** imply a useful relative bound when \(\|\Theta_\star\|_F\) is near zero. This is the premise that fails empirically.

## Exact experiment-aligned falsifier

Before systems promotion, both image and video must satisfy:

1. inverse error at most \(10^{-9}\) and log-Jacobian cancellation at most \(10^{-8}\);
2. local alpha/beta maximum relative error at most .08 and median at most .03;
3. scale relative error at most .03;
4. full-row objective inflation at most 1.01;
5. validation-NLL deviation at most .0025 nat per visible dimension;
6. unchanged R7 endpoint active mask.

The final balanced child passes inverse, log-Jacobian, objective, NLL, scale, and endpoint-mask gates, but local maximum errors are 1.4912 for images and 1.1482 for video. Exact coefficient norms are small, so almost identical objectives permit large relative coefficient changes.

## Cost falsifier

Seven fit repetitions are required. Image median and upper bootstrap endpoint must both be at most 1.447782 s, with at least 6/7 repetitions passing. Video uses the analogous 2.214945 s limit and 6/7 rule. Both domains must also pass full-flow and positive-rate latent-flow latency, memory, bytes and operation gates.

Images fail at 1.677744 s [1.552918,1.851390], 1/7 passes. Video has a passing point median but fails the interval and 6/7 rules.

## No-small-revision boundary

A 262,144-row balanced sketch still has maximum local relative error .6491 and takes 2.0949 s. Full-row exact recovery invokes the original vectorized R8 fitter. Therefore no sampled row count tested reaches the frozen parameter gate before erasing the fitting advantage. Reopening requires a different estimand/parameterization or a fused exact sufficient-statistic kernel, not a new sketch seed, tolerance, or row count.