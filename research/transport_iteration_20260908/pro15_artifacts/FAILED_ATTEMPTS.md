# Preserved failures and limits

## Actual empirical failures

The complete original 72 fitted states and all scores are in results/run_initial.
Nothing was retuned, overwritten or regenerated to replace an unfavorable cell.

At the original 2,000 graph / 2,000 parameter budget, spectral, histogram-likelihood
and old-unconditional discovery selected no edges in both nonnull laws and both
seeds. Their whole-joint densities therefore fell back to the corresponding fitted
product density. The null is not counted as a discovery failure: its active graph
is empty and its hidden structural matching is unidentifiable.

At 65,536 graph / 4,096 parameter arrays, the harmonic method still selected no
edges in the registered smooth off-basis law. Its population signal in frequency
1 is exactly zero. The eight-bin likelihood comparator recovered all eight pairs
in both seeds and reduced KL; this falsifies a broad robustness claim for the
harmonic method. The old unconditional statistic selected no edges everywhere.

Even with the privileged graph supplied, the harmonic parameter family failed
the off-basis law: fitted joint KL was worse than product in all four off-basis
seed/budget cells. Supplying the graph does not repair a misspecified context head.
The histogram oracle and histogram-likelihood method are exactly matched at the
large-budget recovered-graph cells. At harmonic large-budget cells, differences
between spectral and histogram KL are context-head differences, not evidence of
superior graph recovery. Equal graphs with the same parameter fitter imply the
same statistical estimator, up to summation-order rounding.

At the null, oracle-graph fits unnecessarily introduced estimated dependence and
were worse than product. This is retained rather than called a graph-recovery
success. Every exact-copy decoder tied the spectral decoder bitwise.

The original full-size 2,000-graph-array experiment is not rescued. The new full-
size finite guarantee requires roughly 70,000 graph arrays, not 2,000. The Fano
bound is about exact whole-matching recovery averaged over matchings; it does
not rule out all 4,000-array methods, approximate graphs, or graph-free estimators.

## Execution failures

There were no unexpected Python fitting, inverse, quadrature, or test failures
in this standalone implementation's original execution. All 36 initial tests
passed; all 72 fits and evaluations completed on their first run. Original
executed sources are in attempts/initial_sources; original test, run and later
read-only audit logs are preserved. The audit does not refit or draw new sources.
Tests deliberately reject invalid coefficients, overlapping/noninteger edges,
nonuniform first-root states, nonfinite inputs and Gaussian-CDF saturation.
These deliberate checks are not misrepresented as earlier accidental failures.

## Access and delivery limitations

A direct shell clone failed before downloading repository content:

    fatal: unable to access 'https://github.com/yixinwang/diffusion-representation.git/': Could not resolve host: github.com

This line is transcribed from the observed tool output; the original shell
attempt was not redirected into a log file. Repository inspection subsequently
used the GitHub connector successfully. No local byte-identical historical
checkout is claimed.

GitHub tool discovery returned 48 exposed read/search/metadata functions and no
write or branch-creation action. A query for create returned none. Plugin lookup
found the already-enabled GitHub connector, not an additional write capability.
A skill-catalog query did not expose another GitHub workflow. No nonexistent
write action was invoked, and no failed-write response or permission denial is
fabricated. This is a session capability limitation, not a claim that the user's
GitHub account or other Pro sessions cannot publish.

The original complete downloadable archive is the delivery. No Pro15 branch or
commit was created in this session; no active branch was edited. The prior Pro13
publication status and binary deliveries were not assumed, and the concurrently
running native study was not accessed or modified. No PSC, SSH or scheduler tool
was used. The failed local public-network clone was not a PSC/SSH connection.
