# Pro15: context-weighted matching, registered standalone study

Frozen before the first execution of the empirical runner. This is an independent
fabricated mechanism study, not a PSC replication, native image experiment, or
flow-matching comparison. No PSC, SSH, active branch write, native data access, or
changes to Pro14's concurrent seven-arm native study are permitted.

## Inputs and fixed cells

Two development seeds: 150901 and 150902. Dimension 20: four root coordinates and
two eight-coordinate residual blocks, with four unknown disjoint pairs per block.
This is deliberately much smaller than Lane A's 3,072 dimensions. A separate
64-by-3,072 *constructed-model numerical smoke check* tests the whole transport;
it is not a 3,072-dimensional learned graph experiment.

Three laws, all fixed now: (1) 0.075 sin(2 pi C + phase), (2) exactly zero,
(3) 0.075 tanh(3 sin(6 pi C + phase))/tanh(3). The last law is smooth, nonlinear,
has period 1/3 and has exactly zero projection on the discovery frequency 1.
Block phases, signs and matchings are evaluator-owned random draws, not fit inputs.
The first root coordinate is exactly Uniform(0,1), supplied as a public structural
assumption. Other root coordinates are independent eight-bin densities with
floor 0.5. These assumptions match the relevant fabricated construction, not
established properties of native images. All residual univariate conditional
marginals are uniform. No positive marginal-signal fixture is used.

Budgets: low = 2,000 graph + 2,000 parameter arrays; large = 65,536 graph + 4,096
parameter arrays. Root fitting uses all arrays of the applicable budget. The low
budget uses the first 4,000 arrays of the larger saved bank. Comparisons between
budgets are nested development comparisons, not independent replications.
All three laws use the same saved Gaussian source per seed. Fit sees only the
resulting observed arrays, never paired sources. No rerun or replacement of an
unfavorable seed is allowed.

## Arms and fitting

Eligible: phase-invariant sine/cosine weighted discovery with a 0.01 simultaneous
Bernstein threshold; old unconditional feature Gram with threshold 0.175;
split eight-bin conditional-likelihood-ratio discovery with a 0.01 Bonferroni
e-value threshold; and product. The likelihood comparator splits its graph budget
in half, estimates an edgewise conditional density on one half, and tests it on
the other. It uses an exact mathematical log(1+t) <= t screen before expensive
log evaluation. This is a conventional, strong, equally informed comparator,
not a straw-man unconditional test.

The spectral, old and product arms use three-column harmonic parameter regression
on the separate parameter arrays. The likelihood arm uses eight-bin regression.
These different approximation families and their costs are explicit. Both pool
recovered pairs within each block, using the shared-function assumption. A graph
contains only threshold-passing isolated edges; no forced perfect matching and
no graph supplied to an eligible fit. Estimated theta is constrained to [-0.45,
0.45]; source and output values are never clipped. An observed harmonic design
minimum eigenvalue below 0.5 triggers the registered zero-coefficient fallback.

Ineligible diagnostics: oracle graph with harmonic regression, and oracle graph
with eight-bin regression. These receive the hidden graph only, not phase, sign,
coefficients, or paired sources. Under the null, this structural matching is not
an identifiable active graph. Exact-copy stochastic decoder: reload and copy the
spectral state, preserving every source coordinate. It must tie bitwise.

## Freeze, scoring, arrays, and failures

Write every input source, observed array, fit state, graph statistic, regression
response and diagnostic. Freeze hashes of every fitted state across all 72 fits
before any population evaluation (2 seeds x 3 laws x 2 budgets x 6 fitted arms).
Then evaluate every state, including failures, by exact population factorization
with numerical quadrature. Cross-check with independent direct density quadrature.
Store a common generation source, each generated output, forward and inverse
log determinants, recovered source, log density and copy output. Report graph
correct/missed/false counts, root and whole-joint KL, approximation failure,
complete fit stages, full source-to-output decode and full log-density timing.
Do not rename a graph-kernel latency whole-generator speed. Record serialization,
fixture generation, validation and scoring separately from estimator fit time.

The original theorem's probability statement and its sample requirement are not
inferred from these two seeds. No speed or quality threshold against an unrestricted
latent decoder or trained FM is asserted. The off-basis failure must remain a
failure even when the histogram comparator succeeds.

## Reproducibility and publication

Run locally with Python, NumPy and SciPy only; one BLAS thread is requested in
run.sh. Preserve every executed source version, traceback and partial result if
an attempt fails. Fit code has no evaluator import; this is an API separation,
not OS isolation. Tests and a source-contract check inspect this separation.
Publish only an unchanged complete original package under
research/transport_iteration_20260908/pro15_artifacts/ on a new isolated
pro15-artifacts-20260909 branch, if a write connector exists. Otherwise provide
the complete downloadable archive. No registration-only branch is completion.
