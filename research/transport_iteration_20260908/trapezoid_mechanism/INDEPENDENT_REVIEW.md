# Prospective Lane A source review

Reviewed `work/trapezoid_mechanism/{evaluate.py,fixture.py,run.py,test_evaluate.py}` against the separately reviewed trapezoid learner. No full fits, jobs, or population study were run in this review. The three test passes are the parent's reported execution, not an independent rerun. This is a prospective source review before freezing the study.

## Exact population formula

Let $A=\sqrt{3/2}$ and $Z=\psi(U)\psi(V)$ for independent uniform coordinates. The feature distribution is a half-weight uniform distribution on $[-A,A]$ plus mass $1/4$ at each endpoint. Its odd moments vanish and

$$E\psi(U)^{2k}=A^{2k}\frac{k+1}{2k+1},\qquad EZ^{2k}=(3/2)^{2k}\left(\frac{k+1}{2k+1}\right)^2.$$

Writing this last moment as $m_{2k}$, direct absolutely convergent log-series expansion gives

$$L(\eta)=E\log(1+\eta Z)=-\sum_{k\ge1}\frac{\eta^{2k}m_{2k}}{2k},$$
$$C(\eta)=EZ\log(1+\eta Z)=\sum_{k\ge1}\frac{\eta^{2k-1}m_{2k}}{2k-1},$$
$$T(\theta)=E(1+\theta Z)\log(1+\theta Z)=\sum_{k\ge1}\frac{\theta^{2k}m_{2k}}{2k(2k-1)}.$$

These agree with all three returned arrays in `scalar_terms`. Since $|\eta Z|\le.675$, absolute convergence is uniform. For 128 terms, a conservative tail bound for $L$ is $.675^{258}/[258(1-.675^2)]$, and for $C$ it is $1.5\,.675^{257}/[257(1-.675^2)]$. These are much smaller than float64 roundoff. This observation controls series truncation mathematically; it does not enclose floating arithmetic or the context quadrature.

Conditional on the root context, every true residual coordinate is uniform and distinct true pairs are independent. An incorrectly selected edge connects two different true pairs, hence has independent uniform endpoint marginals. Even if the union of the two matchings contains long cycles, each incorrect edge still has that marginal property. Expected log density is a sum of edge terms, so no independence among fitted edges under truth is needed. If a block has $m$ true pairs, $k$ fitted pairs and $o$ correctly recovered pairs, its conditional KL is exactly

$$mT(\theta)-kL(\eta)-o\theta C(\eta).$$

This proves the overlap-count formula in `population_kl`. It covers wrong perfect matchings and empty product fallback. It relies on loaded states actually being perfect matchings or empty; `Learner` validates that condition. It would not justify arbitrary dependent latent laws or repeated fitted edges.

Root KL is the sum of categorical histogram KL terms because true and fitted roots factorize and use the same within-cell uniform densities. The factor `root_bins` cancels. The residual conditional contribution is integrated under the **true** first-root density, correctly supplied by `root_probabilities` from evaluator-owned truth, not by fitted roots.

Context integration splits at true root-density boundaries and fitted context-bin boundaries or interpolation centers. Constant extension at both ends is included. Between breakpoints the fitted coefficient is linear or constant and the true sine is smooth. The 32/64-order comparison is a useful ordinary-numerical accuracy diagnostic, not an interval certificate. No finite-evaluation confidence interval is needed for this analytic integral, but its numerical error remains uncertified.

## Fixture membership and fairness

The initial normalized Uniform(.5,1.5) root weights did not guarantee density at least .5. Before freezing, the parent changed them to $.5/K+.5\widetilde p_k$ and preserved first-root uniformity. The revised construction guarantees the declared root-density floor. The first context is exactly uniform in the ideal law; therefore the positive case has mean coefficient magnitude .35, and the zero-offset case has exactly zero unconditional mean for every phase. The true slope is $2\pi(.075)<.5$, and the positive coefficient magnitude is at most .425, below .45.

The zero-mean world violates the positive signal assumption deliberately. It is a falsifier for unconditional graph recovery, not a counterexample to the theorem's stated class. Every result, including fallback or unexpectedly recovered graphs, must be retained. No graph or true phase is passed to any fit call. All methods receive the same observed arrays and known public feature/chart/configuration; all use the same full-dimensional Gaussian source at generation.

Continuous, binned and constant learners independently repeat their graph/root fits. The product baseline directly fits only roots and incurs no artificial graph overhead. It is therefore an optimized product restriction. The continuous exact-copy control reconstructs the complete fitted state, generates on the same 64 full-dimensional sources, and checks output, determinant and density equality. It is expected to tie, and is not an independently trained competitive method.

There are 24 fitted models across six cells, and all fits finish before any population-risk evaluation. Truth used to generate observations is isolated from estimator APIs; truth files are available only for later evaluation in the runner's logic. Positive and zero-mean worlds share Gaussian source/fixture seeds intentionally. Thus six cells are not six independent repetitions of one condition: each world has three declared fitting seeds, with paired construction across worlds.

## Accounting and preservation

Actual fit source arrays, observed arrays, fitted states and post-fit numerical banks are saved. The 64 generated arrays are for inverse/Jacobian/copy checks, not empirical quality estimation. A single batch timing and ordered single-fit clocks are diagnostics; they are not randomized latency benchmarks or evidence of general efficiency dominance. Source/data/import time is reported separately from per-fit clocks, while the all-fits elapsed record includes it. The product clock excludes the generic validation done in `Learner.fit`, so any eventual precise comparative timing claim would need an explicitly matched validation boundary.

Source snapshots are compared with the requested exact Git commit before imported study code is used. The frozen runner must reside at the documented repository depth; the scratch copy is not intended to run as the final launcher. New output directories prevent overwriting prior studies. Existing completed arrays/states survive failure; an interrupted in-progress NumPy fitting call may not have a serializable partial estimator, and the protocol should not imply otherwise.

Two small runner hardening requests were sent to the parent and the final source changes were checked: explicitly reject nonfinite metrics and numerical outputs rather than relying on comparisons with thresholds (NaN comparisons can silently pass), and record the current evaluation cell/arm before each evaluation so failure status does not retain the last fitting cell. The runner now rejects nonfinite arrays and metrics, preserves a raw evaluation receipt before the gate, and updates evaluation cell/arm state before each arm. The parent owns the execution of final tests. Apart from those requested checks, no blocking evaluator algebra, teacher leakage, or copy-control defect was found.

This Lane A validates a deliberately specified nonlinear, non-Gaussian mechanism and its restricted product/constant ablations. It contains neither the proposed packed-sign graph nor a trained global FM comparator. It must not be described as the full fair-comparator experiment, a native-image experiment, or a proof that the algorithm beats arbitrary latent diffusion/flow matching.

Final preflight source review: the repository launcher executes the copied reference, learner, and evaluator tests after source authentication and before constructing study observations; the subprocess receives only snapshot module paths. Preflight stdout is preserved and a nonzero status fails before study fitting. This source review does not independently rerun the 19-test preflight.
