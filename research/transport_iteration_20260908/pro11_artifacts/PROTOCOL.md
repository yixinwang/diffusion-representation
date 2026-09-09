# Pro11 proposed matched-information changed-law falsifier

## Status

**Not executed.** The already executed three-seed checks in this package are analytic/synthetic learning checks, not this tuned comparison. No PSC job is requested or submitted by any file here. This protocol requires a separate implementation/review of the global-FM adapter and timing harness before execution. It is not represented as a completed end-to-end benchmark.

## One paired change, not an architecture search

Use D=3072, root192, four residual blocks720. Draw an arbitrary secret perfect matching independently in each block, unknown root histogram masses, unknown signs s_b, and unknown phases alpha_b. Keep all truth objects evaluator-side. The root C1 is uniform in this specific fixture but all models estimate its histogram, along with the other191 roots.

The positive world uses theta_b(c)=s_b[0.35+0.075 sin(2pi c+alpha_b)]. The changed world sets **only the intercept to zero**. Retain root law, graph, phases, signs, amplitude, dimensions, and source rows exactly within a paired seed. Both worlds retain all random coordinates and identical residual margins. The Lipschitz and positivity conditions remain valid. The graph marginal-signal condition does not remain valid. Total dependence strength also changes; this is not claimed to be a strength-matched experiment.

The purpose is to falsify the extension “the marginal nonlinear-moment matching rule learns arbitrary context-varying pair dependence.” Failure of that extension does not contradict the theorem, whose signal margin is explicit. The unchanged nonlinear variation makes the surviving dependence directly checkable rather than hiding it in a teacher latent variable.

## Fresh data and access separation

Use three **new, unexecuted** master seeds:1111011,1111012,1111013. For each master, allocate non-overlapping child streams for truth parameters, fit generation, repair generation, final-score generation, training randomness, repair sampling, and final sampling. Record seed derivation, array shapes/dtypes, exact code hashes and SHA256 hashes. Reveal evaluator truth seeds only after model selection is frozen.

For each world create 4000 fit arrays, 1000 fresh repair arrays, and 2000 independent final-score arrays. Root/graph/shape/source-generation parameters are fixed before creating any array. Pair the two worlds using their common underlying fixture Gaussian banks, but refit every model separately from that world's **observed arrays only**. Do not feed the fixture Gaussian sources, true matching, true theta, drift, root probabilities, labels, phases, or evaluator summaries to any learner.

All methods receive the same public model class, coordinate grouping, scalar-context location, positive-density range, and optional public cosine/sine feature functions. None receives fitted candidate parameters except the explicitly labelled exact-copy control. The fitting executable's data inputs are the observed fit file and public configuration only. Repair is available only to the selection evaluator, which returns the prespecified scalar score. Final-score data remain closed until every selected checkpoint/hash is frozen.

Do not touch native CIFAR test/discovery arrays or reused native repair files. The word “repair” here means a newly generated synthetic validation split, not a relabelling of an existing native split.

## Required arms

1. **The theorem estimator:** root add-one; 2000 structural arrays; 2000 context arrays; K32, a=.35, kappa=.45, fixed public cosine basis. No tuning of a or K after seeing results. Failed matching gate is a declared product fallback, never an oracle graph replacement.
2. **Same-information analytic pair learner:** independently estimate root, matching and context functions from the same observed fit arrays. Include conditional moments with public features (1, sin(2pi C1), cos(2pi C1)), not only their constant component. Whiten these three regressors by their empirical fit Gram matrix, with an explicitly fixed numerical eigenvalue floor. Candidate-edge scores are their squared whitened cross-moment norms. Use maximum-weight perfect matching, then fit either a clipped three-feature regression or a clipped K-bin context head. K in {8,16,32,64}; choose by the common repair score. Include the theorem estimator itself as an admissible member of this analytic family. This prevents an artificial win against a knowingly weaker analytic baseline. The signs and phases remain estimated. No recovery or superiority of this control on the changed law is presumed.
3. **Exact stochastic decoder copy:** same fitted root, graph, coefficients, full Gaussian input, CDF-grid convention, and all fit/setup/sample costs as arm1. It must tie arm1's law and observable predictor. Verify source alignment and generated outputs, not merely one scalar score. A non-tie is an implementation failure, not a scientific win.
4. **Global FM:** a globally interacting velocity model, not a pair-independent or locally receptive-field-restricted field. Use the repository's globally attending FM machinery rather than deliberately shrinking it to1472 parameters. Permit a joint3072-coordinate version and a version with the same learned exact root plus a global2880-coordinate conditional residual field as two repair-tuned choices. All public nonlinear/context features are available as deterministic inputs, with their cost included. No teacher graph or velocity is available. A learned invertible analysis is allowed only when its fit/setup/parameters are charged; there is no product-floor argument against it.

Retain the fitted constant-context ablation from the standalone checks as a diagnostic, not as a substitute for the strong analytic control. A scalar-product decoder supplies the theorem's restricted lower-bound comparator, not the only empirical opponent.

## Budget and tuning contract

The prospective cap is **360 synchronized wall seconds per independently trained method family, per world and seed**, on identical hardware. Include model construction, preprocessing, root estimation, graph discovery, feature caches, transfers, compilation/first uses, optimizer creation, all tuning fits, selection-time generation, and any last-step overrun. Do not count360 seconds separately for every candidate hyperparameter. Unused budget is reported, not filled with dummy work.

For global FM keep one globally attending width fixed from the inspected baseline rather than searching architectures; vary learning rate in {0.001,0.0003} and the two root treatments above. Use a prespecified successive-halving allocation: at most45 seconds to each of the four configurations, then at most90 seconds to each of the best two including selection overhead, with a single hard family cap360. A method may exhaust the cap before the nominal rungs finish; retain and report the checkpoint and failure/censoring state. Numerical failures remain in the inventory. Selection and any continuation are charged, and no final-score result affects them.

Evaluate actual FM budgets4/8/16/32/64 with counted velocity calls. When an implementation admits a mathematical endpoint simplification, report mathematical stages and nontrivial kernels separately; do not transfer Pro10's tilted-normal endpoint identity to an unrelated fitted field. Tune sampler budget on repair under the same family cap. Report all completed budgets after freezing, not just the selected one.

For the analytic family share only its own already charged sufficient-statistic computations between its head candidates; graph matching and conditional moment extraction are charged. Give the same opportunities to FM, including public features. Report actual parameter counts rather than forcing a tiny analytic model and a global neural field into an unusable parameter equality. The primary fairness statement is equal information and capped total fit/selection cost; parameter count, exposures, and source draws are separate disclosures.

After selection, measure steady full-generation latency and memory at batch1 and64, with all stages resident as required, randomized method order, common full-D Gaussian banks, and30 repetitions. Report first use/compilation separately and full costs including mandatory finite checks. These final latency measurements are not training-to-quality results.

## A common, proper sample score without pretending FM has a tractable discrete density

All selected samplers use the same independent2000x3072 final Gaussian source bank. Never inject true observed context at generation. Normalize complete-array Euclidean distances by sqrt(D).

The repair-selection score is a prespecified positive-definite kernel score, computable for every sampler:

    S_k(Q,x) = E[k(Y,Y')] - 2 E[k(x,Y)],    Y,Y' iid Q.

Use k = k_RBF + k_dep. k_RBF is a Gaussian kernel on allD coordinates with bandwidth equal to the median full-array fitting distance (computed on a fixed first256 fit records and frozen). For k_dep, let h_ij(u)=psi(u_i)psi(u_j), let w(c)=(1,sin(2pi c),cos(2pi c))/sqrt(2), and define

    k_dep(x,x') = [w(c1) dot w(c1')]/M
                   * sum_groups sum_{i<j in group} h_ij(u) h_ij(u').

Every candidate edge is included; no true graph is used. It is a finite-feature inner-product kernel. The RBF addition makes the population score strictly proper on the complete cube, not only a moment-matching test. The all-edge sum can be computed without a million-entry feature tensor using

    sum_{i<j} a_i a_j = ((sum_i a_i)**2 - sum_i a_i**2)/2,
    a_i = psi(u_i) psi(u'_i).

Use off-diagonal U-statistics for generated-generated kernel expectations and all cross pairs; preserve the finite banks and raw kernel/score summaries. Apply the same score, sources, record order, and numerical precision to all arms. This score is an explicit implementation proposal, not a metric that was run in the standalone checks. The public sine/cosine context features are given to all fitters; they expose the zero-mean counterexample without disclosing its graph or coefficients.

On the fresh final-score split report this fixed kernel score, normalized energy score, root discrepancies, and conditional nonlinear moment errors. Report seedwise results and paired differences; three training seeds do not license broad significance claims. Resampling uncertainty must treat complete arrays and Gaussian source rows as the sampling units, accounting for reused source rows in U-statistics, not falsely regard all pair features as independent.

## KL and representation diagnostics

For analytic arms with valid normalized densities, evaluate population joint KL using evaluator-only truth and independently refined quadrature. Decompose root and residual terms; include the actual grid-density correction or score the actual continuous grid density. Label any oracle statistic. A discrete FM integrator does **not** inherit the continuous ODE density: do not print FM joint KL without a verified inverse/Jacobian for that actual finite-step map or a separate rigorously controlled method. The common sample score above provides the fair comparison without such a fiction.

For each observed residual coordinate j, mask j and score prediction of psi(U_j) using the remaining observations. The analytic theorem predictor is theta_hat_b(C1)psi(U_hat_partner). Its excess error can be evaluated against the known conditional expectation evaluator-side. To compare representation learning across unrelated models, freeze representations and train the same small supervised probe on a separately budgeted common set of masked observed fit arrays; all targets are observed psi(U_j), never hidden latents. No semantic or rotation-recovery claim follows.

## Prespecified decisions

On the positive law the exact-population product floor is about95.874883 nats, while the theorem class bound is0.619102 expected full-array nats. A single seed above an expected bound is not a theorem refutation. Diagnose violations of class assumptions, root estimation, graph failure, the histogram risk calculation, and numerical implementation separately. Refuting an expectation claim statistically requires enough independently trained worlds or a deterministic proof counterexample, not three seeds alone.

On the changed law the population pair-edge means vanish and arm1 should return the product fallback at these settings. Conditional product KL is about2.028220 nats; a robust lower certificate is M*0.075^2/4=2.025. Unchanged margins and retained conditional dependence rule out explaining the effect by scalar marginal repair. Recovery by the conditional analytic control or global FM would establish an empirical limitation of the marginal graph learner, not their universal dominance. If all controls fail, retain that outcome without claiming a method-specific separation.

Reject any claim of broad latent-efficiency advantage unless a separately frozen native comparison improves the declared quality and observed-prediction endpoints against the best repair-tuned strong controls at the charged budget. Copy ties are mandatory. Do not reinterpret the existing mixed native KID/PRDC or checkpoint backend speed results as evidence for this synthetic class.
