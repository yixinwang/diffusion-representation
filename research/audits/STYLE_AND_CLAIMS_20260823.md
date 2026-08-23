# Source, claims, and writing audit — 2026-08-23

Scope: `fiq_fm/theory/FIQ_FM_verified.tex` and `qalt/theory/QALT_repaired.tex`, including their input sections. This is an internal verification note, not manuscript content.

## Hard-failure review and revisions

- Removed universal quality, efficiency, and VAE-representation claims that were not established by a theorem or study.
- Restricted exact generation to declared factorization classes and the diagonal-residual KL gap to a shared fixed chart, active marginal, conditioning, and mean.
- Put the Lipschitz assumption on the learned vector field used by the Wasserstein proof.
- Kept the inherited residual graph restricted to fixed-axis partition recovery. Added a separate commutant-spectral theorem for generic two-block rotations under explicit response-rank, irreducibility, inequivalence, and eigengap conditions; the tied-marginal Haar study remains development evidence until confirmation.
- Marked both old FIQ empirical tables as provenance-incomplete historical snapshots and removed current significance language.
- Added bridging prose before every section's first subsection, defined all notation exposed by compilation, and removed the table overfull box.
- Stated the QALT optimized-control counterexample next to the finite-Euler separation, so the restricted comparator cannot be mistaken for universal dominance.
- Compiled both proof PDFs. FIQ has no unresolved references or overfull/underfull material after adding the residual-rotation theorem and primary JBD attribution; its only remaining build message is LaTeX changing an `h` float to `ht`. QALT compiled without a reported layout failure.

No hard failure remains in the audited scope.

The scorecards were rerun after adding the registered QALT confirmation section. The section reports every registered tie, strict gap, misspecification reversal, and the operation proxy's measurement boundary; it does not promote the oracle result to learned, hardware, image, or video evidence. Scores are unchanged.

## Source-and-claims scorecard

| Item | Score | Evidence |
|---|---:|---|
| Primary evidence | 1 | This is original derivation plus audits of the two supplied drafts and the primary JBD source; a full image/video literature synthesis is still pending. |
| Model | 2 | Ambient/latent variables, active and fiber variables, fixed charts, conditioning, and source noise are explicit. |
| Objective | 2 | Forward KL, Wasserstein error, flow regression, and cost comparisons have stated direction and conditioning. |
| Algorithm | 2 | Active integration, endpoint fiber sampling, partitioning, and estimator pooling are separated. |
| Assumptions | 2 | Fixed chart, realizability, block family, solver, Lipschitzness, matching margin, and equal-information restrictions are visible. |
| Claim type | 2 | Theorem, conditional derivation, archived experiment, smoke test, and proposal are labeled separately. |
| Attribution | 2 | No existing result is claimed as authorial work. |
| Citation | 2 | Internal audit targets are exact and the matrix joint block diagonalization mechanism is attributed to its primary AISTATS source. |
| Retrieval weighting | 2 | No claim uses venue prestige as evidence and private drafts are not treated as publications. |
| Limits | 2 | Discovery, generic rotation, unrestricted baselines, nonlinear decoded strictness, and image/video evidence gaps are explicit. |

Total: **19/20**, pass threshold met with no hard failure.

## Writing scorecard

| Item | Score | Evidence |
|---|---:|---|
| Genre | 2 | Both artifacts are declared empirical-audit theory notes. |
| Reader | 2 | Objects and claim boundaries are introduced before technical results. |
| Story | 2 | Each abstract states the problem, mechanism, surviving theorem, counterexample, evidence status, and next implication. |
| Straight line | 2 | The exposition moves from the clean factorization to identification, perturbation, evidence, and limits. |
| Paragraph jobs | 2 | Paragraphs have one principal mathematical or empirical role. |
| Sentence flow | 2 | Definitions precede consequences and restrictions follow the claims they qualify. |
| Mathematics | 2 | Symbols are named, displays are motivated, and compilation found no undefined notation after repair. |
| Claims | 2 | Consequences and comparator restrictions replace novelty or optimality adjectives. |
| Word choice | 2 | No priority formula, “former/latter,” or unsupported superlative remains. |
| Evidence presentation | 1 | QALT confirmation and the failed FIQ rotation confirmation are reproducible; the projector/rank failures are reported next to the passing quality gates, and no modern image/video evidence exists yet. |

Total: **19/20**, pass threshold met with no hard failure.

## Research-advice check

- Impact is phrased as the outcome of spending repeated computation only on nonlinear active coordinates while retaining full randomness.
- The ladder advances from oracle algebra to learned nonlinear quotient, residual-rotation recovery, then image and video only after gates pass.
- The smallest informative pilots and stopping rules are written before confirmation runs.
- Failures already changed the theorem, algorithm name, residual-basis claim, evaluator, and baseline set.
- Image/video architectures and datasets remain proposals until the mechanism studies pass.
