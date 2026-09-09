# Source provenance, novelty limits, and execution scope

## Repository inspection

Repository: `yixinwang/diffusion-representation`.
Requested branch: `agent/observation-transport-audit-20260908`.
Inspected immutable commit: `f024cc56651da7dc161d555117d9373a4ac8fa99`.
Commit title: `Preserve verified full-generator speed and mixed perceptual results`.
Git tree: `84dda0404334b72344043ba6bd6848f5e6247d97`.
The commit's UTC timestamp is2026-09-09T02:10:04Z, corresponding to September8 in America/Detroit.

Read through the connected GitHub read-only fetch tool:

- `research/transport_iteration_20260908/README.md` (Git blob `504999dd58aec7a1c45fd4d7bf2dc6f1a1f9fdf1`).
- `research/transport_iteration_20260908/cached_prefix_kl_decomposition.md` (Git blob `dae5494010eb9a7812f9490bd88480c618eac4a6`).
- `research/transport_iteration_20260908/dense_model_review.md` (Git blob `a98a29d499290ed893d3c22ca0976c2f38591762`).
- `research/transport_iteration_20260908/perceptual_review.md`.
- `research/transport_iteration_20260908/global_innovation_flow.md` (Git blob `3a56993bb38e2e1bb4dd4a493ad62b325f06e88e`).
- `qalt/experiments/observed_flow_pilot/SHARED_PROTOCOL.md`.
- The retrieved training/generation/evaluation sections of `qalt/experiments/observed_flow_pilot/run_shared.py`. The tool's long-file response was truncated near the later orchestration section; no claim is made to have inspected that unseen remainder.
- The new `check_dense_model_results.py` and `check_perceptual_results.py` patches in the commit response, including array/hash/metric verification code. The native saved-array audits were not rerun here.

An initially guessed `psc_dense_model_review.md` returned404; the correctly listed `dense_model_review.md` was then read. This was a file-location error, not an experimental failure. No repository clone/edit/write, native payload retrieval, model loading, protected-data access, or PSC connection/job occurred. Prior Pro9/Pro10 content supplied in the request is not represented as a fresh independent inspection of their entire isolated artifact branches.

Every path above can be located under the immutable base:

    https://github.com/yixinwang/diffusion-representation/blob/f024cc56651da7dc161d555117d9373a4ac8fa99/

Git blob identities above are tool-reported repository metadata, not newly recomputed SHA256 hashes of downloaded source files. The local package SHA256SUMS covers this package's actual bytes only.

The inspected reports support a mixed native perceptual result and bounded frozen-checkpoint timing gains, not training-to-quality or representation superiority. No new native result is claimed in this package. The cached-prefix decomposition is conditional on a fitted analysis; this note instead proves learning only in an explicit fixed observable chart.

## Closest primary sources and what is not novel

**Exact trigonometric copula and its feature-product moment estimator:**
Martial Longla and Mous-Abou Hamadou, *Estimation problems for some perturbations of the independence copula*, arXiv:2308.14282v1,28August2023. Read HTML Sections2–3. Their sine-cosine family contains the precise single-cosine perturbation used here, and their Eq.(7) estimates coefficients by products of the orthogonal features. They develop asymptotic theory for copula-based Markov chains. Neither the cosine copula nor estimating its coefficient by a cosine product is a novelty claim here.

    https://arxiv.org/html/2308.14282v1

**Nonlinear conditional copulas:**
Christian Schellhase and Fabian Spanhel, *Estimating Non-Simplified Vine Copulas Using Penalized Splines*, arXiv:1603.01424v2 (2016), Statistics and Computing (2017). The primary abstract describes estimation of varying conditional copulas with hierarchical B-splines and out-of-sample KL comparisons. Read abstract/metadata; no claim to have reviewed its full PDF in this conversation. Unknown nonlinear context dependence is established modeling territory.

    https://arxiv.org/abs/1603.01424

**Finite-sample graph/density learning:**
Arnab Bhattacharyya, Sutanu Gayen, Eric Price and N.V.Vinodchandran, *Near-Optimal Learning of Tree-Structured Distributions by Chow-Liu*, arXiv:2011.04144v2 (2021). The primary abstract gives finite-alphabet tree-learning KL guarantees and add-one estimation for a specified graph. These results establish important prior finite-learning work; their finite-alphabet theorem is not simply applied to the continuous copula class here. Read abstract/metadata.

    https://arxiv.org/abs/2011.04144

**Current orthogonal-expansion context:**
Angelo Efoevi Koudou, Yves I.Ngounou Bakam and Denys Pommeret, *Lancaster copulas*, arXiv:2607.01558v1,2July2026. Read HTML, including orthogonal/bi-orthogonal expansions and positivity/truncation discussion. This is additional current context, not the origin of the exact cosine family, which is already present in the2023 source above.

    https://arxiv.org/html/2607.01558v1

**Historical origin:**
O.V.Sarmanov, *Generalized normal correlation and two-dimensional Frechet classes*, Doklady Akademii Nauk SSSR168(1),32–35 (1966). Verified original publisher/archive bibliographic record. Its Russian PDF was not inspected. The general positive product perturbation should not be presented as newly invented.

    https://www.mathnet.ru/eng/dan32257

**Latent identification:**
A.Hyvarinen and P.Pajunen, *Nonlinear Independent Component Analysis: Existence and Uniqueness Results*, Neural Networks12(3),429–439 (1999). The author's primary publication page explicitly describes non-uniqueness without temporal structure. The Gaussian-rotation example in this package is a separate direct argument and does not require an unverified application of nonlinear-ICA identification assumptions.

    https://www.cs.helsinki.fi/u/ahyvarin/papers/udl.shtml

No figures or tables from external PDFs were analyzed or reproduced. These sources support conservative novelty boundaries. This package does not establish that its exact composite bound is globally unprecedented; it supplies a derivation and falsifier that can be checked independently.

## Actual local execution

Standalone scripts were authored outside any repository under `/mnt/data/pro11_dependence_theory/`. Python/NumPy/SciPy versions, source SHA256 hashes, seeds, data shapes and Gaussian/observed-array SHA256 hashes are in each JSON result. The exact tested source files have not been revised after the recorded three-seed runs.

Executed:

```
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 python run_checks.py --math
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 python run_checks.py --seed 1109101
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 python run_checks.py --seed 1109102
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 python run_checks.py --seed 1109103
```

All20 mathematical checks passed. All six full-dimensional fits completed; all three zero-mean fixtures failed the structural signal gate and used the documented product fallback. The failure is a retained scientific outcome, not an exception removed from the results.

Three positive and three changed-law checkpoints are retained, together with evaluator truth objects (not supplied to fit) and fresh64x3072 Gaussian/generated round-trip banks. Full4000x3072 training Gaussian/observation banks are **not** bundled; their generating code, exact seeds, dtypes/shapes and byte hashes are retained. Regenerated byte hashes may differ across mathematical libraries/platforms; do not substitute a close numerical match for an exact byte match.

The proposed fresh-fit/repair/final-score comparison in `PROTOCOL.md`, independent tuned analytic control, global FM, GPU timings, and native images were not run. The reported0.17–0.36-second single-CPU fitting intervals are not all-in benchmark timings: source/fixture generation, generation checks, model scoring, serialization and alternative control fitting are separate or unmeasured. No comparative speed conclusion uses those intervals.

A subsequent `audit_saved.py` audit, also executed locally, rechecked all six saved fitted states without fitting. It used direct two-dimensional copula quadrature rather than the learning runner's entropy-series implementation. Maximum disagreement across joint/conditional/root KL and product-floor quantities was7.389644451905042e-13. All saved64-row generated banks regenerated exactly; maximum Gaussian-coordinate round-trip error was1.4432011141707335e-11. Its source hash and checkpoint hashes are in `saved_audit.json`.
