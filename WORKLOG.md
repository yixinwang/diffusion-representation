# Research work log

This file records repository actions for the FIQ-FM and QALT research program. Each entry distinguishes inspection, edits, experiments, reversions, and publication actions.

## 2026-08-22

- Set the research objective: audit, repair, implement, and fairly evaluate both FIQ-FM and QALT, with matched data, prior, information, compute accounting, and sealed test data.
- Inspected the supplied FIQ-FM and QALT drafts and the experiment-iteration instructions.
- Inspected repository status and history. Preserved all pre-existing tracked deletions, modified result files, and untracked artifacts pending provenance review.
- Found that the current `master` predates several remote FIQ-FM research branches; no branch was merged or checked out.
- Found hard-coded service credentials in the untracked cluster launcher and replaced them with fail-fast checks for externally supplied environment variables. No credential was used.
- Found that the current MNIST and fractal artifacts favor the latent baseline on their main reported metrics; these artifacts do not establish either requested method.

## 2026-08-23

- Created a detached audit worktree for the remote gauge-fixed FIQ-FM branch. This did not alter the main worktree.
- Tried to run the FIQ-FM verification on an allocated CPU node. Two requests failed before allocation because of invalid memory/account settings; they produced no experimental result.
- Re-ran on account `cis260243p` in `RM-shared`. All five unit tests passed and the synthetic smoke experiment completed. The digits smoke experiment trained but crashed in aggregation with `KeyError: 'linear_probe'`; the checked-in digits summary is therefore not reproducible from its stated command.
- Fast-forwarded `master` from `25b3a75` to the isolated remote FIQ-FM research package at `38ec08e`. The merge only added `fiq_fm/`; all pre-existing deletions, modifications, and untracked artifacts remained unchanged.
- Fixed the digits aggregation schema mismatch and made randomized distribution metrics use common subsamples and projections across methods within each seed. This removes method-dependent evaluation randomness from paired comparisons.
- Added a shared research spine, claim ledger, approach registry, and an adversarial QALT package. The QALT pilot requires optimized exact-solver and pooled-estimator baselines to tie QALT in quality; this closes the draft's universal strict-quality claim while preserving the testable compute claim.
- Marked the archived FIQ-FM tables and manuscript empirical statements as unverified. Their result report references missing `seed_*.json` files, and the original digits command failed; the historical tables were preserved rather than deleted.
- Repaired the FIQ-FM Wasserstein perturbation proposition by placing the Lipschitz assumption on the learned field used by the proof. Narrowed the residual claim from generic gauge recovery to partition recovery on fixed axes; the implementation only permutes axes, and the synthetic benchmark's unequal residual variances do not test generic rotation recovery.
- Completed independent proof and provenance audits. Demoted both old FIQ tables to provenance-incomplete historical snapshots, restricted the VAE separation to a shared fixed chart and residual family, made model initialization deterministic before construction, made validation use train-fitted feature normalization, and aligned the empirical graph score with held-out explained variance.
- Added a canonical audited QALT LaTeX note. It states the conditional compute theorem, the finite-Euler separation, the optimized-control countertheorem, pooling limits, and the no-new-information result for equal-dimensional bijections.
- An allocated-node joint test command exposed a duplicate `test_core.py` import collision between the FIQ and QALT folders. Renamed the QALT test module to remove the collision. The subsequent temporary smoke runs completed for both FIQ datasets and all five QALT oracle checks passed; smoke metrics are diagnostic only and are not promoted.
- Extended the QALT runner provenance record before confirmation: Git state, exact source hashes, software/platform information, Slurm job ID, command line, seed-manifest hash, and runtime are now emitted with every run.
- QALT's three-page audited proof note compiled on an allocated node. The same build exposed a malformed Gaussian expression in the inherited FIQ source; corrected it before rerunning the proof build.
- The combined allocated-node suite passed all eight FIQ/QALT tests. A subsequent FIQ proof build exposed one missing manuscript macro; added the definition and retained the failure in this log rather than treating the partial build as success.
- The FIQ proof note then compiled to 11 pages. Tightened the archived synthetic table after the build reported a 28-point overfull box; a clean warning audit remains part of the next build.
- Renamed the audited fixed-axis construction from “Gauge-Fixed” to FIQ-FM because the current code does not recover a generic residual rotation. Added section bridges required by the manuscript style audit and rewrote the archived digits paragraph so historical significance labels cannot be read as current evidence.
- Ran the mandatory Blei-style source/claims and writing audit. After revisions, the notes score 18/20 and 19/20 with no remaining hard failure; the audit records that external novelty citations and modern empirical evidence are still pending.
- Visually inspected all 11 FIQ pages and all three QALT pages after a clean two-pass build. Tables fit within the page and equations are readable. The inspection found and demoted one remaining present-tense interpretation of the archived digits representation numbers.
