# Independent PSC positive-spline result audit

Source `dc0169f50eb0eddfbd5bff6ace6dc2fb384347a2`; PSC job 45554441. All 36 registered cells completed in 56.250 runner seconds, with reported peak RSS 105.57 MiB. This is a complete run with optimization failures, not a converged 36-cell study.

All 39 payload hashes, the SHA256SUMS ledger hash, and all seven committed source hashes verify. Each per-cell JSON matches its embedded summary record; all 36 unique configurations match the registration. No experiment was rerun. No source/repository/remote files were changed.

## Optimization is the principal failed gate

**Only 14/36 cells reached the requested 1e-4 empirical Frank–Wolfe gap. All 18 local-world cells failed; four distant-world cells failed.** Every failed group exhausted 500 updates; status labels correctly preserve failure. Convexity alone does not imply these capped fits converged. The saved gaps remain useful upper bounds on empirical suboptimality in the stated floating-point calculation, not population certificates.

Group response counts equal independent-array count times shared-site count. The audit checked each final gap against its trace, the weighted per-coordinate gap, convergence flags, iteration caps, and monotone objectives (largest upward change 0). Coefficients satisfy bounds and response-hat normalization (maximum integrated-row error 8.88e-16).

## Descriptive development outcomes

Means across three independently fitted seeds. KL entries are nats per coordinate. Energy scores are raw exploratory point estimates; their size is dominated by marginal spread, and no competing model is timed or scored here. No convergence rate or superiority is inferred.

| World | D | n | Gap-met cells | Mean KL/coordinate | Seed SD | Mean energy |
|---|---:|---:|---:|---:|---:|---:|
| local | 8 | 256 | 0/3 | 0.00740517 | 0.00074222 | 0.19946671 |
| local | 8 | 1024 | 0/3 | 0.00166470 | 0.00015255 | 0.19944532 |
| local | 8 | 4096 | 0/3 | 0.00095409 | 0.00021661 | 0.19910953 |
| local | 32 | 256 | 0/3 | 0.00139690 | 0.00012934 | 0.20319336 |
| local | 32 | 1024 | 0/3 | 0.00089756 | 0.00007171 | 0.20290172 |
| local | 32 | 4096 | 0/3 | 0.00054740 | 0.00015016 | 0.20304535 |
| distant | 8 | 256 | 0/3 | 0.01160746 | 0.00048878 | 0.19898514 |
| distant | 8 | 1024 | 3/3 | 0.00831420 | 0.00100636 | 0.19972502 |
| distant | 8 | 4096 | 2/3 | 0.00743486 | 0.00032792 | 0.20008463 |
| distant | 32 | 256 | 3/3 | 0.00372319 | 0.00057288 | 0.20304405 |
| distant | 32 | 1024 | 3/3 | 0.00198818 | 0.00010266 | 0.20306261 |
| distant | 32 | 4096 | 3/3 | 0.00192843 | 0.00013255 | 0.20313801 |

Local mean KL decreases over the recorded sample sizes in both dimensions, even though every local fit misses its optimization tolerance. Fixed four-bin approximation and optimization errors remain; these three points do not establish the theoretical rate. The earlier histogram experiment used a different graph/teacher and sharing scheme, so its numbers are not a fair matched head-to-head baseline.

## Distant-dependence failure persists

The omitted-context joint lower bound is 0.032007576, independent of D (per-coordinate bounds 0.004000947 at D8 and 0.001000237 at D32). This is a conservative lower bound, not the exact asymptotic limit. A separate deterministic 64/128-node tensor Gaussian quadrature of the known two-coordinate density gives KL(P||Uniform)=0.055204483; the two orders differ by 1.18e-15. This diagnostic calculation uses no experimental arrays or fit, is not a new experimental result, and is not a rigorous interval enclosure.

At n4096, mean joint KL is 0.059478843 for D8 and 0.061709698 for D32. These sit above the analytic lower bound and near the diagnostic uniform-model KL; three sizes do not prove convergence to a plateau.

| Distant n4096 | Observed first/last cosine product | Generated first/last cosine product |
|---|---:|---:|
| D8 | 0.16252129 | -0.00166622 |
| D32 | 0.16819180 | 0.00162305 |

The exact target dependence moment is rho/4=.1625. Generated endpoint moments remain near zero: the local graph cannot represent the distant pair. The local-world generated adjacent-pair means range across cells from 0.1202 to 0.1869 against target.1625; observed pair means and marginal/endpoint summaries are preserved in every cell. Individual generated summaries have only 256 arrays and are noisy; no unregistered significance test is attached.

## Integrity, independence and numerical limits

The source uses separate SeedSequence purpose IDs 1/2/3 for fitting/evaluation/Gaussian inputs, and constructs evaluation only after fitting. All 144 recorded training/evaluation/source/generated array hashes are distinct. Hashes and source inspection support intended separation, but raw arrays were not retained and were not regenerated; the audit does not independently recompute KL, energy, MCSE or moments from raw samples. All sample-count metadata correctly treats complete arrays as independent units; shared responses are explicitly not counted as independent arrays.

- source_roundtrip_max: maximum `1.19018e-10` (limit1e-8).
- logdet_cancellation_max: maximum `6.08281e-10` (limit1e-8).
- normalized_density_identity_max: maximum `7.10543e-15` (limit1e-8).
- All 36 numerical checks pass; all 36 copied stochastic decoders tie samples, log determinants and evaluation densities bitwise.

The implementation reports the first disjoint adjacent pairs (0,1),(2,3),... rather than every sliding adjacent pair. The protocol wording “every adjacent-pair” could be read more broadly; the recorded source and outputs make this precise. This does not affect the primary joint KL or endpoint failure diagnostic.

Conclusion: retain the positive local learning trend as bounded synthetic development evidence, retain all 22 optimization failures, and retain the distant-context representational failure. Neither complete execution nor low finite KL establishes image/video performance, an optimized latent-model advantage, or statistical confirmation.
