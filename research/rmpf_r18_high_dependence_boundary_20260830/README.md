# RMPF-R18: source-only high-dependence regime

This append-only milestone preserves the complete R1–R17 record and changes only the development/evaluation regime of the frozen exact normalized all-coordinate no-VAE R16 coupling.

For frozen source-fit state `s(g)`, context `v(g)`, and R16 matrix `B_s`, the pre-outcome statistic is

`H(g)=||0.72 B_s v / sqrt(1+||B_s v||^2)||`.

Thresholds are source-training quantiles.  The preregistered nested tails were `{1/8,1/4,1/2}`, with a minimum of 96 reference and generated examples per seed under the unchanged 1024-sample evaluator.  The smallest tail whose exact true-family oracle proper-energy lower confidence endpoint reached `0.005` would have been selected.  No CIFAR, UCF, replication, or confirmation outcome was allowed in this round.

## Executed result

| Tail mass | Oracle energy gain | Paired 95% interval | Minimum ref/gen count |
|---:|---:|---:|---:|
| 1/8 | 0.00401904 | [-0.00161701, 0.00965509] | 114 / 110 |
| 1/4 | 0.00404556 | [-0.00557284, 0.01366395] | 233 / 232 |
| 1/2 | 0.00270556 | [-0.00360156, 0.00901268] | 506 / 468 |
| Full diagnostic | 0.00209200 | [0.00003259, 0.00415140] | 1024 / 1024 |

No stratum passed.  The oracle mean itself stayed below `0.005` in every selectable tail.  The frozen R16 direct coupling gave the same boundary.

The statistic did order exact NLL and dependence separation: zero-minus-oracle NLL rose from `0.088585` on the full law to `0.107261` in the top eighth, and dependence-error reduction rose from `0.290668` to `0.366982`.  It did not produce a robust proper-energy transition.  This retires the R15/R16/R17 coupling family under the frozen candidate regime rather than changing the margin or opening realistic development.

## Exactness and reproducibility

- Parent R17 `per_seed.csv`, contrasts, diagnostics, and verdict were byte-identical on replay.
- Child replay was byte-identical for all five output files.
- Maximum round-trip error: `1.7763568394002505e-15`.
- Maximum forward/inverse log-Jacobian cancellation: `2.220446049250313e-15`.
- Target invariance of `H`: exact.
- Copied-mechanism mismatch: zero.
- Seeds: `10030–10034`.
- Known-truth execution: 13.41 s, peak RSS 224,164 KiB; exact replay 13.47 s.
- Real development, replication, and confirmation: unopened.

Run `python reproduce.py` in this folder with NumPy, SciPy, and pandas installed.