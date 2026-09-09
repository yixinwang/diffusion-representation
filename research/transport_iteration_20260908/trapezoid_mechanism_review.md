The six-cell Lane A mechanism experiment completed, and independent population/numerical audits passed. In all three positive-signal worlds, the learned continuous-context model recovered all 1,440 true residual pairs and achieved lower ideal joint KL than its binned, constant and product restrictions. In all three zero-mean worlds, graph discovery fell back in every block and all four methods tied the product model. This is a deliberate failure case outside the positive-signal assumption, not a hidden exception.

This evaluates a specified nonlinear, non-Gaussian synthetic family with three fresh fitting seeds paired across two worlds. It does not compare with a trained neural full flow, latent diffusion/FM, native image/video model or learned semantic representation. Four independently fitted restricted methods share each observed fitting bank and public information. The 64-source numerical banks are not empirical quality test sets; population KL is for the ideal continuous densities, not KL between an ideal law and finite-precision atomic samples.

Corrected job 45579788 completed with exit 0:0 on r284 in 2:42, reserving four RM-shared CPU cores and 8000M under a 20-minute limit. Source commit `effc2c565b050193561978c9bce686b3a3f996eb` and all 22 source files matched externally before submission and independently afterward. Preflight passed 19 tests in 26.54 seconds. All 24 fits finished before any population evaluation; the freeze receipt records all six cells. Runner wall time was 154.33 seconds, including 12.19 seconds for evaluation/final hashing. Slurm sampled batch RSS was 220,956 KiB versus process high-water 616,732 KiB; these mechanisms are not interchangeable.

The first job 45579430 failed after three seconds before Python/preflight/data creation because login SBATCH_EXPORT=NONE suppressed required variables. Its log/accounting remain preserved. The authorized second submission explicitly exported the frozen revision and a new attempt2 output directory. No scientific source or hyperparameter changed, and the failed job did not perform a fit.

| Seed | World | Continuous joint KL | Binned | Constant | Product | Continuous correct pairs |
|---|---|---:|---:|---:|---:|---:|
| 13109101 | positive | 0.185491720 | 0.202708356 | 2.809558105 | 94.672233799 | 1440/1440 |
| 13109101 | zero_mean | 2.193057156 | 2.193057156 | 2.193057156 | 2.193057156 | 0/1440 |
| 13109102 | positive | 0.190414473 | 0.214968765 | 2.807955712 | 94.672212079 | 1440/1440 |
| 13109102 | zero_mean | 2.193035436 | 2.193035436 | 2.193035436 | 2.193035436 | 0/1440 |
| 13109103 | positive | 0.189415710 | 0.209184865 | 2.811826478 | 94.671569406 | 1440/1440 |
| 13109103 | zero_mean | 2.192392763 | 2.192392763 | 2.192392763 | 2.192392763 | 0/1440 |

Positive-signal continuous residual dependence is learned effectively within this restricted family. In the zero-mean world, the residual joint KL remains 2.0276049124 nats before adding the fitted-root error, despite conditional dependence being present. The mechanism therefore does not solve graph discovery when unconditional feature correlation cancels. No seed average is used to conceal this falsifier.

Independent population evaluation integrates the exact mixed distribution of psi(U): half uniform on [−sqrt(3/2),sqrt(3/2)] plus quarter mass at each endpoint. A tensor-product quadrature computes true entropy and fitted cross-entropy directly from log1p, with correct/incorrect edge overlap accounting. It does not import the registered moment-series evaluator. Context integration respects root/estimator breakpoints and compares orders 48/80; the mixed-feature integral compares orders 32/48. Across all 24 models, maximum discrepancy from the registered series evaluator is 3.84e-13 nats and maximum independent refinement discrepancy is 2.27e-13. These are ordinary numerical checks, not interval certificates or a population confidence guarantee.

All 24 saved forward/inverse numerical banks are finite and inside the strict unit cube. Source roundtrip maximum is 4.10e-12; determinant cancellation maximum is 1.91e-11; Gaussian change-of-variables log-density agreement is 4.55e-13. All six exact-copy output/determinant/density controls are bitwise equal. All six 64×3072 Gaussian sources regenerate byte-exact locally, and paired worlds share the same source for each seed. The copy control is an equality, not a beatable learned baseline.

Per-fit wall times in seconds (minimum–maximum over the three seeds; single ordered measurements, not a randomized latency benchmark):

| World | Continuous | Binned | Constant | Product |
|---|---:|---:|---:|---:|
| positive | 0.825378–0.957689 | 0.819332–0.918406 | 0.809354–0.924161 | 0.017833–0.020501 |
| zero_mean | 0.484177–0.486655 | 0.464807–0.486014 | 0.443773–0.473908 | 0.017132–0.018960 |

The optimized product restriction is far cheaper to fit because it does not discover a graph. Product fit timing also excludes generic Learner.fit validation, as disclosed prospectively. Continuous fitting is not uniformly faster than the other non-product methods. Data/source setup and the one-batch sampling clocks are preserved separately; no intrinsic efficiency or optimized speed-frontier conclusion follows.

All 188 original artifact hashes, totaling 1,314,835,151 bytes, were independently recomputed on PSC, including the twelve full 4000×3072 fitting-source/observation arrays. The downloaded archive contains 178 files (176 inventoried payloads plus status and ARTIFACTS), 135,214,740 bytes, with local source/state/numerical hashes checked. The twelve large fitting arrays remain on PSC with exact hashes/sizes in ARTIFACTS.json; fitted coefficients were not independently refitted in this audit. Full remote path: /ocean/projects/mth250006p/ywang26/diffusion-results/20260909-trapezoid-mechanism-attempt2. Local records: work/psc-trapezoid-mechanism/results.

Independent scripts are work/psc-trapezoid-mechanism/audit.py and independent-population.py; machinecheck.json, remote-artifact-verification.json, launch/failure/submission records and numerical arrays preserve the evidence. No new fitting, generation, neural evaluation, real-data access or scheduler submission occurred during this audit.
