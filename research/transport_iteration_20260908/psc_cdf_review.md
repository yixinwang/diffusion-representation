# Independent PSC conditional-CDF result audit

Job 45552606 completed with exit 0:0 in 30 seconds of Slurm elapsed time. Runner wall time was 9.738 seconds; reported process peak RSS was 52.68 MiB.

Source: `ffec36e35c325d36bc9cdfc73b04305f6f485205`. All 38 result-manifest hashes and all seven source-file hashes match their committed Git blobs. Exactly the 36 preregistered cells are present, with no duplicate/missing configuration. All numeric scalar values are finite; bins, priors, evaluation sizes, thresholds and copied-decoder equality were independently checked against the records.

Hash verification establishes artifact integrity. Training/evaluation/sample/table hashes are recorded, but their underlying arrays were intentionally not stored; this audit did not independently regenerate them. No training or real-data access was performed in the audit.

## Descriptive density results

Means and sample standard deviations across three independently fitted seeds; nats per coordinate. These are development summaries, not confidence intervals or convergence-rate estimates.

| World | D | n | Mean KL estimate | Seed SD | Mean joint KL estimate |
|---|---:|---:|---:|---:|---:|
| tree | 8 | 256 | 0.03027765 | 0.00489047 | 0.24222120 |
| tree | 8 | 1024 | 0.01524650 | 0.00037188 | 0.12197200 |
| tree | 8 | 4096 | 0.00743533 | 0.00052417 | 0.05948262 |
| tree | 32 | 256 | 0.03040506 | 0.00045772 | 0.97296179 |
| tree | 32 | 1024 | 0.01567468 | 0.00103670 | 0.50158979 |
| tree | 32 | 4096 | 0.00761958 | 0.00057244 | 0.24382666 |
| distant | 8 | 256 | 0.02069356 | 0.00063961 | 0.16554847 |
| distant | 8 | 1024 | 0.01593915 | 0.00217262 | 0.12751316 |
| distant | 8 | 4096 | 0.01075350 | 0.00045928 | 0.08602804 |
| distant | 32 | 256 | 0.01737218 | 0.00115186 | 0.55590968 |
| distant | 32 | 1024 | 0.01144338 | 0.00022851 | 0.36618810 |
| distant | 32 | 4096 | 0.00665310 | 0.00035329 | 0.21289915 |

Tree mean density error decreases across the three tested training sizes for each dimension. This is descriptive agreement with learning on this specified nonlinear non-Gaussian class; it does not empirically establish the asymptotic rate. Evaluation MC standard errors are conditional on individual trained models and cannot replace variation across training seeds.

The omitted-distant-dependence population floor is 0.032007576 joint nats, 0.004000947 per coordinate at D=8 and 0.001000237 at D=32. All recorded MC estimates lie above the corresponding lower bound, though an MC estimate is not theoretically forced to do so. The global joint floor does not shrink with dimension. The data show decreasing estimation error; three n values do not establish an observed plateau or consistency toward the floor.

## Numerical and computational checks

- max_gaussian_roundtrip: `1.05427e-12`.
- max_observation_roundtrip: `5.55112e-16`.
- max_logdet_cancellation: `4.17799e-12`.
- max_density_identity_error: `7.10543e-15`.
- All 36 stochastic latent-copy samples and log determinants are bitwise identical; maximum sample discrepancy is zero. This explicitly prevents a strict quality/cost superiority claim over the same stochastic decoder.
- D=8: median fit time across cells 0.000857 s; median 256-vector sample time 0.001244 s; fit range 0.000734–0.101583 s.
- D=32: median fit time across cells 0.002659 s; median 256-vector sample time 0.004939 s; fit range 0.002368–0.003943 s.

Timing contains warmup/outlier effects (including a roughly 0.1-second first fit); there is no timed competing algorithm. It supports execution within the cap, not a comparative efficiency result.

GPU job 45552255 (`diff-complete-0908`) remains PENDING for Priority at audit time. It has no experimental result yet. No remote worktree changes or job submissions occurred.

Audit conclusion: the completed bounded synthetic observation-only study is internally consistent and source-bound. It supplies no real-image/video generation result, no latent-diffusion superiority, and no empirical asymptotic-rate proof.
