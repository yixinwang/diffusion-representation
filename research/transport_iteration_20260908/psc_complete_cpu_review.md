# Independent complete-generator CPU pilot audit

PSC job 45558401; source `2b58285826a74e32882a6de0af2a1d951a85ba58`. One frozen local-image case: seed 3100, single-channel 8x8 synthetic arrays, 2,048 fitting arrays and 256 independent evaluation/source arrays. This is not CIFAR, video, a learned-analysis latent benchmark, or a statistical confirmation study.

## Preservation and source integrity

All 12 source SHA-256 values and recorded Git blob identifiers match the frozen commit. Final manifest and summary are identical; separate training and evaluation records match their summary copies. All 10 expected artifacts are present: five JSON records, three model state dictionaries, generated samples and copied-latent samples. Checkpoints load strictly with finite tensors and parameter counts matching their records. All five saved generated arrays have shape 256x1x8x8 and finite float32 values.

This older runner does not write a COMPLETE marker or a remote payload-hash manifest. Therefore no original payload-manifest verification can be claimed. The audit records SHA-256 and byte sizes of every downloaded artifact in the machine-check JSON, establishing a new local preservation ledger. It also checks cross-file consistency and reconstructs model outputs. The parent reports Slurm completion; this audit did not contact PSC.

## Independent metric and reconstruction check

The frozen evaluation world and seed 103100 were independently reimplemented, without regenerating fitting data. Gaussian source seed 203100 was recreated for checkpoint checks. Energy/MMD were recomputed from saved generated arrays using SciPy Euclidean distances and independent score formulas; mean/second-moment gaps were recomputed with NumPy. Maximum absolute discrepancy across all 30 scalar scores is 8.38190317e-09. No evaluator or generative model was fitted.

Local reconstruction uses PyTorch 2.14.0, versus original PSC 2.10.0+cu128. Saved spline and saved copied-stochastic-latent samples are bitwise identical. A reconstructed copied decoder also equals the reconstructed spline bitwise. Checkpoint-generated samples are not byte-identical across platforms/versions: maximum sample differences are 1.61e-5 for spline and 1.97e-6–3.16e-6 for the FM outputs. These small differences are reported, not relabeled exact cross-platform reproduction.

## Numerical gates

| Check | Saved PSC value | Local checkpoint reconstruction | Frozen limit |
|---|---:|---:|---:|
| Gaussian-source roundtrip maximum | 5.1259995e-06 | 1.2107193e-05 | .001 |
| Log-determinant cancellation maximum | 3.8146973e-05 | 0.00017547607 | .01 |

Both original and reconstructed gates pass. The local reconstruction encodes the original saved spline samples and compares with the recreated source, so it includes cross-platform sample mismatch. Exact-copied-sample equality is checked separately. Finite numerical validity does not establish approximation, learning convergence or representation quality.

## Cost and finite-budget comparison

| Fitted arm | Parameters | Updates | Training+initialization seconds |
|---|---:|---:|---:|
| spline | 55,390 | 1,164 | 90.011451 |
| full_fm | 5,689 | 15,717 | 90.003839 |
| hierarchical_fm | 18,463 | 12,777 | 90.002146 |

Every model respects the 100,000-update safety cap and the intended 90-second wall budget up to the registered final in-flight update (maximum overrun about .0115s). Initialization is charged. Equal wall budgets do not imply equal parameter counts or equal updates; the spline has about 9.7 times the full-FM parameters. Training objectives are different (NLL versus velocity MSE), so their numeric loss values cannot be compared as quality scores. No convergence claim follows from these capped fits.

Total runner time was 274.142s under a 12-minute one-CPU/2,000-MB Slurm request. Original CPU peak RSS is not present in these result records; null CUDA allocation fields are appropriate for CPU execution and are not measured zero memory. The run uses one case, not all four cases of the GPU pilot.

| Generator | Energy | Mean RMS gap | Second-moment gap | MMD .1 | MMD .3 | MMD1.0 | Generation seconds | Velocity calls |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| spline | 0.17893781 | 0.02477967 | 0.02400988 | 0.00073461 | 0.00412105 | 0.00010501 | 0.039631 | 0 |
| full_fm_heun_8 | 0.17928872 | 0.02128281 | 0.03329108 | 0.00301372 | 0.00558607 | 0.00008199 | 0.092973 | 16 |
| full_fm_heun_32 | 0.17936723 | 0.02112815 | 0.03335862 | 0.00332630 | 0.00599072 | 0.00008814 | 0.353692 | 64 |
| hierarchical_fm_heun_8 | 0.17889944 | 0.03195218 | 0.00647944 | 0.00016087 | 0.00195904 | 0.00045561 | 0.057193 | 48 |
| hierarchical_fm_heun_32 | 0.17891991 | 0.03189716 | 0.00699488 | 0.00021903 | 0.00206822 | 0.00045893 | 0.215205 | 192 |

The 8/32 labels are Heun step counts, not total network evaluations. Full FM makes 16/64 calls; hierarchical FM makes 48/192 calls across three resolutions. These counts were verified from source and records. Counts at different resolutions are not interchangeable FLOPs or latency.

The pattern is mixed: spline has the shortest single recorded generation batch, but hierarchical FM has lower energy, lower MMD at .1/.3, and lower second-moment error. Full FM has lower mean error and lower MMD at 1.0. There is no uniformly better generation-quality arm in this result. The energy differences are small and have no replicate-level uncertainty analysis. All predeclared bandwidths/step grids are retained; no winning score is selected after the fact. Single-batch timings supply no stable speed superiority claim.

## Scope and decision

All three pipelines learn from the same synthetic observations and use 64 Gaussian coordinates, with no observed coarse images during generation. Shared training-array provenance is recorded but fitting arrays were not regenerated by this audit. The hierarchical FM is a small fixed-Haar stochastic hierarchy, not the stronger learned-analysis/global-decoder latent baseline in Pro5. The exact-copy control still rules out a strict separation from a containing stochastic latent class.

Retain this result as successful end-to-end finite-budget synthetic execution with passed numerical gates and mixed exploratory metrics. It does not prove theoretical or practical superiority to latent flow/diffusion, real-image/video quality, or learned representation advantage. Original summary flags correctly leave all such advantages false.

Machine audit: `check_complete_cpu_results.py --output NEW_JSON --scratch NEW_DIRECTORY`; detailed source/payload hashes, recomputed metrics and checkpoint comparisons: `psc_complete_cpu_machinecheck.json`. No PSC actions, real-data reads, fitting, source edits or repository copying/commits occurred.
