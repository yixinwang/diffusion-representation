# Complete-generator pilot execution record

The complete spline model, full-tensor FM, hierarchical FM, tests, and prospective specification are committed at `adc1699357676c3d5015c5258ff3edd226a5c95c`. The full QALT suite passed 147 tests under Python 3.12. Two two-update integration checks completed locally. Their tiny synthetic evaluations were used only to verify execution; no model or scientific threshold was selected from their scores.

Independent source review verified the spline inverse ordering, Haar determinant, conditional Jacobian sum, and generative conditioning. It found that the first runner version retained earlier fitted models on the GPU. Before the committed pilot, each completed model was changed to move to CPU and release gradients. The minibatch RNG was separated from the flow-matching path RNG, and explicit float32 inversion tolerances were added before submission. The second smoke check passed these changes.

PSC job `45552255` runs source `adc1699` from the isolated worktree `/ocean/projects/mth250006p/ywang26/diffusion-complete-20260908`. Its result root is `/ocean/projects/mth250006p/ywang26/diffusion-results/20260908-complete`. The job uses one L40S GPU, four host CPUs, 16 GiB host memory, and a 30-minute allocation cap. Each independent model receives a 90-second training cap in each of four synthetic cases. The official image and video test data are inaccessible to this runner because it has no real-data loader.

The fourth fresh Pro conversation is https://chatgpt.com/c/6aa0633a-e1f8-83e9-812d-73bdd9749c78. It is still reasoning about a conditional-density construction and learning guarantee. The completed code also follows the first Pro review's multiscale proposal, whose correctness has been checked independently. A pending Pro response is not credited as a completed review.

The pilot results are pending. No current generation-quality, speed, memory, representation, real-image, real-video, or latent-diffusion superiority claim is established.

## Before-execution hardware amendment and replacement

On 8 September, Slurm estimated a 13 September start for L40S job45552255. Read-only job inspection confirmed it was still PENDING with zero run time. It was canceled before execution; `sacct` reports CANCELLED with zero elapsed time. No model from that submission ran.

The committed hardware amendment `dc0169f50eb0eddfbd5bff6ace6dc2fb384347a2` permits one available GPU type. Replacement job45554440 runs the identical four-case/model/seed/time/metric design in `/ocean/projects/mth250006p/ywang26/diffusion-next-20260908`, detached at that revision. All arms share its actual GPU, whose type is recorded at runtime. Results are separated at `/ocean/projects/mth250006p/ywang26/diffusion-results/20260908-complete-anygpu`. This revision adds no outcome-driven model or threshold change.

The same immutable checkout supplies CPU job45554441 for the 36-cell positive conditional-spline validation. It has one CPU,2000MiB host memory, a600-second runner limit and a12-minute allocation. Its output is `/ocean/projects/mth250006p/ywang26/diffusion-results/20260908-positive-spline`; Slurm logs are `positive-spline-45554441.out` in the result parent. The previous conditional-CDF study and this positive-spline study are distinct estimators. The new study explicitly records optimization-gap failures and stops after preserving any numerical failure. Neither submission supplies a result until execution and checking complete.

Replacement GPU job 45554440 failed on node v005 before training, after 2 minutes 22 seconds. PyTorch 2.10.0+cu128 reported CUDA error803 (unsupported display-driver/CUDA-driver combination); the initial availability assertion stopped execution before model tests or fitting. This supplies no model result. A separate five-minute hardware/runtime diagnostic was submitted as job45555977; an initial one-CPU/two-GiB diagnostic request was rejected before allocation, then corrected to the previously accepted four-CPU/16-GiB configuration. The original PyTorch installation remains unchanged.
