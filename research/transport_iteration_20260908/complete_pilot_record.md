# Complete-generator pilot execution record

The complete spline model, full-tensor FM, hierarchical FM, tests, and prospective specification are committed at `adc1699357676c3d5015c5258ff3edd226a5c95c`. The full QALT suite passed 147 tests under Python 3.12. Two two-update integration checks completed locally. Their tiny synthetic evaluations were used only to verify execution; no model or scientific threshold was selected from their scores.

Independent source review verified the spline inverse ordering, Haar determinant, conditional Jacobian sum, and generative conditioning. It found that the first runner version retained earlier fitted models on the GPU. Before the committed pilot, each completed model was changed to move to CPU and release gradients. The minibatch RNG was separated from the flow-matching path RNG, and explicit float32 inversion tolerances were added before submission. The second smoke check passed these changes.

PSC job `45552255` runs source `adc1699` from the isolated worktree `/ocean/projects/mth250006p/ywang26/diffusion-complete-20260908`. Its result root is `/ocean/projects/mth250006p/ywang26/diffusion-results/20260908-complete`. The job uses one L40S GPU, four host CPUs, 16 GiB host memory, and a 30-minute allocation cap. Each independent model receives a 90-second training cap in each of four synthetic cases. The official image and video test data are inaccessible to this runner because it has no real-data loader.

The fourth fresh Pro conversation is https://chatgpt.com/c/6aa0633a-e1f8-83e9-812d-73bdd9749c78. It is still reasoning about a conditional-density construction and learning guarantee. The completed code also follows the first Pro review's multiscale proposal, whose correctness has been checked independently. A pending Pro response is not credited as a completed review.

The pilot results are pending. No current generation-quality, speed, memory, representation, real-image, real-video, or latent-diffusion superiority claim is established.
