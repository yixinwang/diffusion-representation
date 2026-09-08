# Complete learned synthetic generator pilot

This prospective pilot follows the failed global covariance experiment. It tests complete learned sampling and numerical correctness. It cannot establish superiority on real images, real videos, or trained state-of-the-art latent systems. It uses no observed image dataset.

## Frozen model families and information

The spline generator learns a coarse flow and two conditional detail flows, retaining every Gaussian source coordinate. The chart is fixed orthonormal Haar. Each scalar map is a monotone rational-quadratic spline with identity tails, as in Neural Spline Flows. Alternating checkerboard couplings use only unchanged coordinates and already available coarse context. Each coupling sequence also has a learned channelwise location and bounded log scale. No hidden teacher coefficient, innovation, inverse chart, or label enters training.

The two independently trained rivals are a full-tensor flow-matching model and a hierarchical flow-matching model with its own learned coarse prior and stochastic conditional detail fields. All receive the same observed training arrays, source law, Haar transform availability, and minibatch-index sequence. Gaussian prior dimensions are equal. The hierarchy does not receive true coarse values during generation. Fixed Heun grids of 8 and 32 steps are both reported. Flow-matching training cost is measured directly and is never multiplied by generation step count.

An exact stochastic-latent copy wraps the complete fitted spline model. Its samples and densities must tie bitwise, and its charged training cost equals the spline model's cost. This control demonstrates class containment. It is not fitted independently. A win over the two finite FM configurations cannot become a claim against every stochastic latent model.

## Data and cases

Each case has 2,048 synthetic training tensors and 256 independent evaluation tensors. The teacher starts from Gaussian variables, applies nonlinear sinh transforms to coarse and detail variables, and gives details nonlinear coarse-dependent locations and positive scales. A second case adds a triangular dependence between separated detail locations. No teacher internals are passed into model constructors or losses.

Four runs use 8 by 8 tensors: one channel with local and separated-detail teachers, and four channels with the same two teachers. Four-channel tensors are synthetic multiple-frame arrays. They provide no real-video evidence. At this small resolution, network receptive fields may span the separated locations. The second teacher is a dependence stress case; it is not a proved representational failure case for this model. Testing such a failure requires separation beyond the generator's actual dependence range.

Seeds are 3100, 3101, 3110, and 3111 for these four cases respectively. Evaluation data are generated only after all three fits have frozen. No hyperparameter or stopping decision uses evaluation metrics. The runner retains each fitted model and generated output array. These four cases are development runs. They provide no replication of a real-data improvement.

## Resources and measurements

One PSC GPU allocation is capped at 30 minutes, four host CPUs, and 16 GiB host memory. Each model receives a 90-second training cap and a 100,000-update safety cap; training stops at the first limit. Initial construction is charged and subtracted from the training cap. A final in-flight update may exceed the cap; its elapsed time is recorded. Shared synthetic-data construction is recorded in total runner cost. Each completed model moves to CPU and releases its gradients before the next arm starts. Peak allocated memory includes the shared training array and only the active model. Parameter counts, training iterations, elapsed time, GPU allocated memory, and generation latency are reported.

All models use width 24 and batch size 64. Spline coarse depth is four couplings; each of two detail stacks has two couplings; every spline has eight bins. FM fields each have two hidden convolutional layers. These are specified pilot configurations, with no assertion of equal parameter counts or optimal baseline tuning. Future competitiveness claims require a baseline budget allocation and tuning comparison.

Metrics include normalized Euclidean energy score, raw-tensor Gaussian-kernel discrepancy at all three fixed bandwidths 0.1, 0.3, and 1.0, mean and second-moment discrepancies. Unbiased kernel estimates may be negative. No bandwidth is selected using evaluation results. The small evaluation set and single timing batch do not support a speed or quality significance claim. No scalar metric replaces image or video quality validation.

Before submission, require spline inverse and gradient tests, complete dense-Jacobian agreement on a small tensor, source-dimension conservation, exact copied-latent equality, FM objective normalization, and generation-context isolation. A two-update integration smoke test may precede the pilot; its metrics do not select the configuration.

## Decision

The trained float32 model must have maximum Gaussian-source round-trip error ≤0.001 and maximum absolute per-tensor log-determinant cancellation error ≤0.01. These are numerical tolerances. Statistical margins are separate. Reject numerical correctness if inversion, Jacobian, nonfinite-value, or copied-latent checks fail. Preserve every failed run. Successful completion means only that all three learned unconditional pipelines execute under a recorded cap. Interpreting their learning curves and output discrepancies requires a fresh Pro review. Real-image and real-video confirmation remain unopened.

Primary method sources: Durkan et al., Neural Spline Flows, https://arxiv.org/abs/1906.04032; Yu et al., Wavelet Flow, https://arxiv.org/abs/2010.13821; Lipman et al., Flow Matching for Generative Modeling, https://arxiv.org/abs/2210.02747. The construction uses established components and claims no architectural novelty.

## Pre-execution hardware amendment

On 8 September 2026, before any pilot arm ran, PSC estimated a 13 September start for the requested L40S job 45552255. The GPU request is broadened to one available GPU type. All three arms and their four cases remain in the same allocation, with unchanged model, data, seed, update, time, metric, and numerical settings. The actual GPU is recorded. This supports only the specified same-machine development comparison; no result is transferred to L40S performance. The original pending job is canceled and retained in the execution history.

## Separate CPU execution for numerical and learning validation

The first available-GPU replacement failed before fitting with CUDA error803. A diagnostic GPU job is pending. Before any complete-generator result is observed, register a separate CPU execution of the unchanged local one-channel case, seed3100,2,048 fitting arrays,256 evaluation arrays and90 seconds per arm. It uses one host core,2000MiB host memory and a12-minute allocation. All other model, source, objective, solver-grid, metric and numerical settings remain identical. `run_cpu.slurm` runs all three arms on that same CPU. This separate record checks a complete learned generator while accelerator compatibility is unresolved. Its speed and learned-model behavior cannot be reported as the registered GPU comparison or as an image/video result. No new teacher, fitting seed, margin, or post-result selection is introduced.
