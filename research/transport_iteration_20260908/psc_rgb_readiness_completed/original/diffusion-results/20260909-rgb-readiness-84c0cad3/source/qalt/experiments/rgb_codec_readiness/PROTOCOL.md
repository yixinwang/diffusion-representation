# Data-free RGB codec and flow-matching readiness

Status: prospective implementation/readiness screen. No real-image fitting or quality comparison is authorized by passing this screen alone. The active seven-arm innovation-response experiment remains frozen at 3b8c0f7 and is unaffected.

## Why this component is needed

The existing learned-analysis FM retains a full-dimensional stochastic residual decoder. A conventional deterministic compression codec with an unconditional latent Gaussian generator is a different control. Review19 supplied a concrete prospective codec and continuous-time field after correcting an erroneous source summary. Its proposed tied scalar-RQS flow is also different from the current innovation-conditioned candidate; it is not substituted here.

The new component is a convolutional deterministic 32x32 RGB codec with a 16x8x8 latent, paired with an unconditional continuous-time FM. A matching field runs directly on 3x32x32 pixels. Neither FM exposes an exact density. The codec uses no external pretraining, discriminator, perceptual network, VAE KL, or artificial decoder noise. Its prescribed reconstruction loss is L1 plus local SSIM and a small code-energy penalty. Architecture, normalization, objective, and exact numerical conventions must be documented with the implementation.

## Bounded screen

Use only fabricated tensors and newly initialized models. Do not import a canonical data loader, read FIT/development/test images, reuse a real-data checkpoint, or retain any synthetic trained weights for later real fits. Synthetic losses and generated arrays are numerical diagnostics, not quality evidence.

On one typed V100-32 GPU, within a one-hour allocation:

1. Record source hashes, environment, actual device, filesystem project identity, and parameter counts. Verify output writes and quota before acquiring compute.
2. Check full default codec, pixel field, and latent field on synthetic float32 inputs. Require intended gradients to be present and finite; check finite parameters after each step. Preserve failures without clipping, retrying, changing architecture, or reducing batch to rescue the screen.
3. Measure batch-32 initialization and training with five warmup and twenty timed steps, and one batch-125 readiness step. Synchronize around measurements and include the common finite-check cost. Keep the batch-125 result separate from batch-32 timings; do not extrapolate it as a measured 125-image throughput.
4. Measure fixed 32-step Heun sampling, with 64 field calls, at batch 1 and batch 32. Include the deterministic decoder for the latent pipeline. Record warmup separately, use the same hardware, retain each timing, and report allocated and reserved memory. The registered encoder remains resident during latent sampling and is disclosed in memory accounting; this is not a minimal deployment memory claim. Verify exact same-device checkpoint reload on one saved source as a separate numerical check outside warmed timing samples. Any sampling state is synthetic only.
5. Measure a synthetic 256-image codec encoding/cache write and checkpoint serialization. Report actual bytes and elapsed time. These are small readiness measurements, not measured full-4000-image cache costs or final study reservations.
6. Save all stage outcomes, source identity, synthetic seeds, counts, timing samples, failure traces, and a hash manifest. Release the allocation immediately on completion or failure.

This screen cannot qualify optimization on real images, justify the review's 3,000-second budget, demonstrate a trained baseline, or establish any candidate advantage. A future full profiling stage must measure all registered arms and complete sampling/checkpoint reservations before fixing a real-data budget.

## Conditions before a real-image protocol

- Use the identical underlying per-record observation arrays for every arm. If the exact-flow density uses dequantization and a logit chart, the pixel/codec pipelines receive an equivalent affine chart of those same dequantized observations. Do not dequantize only the density arms while silently changing the modeled distribution for other arms.
- Fit codec normalization on FIT only, freeze and serialize its means/scales, and charge fitting encodes/cache and all codec training to each standalone pipeline. Codec reconstruction and latent variance checks are engineering floors, not proof of a competent or state-of-the-art baseline.
- Resolve sequential codec/latent learning-curve accounting: early total-budget checkpoints cannot be described as completed latent generators before codec fitting/cache and some prior fitting have finished. Every plotted checkpoint must include its full prerequisite cost, with absent early generators marked unavailable rather than assigned a fabricated quality.
- Keep natural latent source dimension 1,024 explicit, with standard-normal base convention shared but no forced untrained residual noise. Ambient density and sample quality are distinct comparisons.
- Use newly initialized real-data fits, fully charged common setup, optimizer, cache, model transfer, checkpoint, and complete sampling/decoder costs. Do not use a partial failed fit as a final result.
- Keep all fits and required numerical checks ahead of development quality. Previously reused development data remain exploratory; fresh seeds do not create a fresh holdout.
- A conventional codec+FM control remains unresolved until appropriately trained and checked. The current candidate's mathematics is a conditional fixed-chart decomposition with exact-copy equality cases, not a universal advantage theorem.
