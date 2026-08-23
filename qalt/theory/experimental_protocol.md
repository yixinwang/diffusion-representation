# Registered strict-oracle protocol

## Question

Does QALT retain a strict generation-quality advantage after a full-latent baseline receives the same chart, fiber family, sharing information, source prior, training data, and active samples?

## Population and independent unit

The active variable is a four-dimensional, three-component Student-mixture. The twelve-dimensional fiber is Gaussian and the observation applies a nonlinear triangular bijection. One independently generated fiber-training dataset is the independent unit. Confirmation uses 200 fixed units per sample size.

## Fixed arms

1. QALT with the exact endpoint fiber.
2. Full-latent Euler integration of the same known linear fiber.
3. Full-latent exponential integration of that fiber.
4. Full-latent structural splitting with the same one-pass endpoint fiber.
5. QALT pooled variance estimation under exact sharing.
6. Full-latent coordinatewise variance estimation.
7. Full-latent pooled variance estimation using the same sharing information.
8. Pooled and separate estimation under misspecified alternating scales.

The active sample is reused within each unit so that every quality contrast isolates the fiber. Arms 1, 3, and 4 must tie exactly. Arms 5 and 7 must tie exactly.

## Metrics and inference

The primary metric is exact forward fiber KL. Squared Wasserstein distance is secondary for the known-scale solver study. For estimation, the unit-level KL is summarized by its mean and a paired percentile interval. Monte Carlo samples are used only to fit variances; the quality metrics are analytic.

## Gates

- Euler must have positive KL and Wasserstein error at every registered finite step.
- Exact full-latent and structural-split controls must equal QALT to numerical tolerance.
- Under exact sharing, pooled QALT must improve over coordinatewise estimation and tie the pooled full-latent control.
- Under alternating scales, pooling must eventually lose to separate estimation.
- No output supports a universal strict-quality claim over latent flow or diffusion.

## Leakage controls

The step, dimension, scale, sample-size, seed, metric, and tolerance grids are fixed in code. Confirmation seeds begin at 10000. No confirmation output selects an arm, group, metric, or threshold. Development smoke runs use a disjoint seed range and output directory.

Each configuration records the seed-manifest hash, Git commit and tracked-worktree state, exact hashes of the runner, core, tests, protocol, and theory note, software and platform versions, Slurm job ID, command line, and runtime.
