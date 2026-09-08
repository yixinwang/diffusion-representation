# Convex positive-density spline development validation

Status: prospective CPU validation of `PositiveSplineFlow`. This estimator has convex coefficient fitting and C1 conditional CDFs. It is a separate algorithm from the neural rational-quadratic-spline pilot. This validation supplies no image/video, efficiency, or population-confirmation claim.

## Frozen cells and independent units

Run all 36 combinations of:

- world: `local`, `distant`;
- observed dimension: 8, 32;
- fitting arrays: 256, 1,024, 4,096;
- seed: 8101, 8102, 8103.

Every row is one independently simulated complete array in the open unit cube. Parameter sharing pools coordinate responses within each row. Reports preserve the independent-array count and the response count for each shared head. Pooled responses supply no extra independent sample count.

Use NumPy `SeedSequence([seed, dimension, fitting_count, world_id, purpose])`, with world IDs zero and one and purpose IDs one for fitting, two for development evaluation, and three for the Gaussian source. The streams are separate. Development arrays are generated only after the corresponding fit completes. All cells, graphs, and evaluation rules are specified before any development result. No cell selects another cell's configuration.

## Simulator and public graphs

Set rho to 0.65. Each correlated pair has density

`p(x,y) = 1 + rho*cos(2*pi*x)*cos(2*pi*y)`.

In the local world, independent adjacent pairs cover the vector. Even coordinates are roots; each odd coordinate has its preceding coordinate as its sole parent. All roots share group zero, and all children share group one.

In the distant world, only coordinates zero and D-1 have this dependence. The remaining coordinates are independent uniforms. The fitted graph is a one-step chain: coordinate zero is a root, and coordinate j has parent j-1. The root has group zero and every other coordinate shares group one. The true selected local conditionals in this world are uniform; the graph omits the distant dependency.

The simulator starts from independent uniforms and uses 54 vectorized bisection steps to invert the true conditional CDF. It returns observed arrays alone. The fitter receives no true densities, simulator coefficients, uniforms, or inverse maps. The evaluator can use the simulator's exact log density after fitting. A simulator floating-point endpoint causes an explicit failure; it is never clipped.

## Estimator and optimization

Use four intervals, quadratic open-uniform context B-splines, and linear response-density hats. Coefficient bounds are 0.175 and 3.3. Every response row integrates to one. Start at uniform density; use the implementation's bounded-knapsack linear minimizer and convex scalar line search. Each group has a cap of 500 Frank–Wolfe updates and a requested empirical optimization gap of `1e-4` nats per pooled response.

Report each group's objective trace, gap trace, step sizes, response count, update count, and convergence flag. Also report the whole-array per-coordinate gap, weighted by the number of coordinates in each group. A capped fit whose gap exceeds its requested tolerance is an optimization failure. It cannot be reported as a converged estimator. Its density and development measurements remain available for diagnosis.

The gap concerns the empirical convex objective and ordinary floating-point arithmetic. It supplies no population or interval-arithmetic certificate. With four intervals fixed across fitting sizes, this study measures a finite estimator configuration; it does not verify an asymptotically optimal bin schedule or eliminate approximation error.

## Development measurements

For every fitted cell, draw 4,096 new complete evaluation arrays. Compute per-array `log p_true - log q_fit`, then report joint KL and per-coordinate KL estimates with standard errors of the sample mean. Whole arrays are the resampling units. Monte Carlo estimates can be negative and are retained without clipping. These standard errors condition on the fitted model and omit retraining variability.

Generate 256 complete arrays from independent D-dimensional standard Gaussian sources. Retain every source coordinate. Report the energy score using Euclidean distances divided by `sqrt(D)`: average generated/evaluation cross distance minus half the unbiased generated self-distance U-statistic. Self-pairs are excluded. Exact-copy comparisons reuse all Gaussian and evaluation draws. This is an exploratory point estimate; it supports no significance claim.

For evaluation and generated arrays, report marginal cosine means, every adjacent-pair cosine-product mean, and the first/last cosine-product mean, with whole-array Monte Carlo standard errors. The distant graph's failure is assessed through the joint law and these dependence statistics. No latent statistic replaces the joint KL or energy calculation.

On all 256 Gaussian-source arrays, check source round trips, forward/inverse log-determinant cancellation, and the complete Gaussian change-of-variables density identity. The fixed absolute tolerance is `1e-8`; report each maximum and the resulting numerical-validity flag. Boundary saturation raises an explicit error. Construct a second wrapper sharing the identical fitted heads and graph; require bitwise identical samples, log determinants, and evaluation log densities. This stochastic-decoder copy inherits the full fit and sampling computation. It proves equality of this copied implementation only.

Record fitting, evaluation-density, and generation times separately. These are single CPU measurements and support no efficiency superiority claim. Serialize configuration, fitted coefficients, scalar diagnostics, simulation/source array hashes, source hashes, and environment details as JSON. No real-data loader or dataset path is used.

## Source binding, runtime, and preservation

Before creating outputs, require the submitted commit to equal HEAD and verify that all seven registered files match their committed blobs: the two estimator modules, their two test files, this runner, `PROTOCOL.md`, and the Slurm script. Record SHA-256 values for the same files. Refuse any existing output directory.

Use one computational thread, one allocated CPU, 2,000 MiB, and no GPU, under PSC account `cis260243p`. The executable deadline is 600 seconds, including source checks. A process timer interrupts an unfinished cell when that deadline is reached. Completed cells are written atomically as they finish, followed by a progress record. On timeout or failure, preserve completed cells, identify the pending cells, write a partial summary, and bind all completed JSON payloads through `SHA256SUMS` and `COMPLETE.json`. Final preservation can take a short additional interval after the timer is disabled; the Slurm allocation is twelve minutes for that writeout. A timeout is a partial run, never completion of the 36-cell registration.

After a numerical-validity failure, preserve that cell JSON and stop before evaluating another cell. Optimization-gap failures remain descriptive and do not abort subsequent registered cells. There is no development-result threshold for promotion. Numerical validity and optimization convergence are separate flags. Even a complete run in which both flags pass establishes no real-image/video quality, superiority to optimized latent models, or confirmation coverage.

## Optional API smoke

`--smoke` runs one local cell with dimension four, 32 fitting arrays, 64 evaluation arrays, 16 Gaussian-source arrays, and two optimizer updates. The output is labeled `api_smoke_only`. It exercises the function calls and serialization only and is excluded from the 36 registered development cells. It retains the same source binding and numerical checks.
