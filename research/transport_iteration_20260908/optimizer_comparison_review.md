# Independent review of the 36-cell optimizer comparison

The accelerated solver did **not** improve the registered final-gap success rate or total measured fitting time. It returned slightly lower empirical negative log likelihood in 68 of 72 head comparisons, but the original Frank–Wolfe method met the 1e-4 gap target on 50 heads and 14 complete cells, compared with 49 heads and 13 cells for acceleration. This distinction matters: a lower objective does not establish a smaller final Frank–Wolfe gap.

The independently checked run used source `a397981d13f1bbe2121b58d5dac9f13089c243a7`, all 36 registered synthetic cells, and two shared heads per cell. Each method retained the original 500-update cap, four bins, coefficient bounds 0.175 and 3.3, and gap tolerance 1e-4. No observations beyond the registered synthetic fitting arrays were accessed during this review. No model was refitted, and no generation or population quality was evaluated.

## Results

| Quantity | Original Frank–Wolfe | Feasible acceleration |
| --- | ---: | ---: |
| Heads meeting gap target | 50 / 72 | 49 / 72 |
| Cells with both heads meeting target | 14 / 36 | 13 / 36 |
| Unconditional heads meeting target | 36 / 36 | 36 / 36 |
| Conditional heads meeting target | 14 / 36 | 13 / 36 |
| Total updates | 15,441 | 21,194 |
| Total instrumented fitting seconds | 36.1403 | 39.1299 |
| Independent post-fit check seconds | 0.5555 | 0.5813 |
| Forward sparse feature products | 46,395 | 42,581 |
| Transpose feature products / gradients | 15,513 | 21,315 |
| Objective evaluations | 30,954 | 42,581 |
| Linear minimizations | 15,513 | 21,315 |
| Line-derivative evaluations | 126,262 | 0 |
| Projection calls | 0 | 21,194 |
| Projected rows | 0 | 100,954 |

Both methods constructed 72 fitting feature matrices. Acceleration additionally performed 72 curvature column-sum passes. Frank–Wolfe performed 15,441 Brent root searches containing 110,821 callback evaluations; the endpoint tests explain the larger total line-derivative count. Full per-head counters remain in the machine-readable review and original payload.

For the local world, each solver met 18 of 36 head targets, and measured fitting time was 25.1916 seconds for Frank–Wolfe versus 22.1856 for acceleration. Thus neither solver met the conditional-head target in any local cell. For the distant world, the respective head counts were 32 versus 31 and times were 10.9487 versus 16.9443 seconds. The distant context graph is misspecified by construction; empirical optimization success does not resolve that modeling error.

Acceleration failed the target where Frank–Wolfe passed for the distant dimension-32, sample-count-256 cells with seeds 8101 and 8103. It passed where Frank–Wolfe failed for distant dimension eight, sample count 4096, seed 8102. Its four higher-objective returns were unconditional heads, with objective differences between about 1.97e-9 and 1.96e-8. The 68 lower-objective returns do not compensate for the registered gap and work outcomes when assessing the proposed optimizer improvement.

Timing is the original instrumented single-run CPU measurement in fixed Frank–Wolfe-first order. It includes wrapper overhead and has cache/order uncertainty. The original runner's full external elapsed time was 140.2376 seconds, including simulation, saving, checks, and other orchestration. Neither the fitting sums nor that runner time should be confused with scheduler allocation duration. This study supports no general speed advantage.

## Independent checks and numerical limits

The portable checker verifies all 449 recorded payload-file hashes and the full source snapshot inventory of 14 files against both the recorded SHA-256 values and Git blobs at the registered source commit. Every cell's report agrees with the summary, and every head report agrees with its saved cell report. The payload occupies 47,544,403 bytes, including its hash list. It was checked in place; this review does not silently copy or omit its contents.

The saved observations and head-input hashes match exactly, and the checker reconstructs the original graph, group assignments, site order, and observed pooling. It independently regenerates the original purpose-1 simulator stream. Untransformed coordinates match exactly. Transformed coordinates differ in 34 cells by no more than 2.22e-16 between the saved NumPy 2.2.6 environment and the checking NumPy 2.5.3 environment. The review records both reconstructed and saved hashes; it does not claim cross-platform byte equality. The required transformed-coordinate tolerance was 1e-14. Only simulated streams were reconstructed; teacher information was never used for fitting.

Using the saved input arrays, the checker independently evaluates each scalar context B-spline and response hat to build a dense design matrix. It then recomputes each returned model's objective and gradient and solves the row-constrained linear minimization with a separate greedy allocation implementation. Across all 144 fitted heads, the maximum objective discrepancy is 2.78e-17, maximum gap discrepancy 3.03e-15, and maximum row-integral error 1.34e-15. All box constraints pass. These are ordinary floating-point consistency checks, without interval validation.

Work counts are checked against exact identities for the original loop and against the accelerated diagnostics, including projection-row counts. This establishes consistency of recorded work, without replaying or refitting the optimization trajectory. It cannot independently reconstruct original wall-clock timing from saved coefficients.

The [portable checker](recompute_optimizer_comparison.py) accepts `--payload`, `--repo`, and a new `--output` path. The [recomputed results](optimizer_comparison_recomputed.json) contain every head's objective, gap, feasibility, timing, work counts, stream reconstruction record, and aggregate outcomes. This result is a negative development finding for the proposed blanket optimizer improvement; it supplies no generation-quality or population claim.
