# QALT strict-oracle confirmation — 2026-08-23

This is the first confirmatory result generated after the protocol was frozen and pushed at Git commit `983a7a758b9763afb294b8a143e1a0b956c7d813`. The hashes in `config.json` exactly match the five source files at that commit. The run used Slurm job `44234187`, 200 independent Monte Carlo units at each registered sample size, and the untouched confirmation seed manifest whose SHA-256 is `153516355158d52e18f4551a8e89fa4105b456921d65c3a5c1f1b4e7ef08994c`.

## Registered outcomes

All five gates passed:

- exact exponential and structural-split full-latent controls tied QALT exactly;
- the registered finite-step Euler comparator had strictly positive KL and squared Wasserstein error at 4, 10, 20, and 50 steps;
- the same-information pooled full-latent estimator tied QALT exactly;
- under exact variance sharing, pooled estimation beat coordinatewise estimation at every registered sample size;
- under alternating variances, misspecified pooling lost at the largest registered sample size.

The finite-Euler KL gaps were `0.0361648`, `0.00642798`, `0.00166719`, and `0.000272823` at 4, 10, 20, and 50 steps. Under exact sharing at `n=512`, coordinatewise-minus-pooled KL was `0.0106015` with paired 95% interval `[0.00999721, 0.0112269]`. Under alternating variances at `n=512`, the same contrast was `-1.32849` with interval `[-1.32907, -1.32790]`, correctly revealing pooling bias.

The declared operation proxy was approximately `0.213` times the full-latent proxy. This is an algebraic architecture proxy, not measured latency, memory, or hardware efficiency. The nonlinear triangular decoder inverted to maximum absolute error `8.88e-16`, and the active sample had excess kurtosis `27.81`; these are mechanism sanity checks, not learned nonlinear quotient evidence.

## Maximum supported claim

The restricted oracle mechanisms are reproduced. Optimized same-information controls tie QALT in quality. QALT can retain a compute advantage only when measured chart and fiber overhead is smaller than the repeated full-token computation it removes.

This result does **not** establish universal quality dominance, learned quotient discovery, a better-information representation, wall-clock speedup, or image/video performance.

## Artifacts

- `config.json`: frozen grid, seed-manifest hash, exact source hashes, Git/environment/Slurm provenance.
- `solver_metrics.csv`: all analytic solver/control rows.
- `estimation_metrics.csv`: all 8,400 unit-level estimator rows.
- `summary.json`: intervals, gates, nonlinear sanity checks, runtime, and maximum claim.
- `run.log`: complete standard output from the allocated-node run.
- `SHA256SUMS`: integrity hashes for every machine-generated result artifact.
