# RMPF-R15: source-fit global coarse-context conditioner

This append-only milestone records a diagnosis-driven child of the exact normalized, all-coordinate, no-VAE RMPF flow. The preserved parent is RMPF-R14 at `76d518519a1a3f8fd6a088a651214b8afd8738e2`.

R15 changes one layer only: R14's four-dimensional fine-stage conditioner statistic is replaced by the first four coordinates of a reversible source-fit signed permutation of the existing rank-eight global/coarse projection. The target Möbius coupling, optimizer, likelihood, parameter rank, BIC fallback, datasets, roles, controls, and seals are unchanged.

## Executed result

- Aligned seeds: 10030--10034.
- Identity minus R15 NLL/dimension: `0.0880203 [0.0859690,0.0900716]`.
- Identity minus R15 dependence error: `0.290271 [0.276786,0.303757]`.
- Identity minus R15 proper energy: `0.00210599 [0.00011571,0.00409628]`, below the frozen `0.005` practical lower-bound gate.
- Maximum round-trip error: `1.11e-15`; maximum log-Jacobian cancellation error: `2.22e-15`; exact copy mismatch: zero.

The systems route passed every video gate but narrowly missed the inherited image full-flow batch ratio (`0.0316516 / 0.0419808 = 0.75395`, limit `0.75`).

A separately frozen, nonpromotable seed-9400 CIFAR/UCF smoke was then executed only to localize transfer. Both domains selected exact identity fallback. CIFAR's held-out gains (`0.02530`, `0.03552`) were below BIC charges (`0.07085`, `0.07045`); UCF state counts (`39`, `27`) were below the fixed minimum `128`. Proper-energy and dependence attribution were therefore effectively zero.

Replication seeds 9401--9404 and untouched confirmation remain unopened.

Local complete scientific state:

- branch `pro/rmpf-r15-global-coarse-context-conditioner`
- head `70702985f4834a08cab60af4a190877e76c67c56`
- tag `rmpf-r15-global-context-boundary-20260830`

This compact branch does not modify or delete any prior QALT, FiberLift, FIQ-FM, MCQF, RMPF/RMGL/HSRF, R1--R14, data, split, result, failure, hash, tag, or PR artifact.