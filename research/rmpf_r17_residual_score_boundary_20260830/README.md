# RMPF-R17: source dependence-score residual direction

## Decision

`SOURCE_NLL_AND_DEPENDENCE_PASS | PRACTICAL_PROPER_ENERGY_FAIL | FINITE_ORACLE_GATE_FAIL | REAL_DEVELOPMENT_NOT_OPENED | REPLICATION_AND_CONFIRMATION_SEALED`

R17 preserves the exact normalized all-coordinate no-VAE R16 flow and changes only the two statewise 4x4 Möbius direction matrices. The amplitude candidates `{0,0.5,1,1.5,2,3}`, states, rank, coupling algebra, likelihood, source identities, controls, systems budgets, development split and confirmation seal are unchanged.

Within each state, let `u` be the unit target direction, `v` the unit global context, and `l` the unit copied-local feature. The identity Möbius score is `-6u`. R17 projects `u` off `[1,l]`, regresses the residual score on `v`, and applies the fixed identity-Fisher natural-gradient scale. This leaves one 4x4 matrix per state and exactly the same nonzero-amplitude inference graph as R16.

The inverse remains `M_a^{-1}=M_{-a}`. The added log-Jacobian is

```text
3 [ log(1-||a||^2) - log(1+||a||^2+2 a^T u) ].
```

Gamma zero exactly recovers the parent. The copied-mechanism control ties exactly.

## Executed five-seed result

Seeds: `10030--10034`. All selected gamma `1.5`.

| Contrast | Estimate | 95% interval | Decision |
|---|---:|---:|---|
| identity minus R17 NLL/dim | 0.074131 | [0.072450, 0.075811] | pass |
| identity minus R17 dependence error | 0.216486 | [0.208504, 0.224469] | pass |
| identity minus R17 proper energy | 0.001798 | [0.001247, 0.002349] | fail: LCB < 0.005 |
| R16 minus R17 proper energy | -0.000308 | [-0.001932, 0.001315] | fail |
| R16 minus R17 NLL/dim | -0.013890 | [-0.014596, -0.013183] | R17 worse |
| R16 minus R17 dependence error | -0.073785 | [-0.082789, -0.064781] | R17 worse |
| identity minus exact finite oracle energy | 0.002092 | [0.000033, 0.004151] | oracle upper < 0.005 |

Exact checks:

- round trip: `4.4409e-16`;
- forward/inverse logdet cancellation: `1.4988e-15`;
- source residual orthogonality: `2.1148e-16`;
- copied-mechanism mismatch: `0`;
- parent R16 scientific rows reproduced with maximum difference `0`.

A post-gate nonpromotable 2,000,000-term population diagnostic estimated oracle energy headroom `0.006363 [0.004766,0.007961]`; its lower endpoint still missed 0.005.

## Boundary

The failed premise was that removing copied-local score components would expose at least 0.005 proper-energy headroom and improve over R16 direct likelihood. The direction is nonzero and numerically correct, but the frozen teacher lacks certifiable finite proper-score headroom and direct likelihood remains better on NLL and dependence. Real CIFAR/UCF development was therefore not opened; seeds 9401--9404 and untouched confirmation remain sealed.

Local scientific state:

- parent: `645e9d6e3030f7ce07dd4152c3651a4300febba7`;
- branch: `pro/rmpf-r17-residual-score-final`;
- head: `e338a6608aa1ccc5adfa7ad0d4257b2967b3baf9`;
- tree: `42eda8f6d21b4f3bf4b2c9e40ea246ff073d0af5`;
- tag: `rmpf-r17-residual-score-final-20260830`;
- tests: 9 passed;
- independent verification: PASS, 17 checks.
