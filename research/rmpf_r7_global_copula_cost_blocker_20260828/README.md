# RMPF-R7 reversible butterfly global-copula round

## Decision

`KNOWN_TRUTH_PASS | COST_ORACLE_FAIL | REALISTIC_R7_NOT_OPENED | CONFIRMATION_SEALED`

R7 changed exactly one scientific layer relative to R6: coordinatewise/projection endpoint splines were replaced by a coupled global-copula transform. The final balanced block coupling passed the hidden-copula known-truth gate that coordinatewise R6 could not pass. The frozen end-to-end cost oracle failed even after one exact systems-only realization child, so the protocol prohibited opening new R7 CIFAR/UCF outcomes.

## Parent evidence preserved

R5/RCC remains immutable: CIFAR exact-NLL gain `0.0465464 [0.0461535,0.0469394]`, adverse energy change `-0.0036067 [-0.0041096,-0.0031039]`, video rank zero, clipping/variance failure, and all training/inference/memory/storage losses. R6 remains immutable: known-truth marginal-spline success, realistic rank-zero selection, forced-positive NLL-only gain, support-child failures, and confirmation sealed.

## Scientific sequence

1. Sequential parity coupling: exact and diagnostically effective, but only `0.00585 [0.00550,0.00620]` nat/dimension over coordinatewise R6, below the frozen `0.01` margin.
2. Stronger-shift child: failed because greedy ordering selected a shifted coordinate as conditioner, leaving later states nearly degenerate.
3. Balanced block-parity coupling: changed only dependency factorization. It passed all five seeds.

## Final known-truth result

Across seeds 9500--9504:

- identity minus R7 NLL: `0.569499 [0.568519,0.570480]` nat/dimension;
- coordinatewise R6 minus R7 NLL: `0.092466 [0.092273,0.092659]`;
- R7 transformed hidden-parity gap: `0.01098 [0.00673,0.01523]`;
- coordinatewise R6 gap: `1.64090 [1.63681,1.64499]`;
- maximum round-trip error: `5.33e-15`;
- maximum finite-difference log-Jacobian error: `3.36e-9`;
- copied-mechanism mismatch: exactly zero.

## Cost oracle

The initial full-DCT implementation failed because it materialized complete detail arrays and the unchanged B+S trunk dominated latency. One systems-only child replaced the full DCT with exact selected DCT columns and collapsed the affine B+S forward recurrence.

Final endpoint-only results:

| Domain | Endpoint/full latency | Incremental FLOP/full | Learned bytes/full | Incremental workspace/full |
|---|---:|---:|---:|---:|
| CIFAR | 0.0616 | 0.0293 | 0.000181 | 0.0395 |
| UCF | 0.0576 | 0.0147 | 0.000015 | 0.1180 |

Final total results:

| Domain | Candidate/full latency |
|---|---:|
| CIFAR | 5.238 |
| UCF | 3.531 |

The exact collapsed forward matched the original B+S sampler within `7.11e-15` for images and `1.29e-14` for video. Therefore the blocker is not numerical mismatch.

## First failed layer and stop boundary

The coupled global-copula layer is scientifically validated on the registered hidden-copula regime and is individually cheap. The first failed layer is the preserved B+S trunk's training/inference realization. It already exceeds the full-flow latency frontier before R7 is added. A second endpoint change cannot produce end-to-end latent-like cost.

The preregistration required the cost oracle before any new R7 realistic outcome. Because it failed after the permitted systems child, no new R7 development sample-quality metrics were opened. All prior opened CIFAR/UCF metrics remain available as unchanged controls, and confirmation remains absent.

## Reproduce

```bash
python reproduce.py
```

The script is self-contained apart from NumPy and SciPy and reproduces the five-seed known-truth contrasts, exact inverse/Jacobian checks, and copied-mechanism tie.