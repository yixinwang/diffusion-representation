# RMPF-R16: source-only finite amplitude boundary

This append-only milestone records the diagnosis-driven R16 child. It preserves every prior QALT, FiberLift, FIQ-FM, MCQF, RMPF/RMGL/HSRF, R1–R15 result, failure, split, hash, tag, PR, and seal.

## One changed layer

Relative to local parent `70702985f4834a08cab60af4a190877e76c67c56`, R16 changes only the scalar amplitude of the already fitted R15 global-context Möbius coupling. The finite source-only candidate set is `{0, 0.5, 1, 1.5, 2, 3}`. The conditioner, coupling family, rank, states, optimizer, exact likelihood, data roles, controls, systems budgets, and endpoints are fixed.

## Executed result

- Five aligned seeds 10030–10034 all selected gamma 1 by held-out exact log score.
- Zero-minus-selected exact NLL/dimension: `0.0880203 [0.0859690, 0.0900716]`.
- Zero-minus-selected dependence error: `0.290271 [0.276786, 0.303757]`.
- Zero-minus-selected proper energy: `0.00210599 [0.00011571, 0.00409628]`; the lower endpoint again misses the frozen 0.005 margin.
- Gamma 0.5 had better proper energy than gamma 1 by `0.000689812 [0.000000992, 0.001378632]`, while gamma 1 had the best exact NLL and dependence error. This is the registered source-log-score versus downstream-proper-quality target mismatch; the effect remains below the fixed 0.001 amplitude-attribution margin and cannot justify retrospective retuning.
- Exact copied-mechanism mismatch: zero; maximum round-trip `1.11e-15`; maximum logdet cancellation `2.22e-15`.

Both real domains passed the inherited systems gates but selected exact gamma-zero fallback:

- CIFAR: statewise held-out gains were below BIC charges. Seed-9400 energy `20.27977435`, versus full flow `20.24468812` and positive-rate latent flow `20.18914595`; amplitude attribution was effectively zero.
- UCF: frozen-state counts were only 39 and 27, below 128. Seed-9400 energy `48.95388989`; positive-rate latent diffusion was `46.79402453`; amplitude attribution was effectively zero.

Replication seeds 9401–9404 and untouched confirmation remain unopened. Scalar amplitude, grid, state, rank, BIC, and endpoint retuning are retired. The next distinct layer would require source-shared hierarchical estimation of a genuinely multimodal global-context law with a finite proper-energy lower bound.

## Local scientific state

- Branch: `pro/rmpf-r16-source-amplitude`
- Preregistration commit: `d9b201cfa08744550fdec3e1153fedbf5844ec8b`
- Implementation commit: `4317aa4bc0ea738effdb78887d56dafb485612d1`
- Evidence head: `645e9d6e3030f7ce07dd4152c3651a4300febba7`
- Tree: `67820009604dcba6e8a3172a8b4d64faed6ac875`
- Tag: `rmpf-r16-source-amplitude-boundary-20260830`
- External verifier: PASS, 160 checks
- Confirmation opened: false
