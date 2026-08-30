# RMPF-R21: two-harmonic conditioner boundary

## Scientific decision

`ABSOLUTE_KNOWN_TRUTH_PASS | INCREMENTAL_PARENT_ENERGY_FAIL | CIFAR_SOURCE_INCREMENTAL_FAIL | UCF_SUPPORT_FAIL | DEVELOPMENT/REPLICATION/CONFIRMATION SEALED`

R21 changed exactly one layer of the preserved R19/R20 exact normalized no-VAE flow: the source-fit first-harmonic phase conditioner was extended to the smallest direct two-harmonic conditioner. The periodic six-bin rational-quadratic map, two frozen states, rank-eight angular organization, reversible multiresolution scaffold, standard-Gaussian prior, likelihood, datasets, seeds, budgets, identity/R16/additive/nonlinear/copy controls, and every real-development and confirmation seal were unchanged.

The result is negative for the requested incremental claim. R21 has a substantial absolute known-truth advantage over identity, and it improves exact NLL and the registered dependence diagnostic over the parameter/compute-matched padded first-harmonic parent. It does not establish incremental proper-energy headroom over that parent. The requested CIFAR/UCF source gate was subsequently run only as a nonpromotable source diagnostic; it independently reproduced the same mismatch and did not authorize development.

## Frozen mechanism

For state `s`, conditioner angle `phi_c`, and target angle `phi_t`, define

\[
h_2(\phi_c)=
(\sin\phi_c,\cos\phi_c,\sin2\phi_c,\cos2\phi_c)^\top,
\qquad
m_s(\phi_c)=\theta_s^\top h_2(\phi_c).
\]

With the unchanged periodic rational-quadratic circle map `R_s`,

\[
\eta=\operatorname{wrap}\{\phi_t-m_s(\phi_c)\},
\qquad
\xi=R_s(\eta),
\qquad
\phi_t'=\operatorname{wrap}\{\xi+m_s(\phi_c)\}.
\]

The conditioner and target radius are unchanged. Hence the inverse is

\[
\eta=R_s^{-1}(\operatorname{wrap}\{\phi_t'-m_s(\phi_c)\}),
\qquad
\phi_t=\operatorname{wrap}\{\eta+m_s(\phi_c)\}.
\]

In `(conditioner angle, target angle)` order the Jacobian is triangular. The offset derivative appears only in the lower-left block, so the exact added log-Jacobian is

\[
\log |\det DT|=\sum_{j=1}^{2}\log R_s'(\eta_j).
\]

Zero second-harmonic coefficients recover the parameter/compute-matched padded R19 conditioner. Identity splines recover the ordinary parent flow. Every visible coordinate remains in the exact change-of-variables density; there is no VAE, ELBO, reconstruction loss, stochastic fiber, deleted coordinate, or extra visible dimension.

## Deterministic and known-truth execution

Tests: **3 passed**. Across seeds `10030–10034`:

| Check | Maximum or result |
|---|---:|
| Round-trip error | `3.2474e-15` |
| Forward/inverse logdet cancellation | `1.1124e-13` |
| Finite-difference logdet error | `5.9959e-10` |
| Compact reload base/logdet mismatch | `0 / 0` |
| Copied-mechanism mismatch | `0` |
| Active states | `2/2` in every seed |
| Learned degrees | `18` |
| Compact bytes | `314` |

Registered paired contrasts, reported as control minus R21:

| Control | Metric | Estimate | Paired 95% interval | Gate |
|---|---|---:|---:|---|
| Identity | Proper energy | **0.0097010** | **[0.0068722, 0.0125299]** | absolute `>=0.005` LCB: pass |
| Identity | Exact NLL/dim | **0.0653377** | **[0.0561659, 0.0745095]** | pass |
| Identity | Dependence error | **0.931982** | **[0.884506, 0.979459]** | pass |
| Padded R19 | Proper energy | **0.00027568** | **[-0.00028881, 0.00084017]** | incremental `>=0.002` LCB: **fail** |
| Padded R19 | Exact NLL/dim | **0.0539413** | **[0.0451234, 0.0627591]** | pass |
| Padded R19 | Dependence error | **0.709091** | **[0.671247, 0.746935]** | pass |
| R16 | Proper energy | `0.0006854` | `[-0.0002930, 0.0016638]` | no established superiority |
| Additive | Proper energy | `0.001489` | `[0.000255, 0.002723]` | below incremental practical margin |
| Matched nonlinear additive | Proper energy | `0.001446` | `[0.000271, 0.002622]` | below incremental practical margin |

## Source-only diagnostic

The known-truth incremental attribution gate had already failed, so this diagnostic was frozen as nonpromotable and could not authorize development.

CIFAR state 0 retained identity-relative source headroom: **0.012206 [0.005765, 0.018834]**. But padded-R19-minus-R21 proper energy was only `0.000524 [-0.000119, 0.001196]`; the incremental held-out logdet gain minus BIC was `-0.006181`; and the second-harmonic score gain was `-0.007523`. State 1 failed all practical source criteria. UCF states contained only **39** and **27** source-separated examples, below the frozen support minimum.

No realistic development, replication seed, or untouched confirmation outcome was opened.

## Compute and state

| Job | Wall time | Peak RSS |
|---|---:|---:|
| Deterministic tests | 3.41 s | 189,036 KiB |
| Five-seed known truth | 16.83 s | 215,216 KiB |
| CIFAR source diagnostic | 4.88 s | 264,084 KiB |
| UCF source diagnostic | 2.40 s | 249,236 KiB |

Local verified state:

- branch `pro/rmpf-r21-two-harmonic-conditioner`
- head `9fa16d0fefa9cc1024f394b256c8cac144a60e6c`
- tree `db3d6305bad8f4b2c4e88216e34d47f4c1165835`
- parent R20 `052d419b2306d4052d136afa27c96a8570a046e3`
- tag `rmpf-r21-two-harmonic-final-verified-20260830`
- working tree clean
- independent verifier PASS

The first failed layer is incremental proper-score headroom and conditioner target alignment—not invertibility, normalization, optimization, state identification, or compute. Higher harmonic order, knot count, state, rank, and threshold changes are blocked by this result.