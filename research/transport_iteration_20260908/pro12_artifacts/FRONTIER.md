# Certified restricted frontier

All KL quantities are nats per complete D3072 array. Endpoints below are the
actual outward decimals used for classification, not ordinary quadrature estimates.

## Conditional and expected-training-risk enclosures

| N stages | Conditional lower | Conditional upper | Expected-risk lower | Expected-risk upper |
|---:|---:|---:|---:|---:|
| 4 | 0.22372470287378568 | 0.22372470323412066 | 0.22372470285456023 | 0.22372471382618164 |
| 8 | 0.032062905626935562 | 0.032062906091946647 | 0.032062905624180273 | 0.032062915221547965 |
| 16 | 0.0021465015491993599 | 0.0021465020022553211 | 0.0021465015490149033 | 0.0021465104787589165 |
| 32 | 0.00013774772474983807 | 0.00013774817896396868 | 0.00013774772473800089 | 0.00013775648599630906 |
| 64 | 0.0000087093770403469173 | 0.0000087098339525953205 | 0.0000087093770395984875 | 0.0000087180979953948377 |

Exact expected-training-risk upper: **8.2496059247963795E-9**. Its correct-recovery KL is zero.

The primary conditional widths are at most 4.66e-10. The final expected-risk
widths are at most 1.098e-8. Both are below 2e-8. The unconditional lower
bound is (1-p_gamma-p_j) times the certified conditional lower, not the
conditional lower copied unchanged. The final learning correction is derived
in MATH.md, section 8, and evaluated in `results/n*_risk.json`.

## Fastest quality-eligible primary endpoint-Heun comparator

Ratios are **Heun time / central-exact time** from paired archived medians.
Ranges summarize the three fixed Gaussian source seeds within each batch size.
They are descriptive, not sampling confidence intervals or certified latency bounds.

| Target | Fastest eligible N | Nontrivial kernels | Ratio, batch 1 | Ratio, batch 64 |
|---:|---:|---:|---:|---:|
| 0.1 | 8 | 6 | 1.704024–1.712099 | 2.003946–2.049441 |
| 0.01 | 16 | 14 | 3.055893–3.073486 | 3.980559–4.144089 |
| 0.001 | 32 | 30 | 5.742757–5.787799 | 8.086462–8.346958 |
| 0.0001 | 64 | 62 | 11.028875–11.160989 | 16.248679–16.766971 |
| 0.00001 | 64 | 62 | 11.028875–11.160989 | 16.248679–16.766971 |
| 0.000001 | None; all five excluded | — | Not defined | Not defined |

Conditional and expected-training-risk classifications agree at all requested
quality targets. The exact central sampler qualifies throughout. No fixed-grid
Heun comparator qualifies at 1e-6; this is not infinite speedup and says nothing
about untested larger grids, different solvers or copied exact decoders.

**Retained adverse cost result:** endpoint-Heun4 is 1.7052855094513042% to
7.849811272818541% faster than exact central across the six primary cases.
It is ineligible at every requested target because even its certified KL lower
bound exceeds 0.1. At sufficiently loose targets its measured speed advantage
remains. The supplied traced-memory negative is not reversed.

## Certificate cost and checks

| N | Accepted rectangles | Splits | Function nodes | Local integration seconds |
|---:|---:|---:|---:|---:|
| 4 | 274 | 82 | 9864 | 3.380450 |
| 8 | 299 | 107 | 10764 | 8.609630 |
| 16 | 306 | 114 | 11016 | 18.489247 |
| 32 | 307 | 115 | 11052 | 37.387286 |
| 64 | 307 | 115 | 11052 | 76.051124 |

Total primary integration wall time: 143.917737513 seconds. These are local
certificate costs, not PSC sampler latencies. All 256 consistency checks and
the separate global-constant checks passed. An order-8, 192-bit, changed-mesh
N4 certificate completed in 3.468545063 seconds with interval
[0.22372470289488791, 0.22372470321301843]. It corroborates but does not validate
by resolution agreement; its own derivative remainder validates it.

## Implementation/quality boundary

This is an **ideal-real-law certified quality / observed binary64 cost frontier**.
MPFR quadrature roundoff is included. Binary64 generator-law KL, rounded ML
and floating QR are not certified. See PROTOCOL.md for the discrete-output issue.
An exact-copy decoder ties. There is no general VAE, native-image, video, GPU
latency, memory or universal sampler-superiority conclusion.
