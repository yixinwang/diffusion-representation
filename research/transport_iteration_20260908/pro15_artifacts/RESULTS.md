# Pro15 actual standalone results

Original run only. Two development seeds, dimension 20, four roots and two blocks of eight residual coordinates: eight unknown true pairs in each nonnull law. All 72 fits were frozen before scoring. No full-size learned graph, PSC or native-image experiment was run.

## Large-budget results

65,536 graph arrays plus 4,096 parameter arrays. Values are whole-joint KL in nats, not per-coordinate KL.

| Seed | Law | Spectral correct/false | Histogram LR correct/false | Spectral KL | Histogram LR KL | Product KL |
| --- | --- | --- | --- | ---: | ---: | ---: |
| 150901 | harmonic | 8/0 | 8/0 | 0.000647247942797 | 0.00284089239792 | 0.0114943346946 |
| 150901 | null | 0/0 | 0/0 | 0.000229862959088 | 0.000229862959088 | 0.000229862959088 |
| 150901 | off_basis | 0/0 | 8/0 | 0.0178614991823 | 0.0095254672013 | 0.0178614991823 |
| 150902 | harmonic | 8/0 | 8/0 | 0.00114767575985 | 0.00248592318501 | 0.0114570964827 |
| 150902 | null | 0/0 | 0/0 | 0.000192624747202 | 0.000192624747202 | 0.000192624747202 |
| 150902 | off_basis | 0/0 | 8/0 | 0.0178242609706 | 0.00964412870249 | 0.0178242609706 |

Both conditional graph methods recover all eight pairs on the harmonic law. Their different harmonic-versus-histogram parameter families explain the KL difference; this is not a graph-discovery superiority result. On the off-basis law, spectral and old unconditional discovery fail, while the conventional histogram likelihood method recovers all eight pairs. Under the null every eligible graph method selects no edges. An arbitrary oracle structural matching under the null is not a recovered active graph.

## Original 4,000-array budget

2,000 graph plus 2,000 parameter arrays. Spectral, histogram LR and old unconditional all select zero edges in every nonnull seed/law cell. Their KL values equal product exactly in these cells.

| Seed | Law | Eligible discovery-method / product KL | Oracle graph + harmonic KL | Oracle graph + histogram KL |
| --- | --- | ---: | ---: | ---: |
| 150901 | harmonic | 0.0135758178725 | 0.00303948921767 | 0.00847495337656 |
| 150901 | off_basis | 0.0199429823602 | 0.0207015436419 | 0.0148512002078 |
| 150902 | harmonic | 0.0145363723904 | 0.00387290205801 | 0.00575538794694 |
| 150902 | off_basis | 0.0209035368783 | 0.0215355351994 | 0.0128737330474 |

Oracle graph results are privileged diagnostics, never eligible comparators. The harmonic oracle is worse than product on the off-basis law in both budgets: graph knowledge cannot cure context-head misspecification. Null oracle fits also add unnecessary noisy dependence. All these failures remain in the original scores.

## Numeric and provenance checks

- 36/36 original tests passed; no unexpected execution failure or replacement run.
- 72/72 actual fitted models sealed and scored; 12/12 exact-copy checks are bitwise equal.
- Independent direct-density quadrature versus the series evaluator: maximum 2.0050627824730327e-13.
- Context quadrature refinement maximum: 2.0056525884548648e-13.
- Independent edgewise resummation versus blocked weighted Grams: maximum 1.7347234759768071e-16.
- Saved-state source-to-output and forward-logdet replay: exactly zero discrepancy for all 72 models.
- Small-model Gaussian roundtrip maximum: 3.0642155479654321e-14.
- Small-model logdet cancellation maximum: 1.0658141036401503e-13.

These are ordinary floating-point checks, not interval enclosures or an independent retraining replication.

## Full-dimensional numerical smoke check

A constructed, unfitted D=3,072 model used 64 saved full-dimensional Gaussian sources. Roundtrip error was 1.198774413069259e-11; logdet cancellation error was 5.8207660913467407e-11. The exact copy tied bitwise. Complete source-to-output time was 0.016015276 seconds, including Gaussian CDF, root, context, residual transforms and logdet. This is not learned graph validation, native image quality or a speed comparison to FM.

## Costs and inference limits

The complete original local runner took 4.45740892 seconds through its status receipt and recorded process peak RSS 186932 KiB. This includes all small fits, fixture generation/saving, scores and numerical checks, but excludes subsequent read-only auditing, documentation and archive creation. It is not a full-size timing projection.

All fit-stage, fit-plus-serialization, complete decode and complete log-density times are retained per model. A separate audit reloaded models and existing arrays without fitting or fresh randomness. Both conditional graph methods share the same observations and public marginal assumptions. Histogram LR uses its graph budget for proposal/testing, rather than receiving additional observations. Root fits are recomputed and charged separately for each arm.

The formal Lane A-scale bound in bounds.json uses 69,507 graph and 4,000 parameter arrays, a different budget and dimension from these empirical small cells. Its combined high-probability joint KL upper bound is 0.8576468021727498 at probability at least 0.9699832735144035 under the exact theorem assumptions. The input bank would occupy 1,806,508,032 float64 bytes; the two graph Grams require about 576.5 billion arithmetic operations. No such full-size fit or optimized runtime comparison was performed.

The weighted conditional-covariance mechanism is established methodology, not a claimed new general-purpose CI test. No native, universal latent, or trained-FM superiority was established. The exact-copy tie and off-basis failure are decisive scope limits.

## Old-control implementation qualification

The old control preserves the original unconditional psi-Gram statistic and
0.175 threshold, while using the new common isolated-edge retention interface
rather than requiring a perfect matching for the entire block. It is not a
byte-identical reproduction of the historical learner. Across every original
registered cell the largest absolute off-diagonal old score was
0.05771266793551603, below 0.175. Thus both retention rules produce the identical
empty graph and fitted-product law on all reported cells. The difference cannot
explain any reported comparison; it remains explicit for reuse on other data.
