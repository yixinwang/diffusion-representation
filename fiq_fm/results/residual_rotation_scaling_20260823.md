# Residual-rotation sample-scaling decision

The independently registered development ladder used new seeds `200..204` and changed only the number of training observations. The frozen cross-fitted commutant estimator did not meet its eligibility rule at any size.

| Train n | All charts pass | Haar projector mean | 95% interval | Eligible |
|---:|:---:|---:|---:|:---:|
| 12,000 | no | 0.0722 | [0.0097, 0.1346] | no |
| 24,000 | yes | 0.0911 | [0.0439, 0.1383] | no |
| 48,000 | yes | 0.0521 | [0.0126, 0.0917] | no |

Eligibility required every chart to pass and the projector interval upper endpoint to be below `0.08`. Although 48,000 observations improved the mean, its upper endpoint was `0.0917`. No new confirmation was run and seeds `300..329` remain unopened.

Quality controls continued to behave correctly at 48,000 observations: JBD minus oracle NLL was `0.0000401` nat per residual dimension with interval `[-0.00000245, 0.0000826]`; permutation minus JBD was `0.01587` `[0.00548, 0.02625]`; diagonal minus oracle was `0.02468` `[0.02279, 0.02658]`; full minus oracle was `0.000240` `[0.0000211, 0.000459]`.

The estimator is therefore retired from scaling. Raw configurations, logs, rows, unit diagnostics, and hashes are stored in the three `residual_rotation_scaling_n*_development` directories.
