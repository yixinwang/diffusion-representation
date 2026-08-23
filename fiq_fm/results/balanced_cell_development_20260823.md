# Balanced-cell residual-rotation development result

This is development evidence on the preregistered seeds `500..504`. It is not confirmation evidence.

| depth | leaves | projector error, mean [95% CI] | worst held-out loss | charts accepted | decision |
|---:|---:|---:|---:|---:|:---|
| 2 | 4 | 0.428 [0.154, 0.702] | 0.0109 | 3/5 | reject |
| 3 | 8 | 0.261 [-0.029, 0.552] | 0.0268 | 3/5 | reject |
| 4 | 16 | 0.189 [-0.060, 0.439] | 0.0440 | 2/5 | reject |

The frozen projector eligibility condition required the upper endpoint to be below `0.07`; no depth was close. Every depth also had at least two Haar units below the frozen `0.50` commutant relative-eigengap threshold. Minimum training and validation cell-count gates passed, and held-out loss remained below `0.05`.

The quality controls behaved in the intended direction. Across depths, JBD-minus-oracle mean NLL was `0.000302` to `0.000394` nat per residual dimension, permutation-minus-JBD was `0.02187` to `0.02196`, diagonal-minus-oracle was `0.02436`, and full-minus-oracle was `0.00152`. These test-set diagnostics cannot rescue a chart estimator that failed its train/validation identification gates.

No depth is selected. Confirmation seeds `600..629` remain unopened, and the balanced-cell estimator is retired in its registered form.
