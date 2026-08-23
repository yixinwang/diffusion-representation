# Quotient-Analytic Latent Transport

QALT is the latent-space specialization of the quotient--fiber program. It keeps the complete latent state but sends only an active quotient through repeated neural transport; a retained fiber is sampled once by an analytic or local conditional map.

The package begins with an adversarial oracle experiment. It does **not** claim universal dominance. A same-information full-latent baseline that uses the same exact fiber update must tie QALT in quality. The registered claim is:

> Under a common realized active sampler and an exactly solvable fiber, QALT is cheaper when the measured saved repeated-token cost exceeds chart and fiber overhead. It is strictly more accurate only than a declared baseline that integrates the solvable fiber inexactly or estimates correctly shared parameters separately.

Run the tests and registered pilot from the repository root:

```bash
PYTHONPATH=qalt/src pytest -q qalt/tests
PYTHONPATH=qalt/src python qalt/experiments/strict_oracle/run.py \
  --output qalt/results/strict_oracle
```

The test split is generated from fixed, untouched confirmation seeds. The output records the complete configuration, seed-level metrics, analytic targets, and promotion decisions.

The canonical audited theory note is `theory/QALT_repaired.tex`. It proves a conditional cost separation and a strict finite-step gap only against the registered inexact Euler comparator; it also proves why an optimized same-information control can tie QALT.
