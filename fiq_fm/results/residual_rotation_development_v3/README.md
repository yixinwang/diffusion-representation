# Residual-rotation development v3: cross-fitted response operator

This is the final five-seed development snapshot before freezing confirmation. It is not confirmation and does not promote an image/video or universal-quality claim. It was run on allocated Slurm job `44280501` using only development seeds `0..4`, with 12,000 training, 4,000 validation, and 8,000 sealed test observations per seed.

The v3 cross-fitted response operator passed every frozen development diagnostic:

- all five Haar charts passed the response-gap, commutant-gap, and held-out-loss rule;
- Haar JBD projector error: mean `0.05467`, development interval `[0.03326, 0.07608]`;
- Haar JBD normalized true leakage: mean `0.000531`, interval `[0.000325, 0.000803]`;
- Haar JBD minus oracle-block NLL: mean `0.0000432` nat per residual dimension, interval `[-0.00000932, 0.0000895]`;
- Haar permutation minus JBD NLL: mean `0.01560`, interval `[0.00943, 0.02108]`;
- Haar oracle-diagonal minus oracle-block NLL: mean `0.02382`, interval `[0.02243, 0.02485]`;
- Haar provisional-full minus oracle-block NLL: mean `0.001087`, interval `[0.000762, 0.001372]`.

The smallest response relative eigengap was `0.2346` against the frozen `0.10` minimum; the smallest commutant relative eigengap was `0.6686` against `0.50`; the largest held-out off-block loss was `0.00777` against the `0.05` maximum. Confirmation seeds `100..129` were not accessed.

Exact source hashes and Slurm provenance are in `config.json`; per-unit diagnostics are in `seed_*.json`; `SHA256SUMS` covers the generated artifact.
