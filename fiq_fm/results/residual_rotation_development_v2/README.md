# Residual-rotation development v2: predicted-moment SVD

This is a five-seed development snapshot, not confirmation and not a promoted result. It was run on allocated Slurm job `44280101` using only allowed seeds `0..4`, with 12,000 training, 4,000 validation, and 8,000 sealed test observations per seed.

Replacing coefficient-row SVD with a feature-basis-invariant SVD of fitted conditional moments repaired the v1 failure:

- Haar JBD projector error: mean `0.06820`, development interval `[0.05797, 0.07656]`;
- Haar JBD normalized true leakage: mean `0.000582`, interval `[0.000360, 0.000803]`;
- Haar JBD minus oracle-block NLL: mean `0.00000334` nat per residual dimension, interval `[-0.0000394, 0.0000461]`;
- Haar permutation minus JBD NLL: mean `0.01564`, interval `[0.00950, 0.02110]`;
- Haar oracle-diagonal minus oracle-block NLL: mean `0.02382`, interval `[0.02243, 0.02485]`;
- Haar provisional-full minus oracle-block NLL: mean `0.001087`, interval `[0.000762, 0.001372]`.

The fourth and fifth fitted-moment singular values remained close in some units, so v2 is retained as a successful but rank-fragile development iteration. A separately logged two-split cross-fitted response operator is the next planned repair. Confirmation seeds `100..129` were not accessed.

Exact source hashes and Slurm provenance are in `config.json`; per-unit diagnostics are in `seed_*.json`; `SHA256SUMS` covers the generated artifact.
