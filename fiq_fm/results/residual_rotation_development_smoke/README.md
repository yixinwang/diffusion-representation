# Residual-rotation development failure: coefficient-SVD v1

This is a diagnostic development snapshot, not confirmatory evidence and not a promoted result. It was produced on allocated Slurm job `44279806` with seed `0`, 2,000 training observations, 1,000 validation observations, and 1,000 sealed test observations.

The first learned joint block diagonalizer did not recover the hidden Haar-rotated residual partition:

- projector error: `0.82099`;
- normalized true cross-block leakage: `0.348995`;
- JBD minus oracle-block NLL: `0.0142035` nat per residual dimension;
- lowest two commutant-Gram eigenvalues: `2.8305` and `3.0494`.

The implementation extracted covariance contrasts by applying an SVD directly to the non-intercept rows of an overcomplete ridge-regression coefficient matrix. That span is sensitive to feature parameterization and finite-sample noise. This failed snapshot is retained to make the subsequent repair and any reversion auditable. Confirmation seeds `100..129` were not accessed.

The complete configuration and source hashes are in `config.json`; unit diagnostics are in `seed_0.json`; aggregate diagnostics are in `summary.json` and `metrics_long.csv`.
