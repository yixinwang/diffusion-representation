# Experiment trace

## Nonlinear triangular chart

- The linear covariance FIQ probe could not identify the nonlinear shear.
- Replaced covariance identification with paired-future conditional variance.
- Added the exact gauge `f(0)=0`; otherwise the shear is identifiable only up to a constant.
- Limited ambient GMM EM uniformly after seed-dependent convergence tails.
- Final run: 10 seeds, 40 paired samples per context, fixed train/validation/test generation.

## Dense rotation

- Directly exposed residual coordinates were judged too favorable.
- Added a random dense orthogonal observation mixing.
- A pure linear conditional-variance nullspace retained substantial state error.
- Added a Cayley orthogonal layer, nonlinear shear, and log-determinant state-spread gauge.
- Final five-seed recovery is evaluated after train-only affine alignment.

## VAE audit

- Used the same four-dimensional active-code budget as `(S,Q)` in DIQ.
- Swept beta over `{0, 1e-4, 1e-3, 1e-2}` using validation only.
- Added calibrated isotropic decoder noise rather than evaluating decoder means alone.
- DIQ improved state recovery and distribution metrics on all five paired seeds.
