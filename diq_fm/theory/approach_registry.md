# Approach registry

| Family | Concrete mechanism | Audit status |
|---|---|---|
| Reconstruction VAE | Compress pixels with ELBO, then learn latent flow | Blocked for exact density parity; objective need not identify dynamical state |
| Coordinate variance gate | Use heteroscedastic variance in supplied coordinates | Useful diagnostic, but dense nonlinear mixing can spread innovation over every coordinate |
| Pair-invariant nonlinear chart | Minimize conditional variance of selected reversible coordinates | Selected; exact theorem in triangular-shear class |
| Quadratic-variation nullspace | Penalize represented diffusion covariance | Selected continuous-time regularizer; distribution-free with nondegenerate fiber noise |
| Full moving FIQ chart | Triangularize conditional flow labels | Retained for general quotient/fiber parity; requires gauge control |
| Deterministic fiber | Drop residual coordinates | Rejected for full-dimensional non-Gaussian targets |
| Diagonal Gaussian fiber | One-step factorized residual | Strict KL gap under conditional dependence or non-Gaussianity |
| Local empirical/mixture fiber | One-step local residual bank or small mixture | Selected for pixel disocclusion |
| Full latent flow | Iterate on every latent token | Strong quality reference; compute scales with all tokens |
| Innovation-token flow | Pack only high-innovation tokens | Selected practical model; exact under projectability, auditable otherwise |
