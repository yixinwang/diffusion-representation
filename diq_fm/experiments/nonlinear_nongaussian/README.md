# Nonlinear non-Gaussian quotient and density benchmark

The target is generated in latent coordinates `(s, q, r)`:

- `s` is a two-dimensional context-determined state plus small measurement noise;
- `q` is a two-dimensional context-dependent Student-t Gaussian mixture with a banana warp;
- `r | q, context` is a correlated conditional Gaussian with nonlinear mean;
- the observation is the invertible nonlinear shear `x = (s + h(q,r), q, r)`.

DIQ learns `h` without latent labels from pairs of independent futures with the same context. The
population objective is one half of the squared difference between encoded states. Its unique
minimizer is the inverse shear, up to a fixed additive gauge.

The density model then fits a mixture flow to `q` and a one-step conditional fiber for `r`. Baselines
are a misspecified linear shear, a full eight-dimensional Gaussian mixture, a rank-four PCA latent
model, and a four-dimensional beta-VAE latent model.

Verified configuration: 10 seeds for the main comparison, 40 paired training examples per context,
8 contexts, 800 test examples per context. The beta-VAE audit uses 5 paired seeds and chooses beta
from `{0, 1e-4, 1e-3, 1e-2}` using validation only.
