# Flow-Aligned Conditional Reparameterization (FACR)

FACR is a same-dimensional, invertible wrapper for latent flow matching and latent diffusion. It does **not** use a variance gate or choose a subset of latent tokens. Every method sees the same fixed tokenizer, context, predictor, training split, latent dimension, standard-normal source, field architecture, optimizer, number of updates, and sampler NFE.

For context `C`, a shared train-only predictor `m(C)`, and a diagonal residual scale `D(C)`, FACR uses

\[
H_\rho(Z;C)=D(C)^{-\rho}(Z-m(C)),\qquad
G_\rho(Y;C)=m(C)+D(C)^\rho Y.
\]

The identity chart (`latent FM/diffusion`) and centered residual chart are explicit candidates. The exponent and diffusion parameterization are selected using validation latent SWD; test data are untouched. Generation integrates the ordinary full-dimensional latent model in chart coordinates, applies one diagonal inverse, then invokes the shared decoder.

## Why the representation can be strictly easier

Under the conditional location-scale model

\[
Z=m(C)+D(C)R,\qquad E[R\mid C]=0,\quad \operatorname{Var}(R_j\mid C)=1,
\]

`R` may be arbitrarily non-Gaussian. At rectified-flow time `t`, put `q_j=s_j^(1-rho)`. For a tied linear probe, the coordinate state curvature and optimal slope are

\[
h_j=(1-t)^2+t^2q_j^2,\qquad
a_j=\frac{tq_j^2-(1-t)}{t^2q_j^2+(1-t)^2}.
\]

The exact excess caused by one shared field coefficient is

\[
A_\rho=\frac1d\sum_j h_j(a_j-\bar a_\rho)^2,
\qquad
\bar a_\rho=\frac{\sum_jh_ja_j}{\sum_jh_j}.
\]

At `rho=1`, all `q_j=1`, so the scale-induced approximation gap is exactly zero and the quadratic optimization condition number is one. If scales differ and `rho<1`, the gap is strictly positive for interior times. Analogous closed forms hold for VP epsilon and v diffusion objectives. These statements use only second moments, not Gaussianity.

The candidate family contains the identity baseline. Validation selection therefore has an oracle inequality: on an event where every validation estimate is within `epsilon` of its population value, selected risk is at most the best candidate risk plus `2 epsilon`.

## Implemented realistic tasks

The code uses only natural images bundled with `scikit-image`, so reproduction requires no dataset download.

- `image_superresolution`: conditional 4x4-to-16x16 RGB patch generation.
- `video_prediction`: action-conditioned future RGB crop prediction under four camera translations.

Train, validation, and test samples occupy guarded, disjoint vertical bands of each source image. A convolutional 64-dimensional tokenizer is trained once per seed and frozen for all compared generators.

## Reproduce tests

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e . pytest
PYTHONPATH=src pytest -q
```

The experiment runners and full multi-seed result tables are added incrementally after each audited run.
