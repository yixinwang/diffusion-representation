# Drift-Innovation Quotient Flow Matching (DIQ-FM)

DIQ-FM extends Flow-Induced Quotient Flow Matching to nonlinear, non-Gaussian,
conditional image and video models. A reversible chart separates:

- a deterministic drift state, identified by conditional-variance or quadratic-variation loss;
- a low-dimensional stochastic innovation quotient, modeled by an iterative flow/diffusion;
- a local stochastic fiber, modeled in one step.

The iterative model therefore pays for the innovation dimension, while the reversible chart and
fiber retain information that a VAE bottleneck would discard.

## Verified settings

1. `experiments/nonlinear_nongaussian`: nonlinear triangular observation chart, Student-t/mixture
   innovations, correlated conditional fiber, exact density evaluation, full-ambient and VAE baselines.
2. `experiments/rotated_nonlinear_identification`: all observed coordinates are densely rotated;
   a learned orthogonal-plus-nonlinear chart recovers the deterministic state up to reparameterization.
3. `experiments/latent_video_disocclusion` (added in the next commit): pixel-input moving-crop
   video on real image patches with a learned convolutional latent model and innovation-token flow.

All splits and model selection use training/validation data only. Ground-truth latent variables in
synthetic settings are used only for final diagnostic metrics.

## Reproduce

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e . pytest
pytest -q
PYTHONPATH=src python experiments/nonlinear_nongaussian/run.py
PYTHONPATH=src python experiments/nonlinear_nongaussian/run_vae.py
PYTHONPATH=src python experiments/rotated_nonlinear_identification/run.py
```

The full verified runs are CPU-compatible but take longer than smoke tests. Committed CSV files are
raw per-seed outputs, not hand-entered summaries.
