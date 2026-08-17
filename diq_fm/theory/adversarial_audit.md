# Adversarial audit: nonlinear settings

## A. Pair loss does not require Gaussian data

The proof expands conditional second moments only. Student-t, mixtures, heteroscedasticity, and
multimodality are allowed. A unit test compares the pair estimator with the known conditional variance.

## B. Zero conditional variance alone can collapse

A constant code has zero variance. DIQ prevents this with a reversible full chart, fixed coordinate
dimension, a state-spread/conditioning gauge, and rank assumptions.

## C. Identification is only up to reparameterization

The general theorem cannot select a preferred coordinate gauge. Dense-rotation experiments fit an
affine alignment on training data and report held-out recovery; raw latent coordinates are not compared
directly to ground truth.

## D. A flexible chart could hide the full generator

The theorem experiment uses a shallow orthogonal-plus-shear chart. The full algorithm counts chart and
fiber FLOPs, regularizes Jacobian conditioning, and freezes the chart before scaling the iterative model.

## E. Full ambient GMM optimization

EM uses the same one initialization and 120-iteration cap for every method. Primary evidence is held-out
NLL and paired multi-seed metrics; no method receives additional optimization after test inspection.

## F. VAE claim scope

The strict theorem covers deterministic, independent, diagonal-Gaussian, or finite-rate decoder
families. It does not rank an unrestricted decoder that is itself a full conditional generator; such a
decoder pays the fiber complexity DIQ exposes.

## G. No test leakage

Synthetic latent labels are unused in training. Training and validation select charts, beta, mixture
counts, regressions, and stopping. Ground truth enters only final diagnostics.
