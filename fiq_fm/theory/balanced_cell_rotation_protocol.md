# Balanced-cell residual-rotation study

The cross-fitted feature-regression estimator is retired after a failed confirmation and an ineligible sample-scaling ladder. This study registers a different estimator: direct conditional covariance contrasts from train-only balanced cells in the active space.

## Mechanism

A deterministic median tree recursively partitions the two-dimensional active coordinates. Split dimension alternates by depth; each training node is divided at the midpoint between its two central ordered observations, so leaves have nearly equal training counts. The learned thresholds alone assign validation and test observations.

For each leaf, the estimator computes the centered residual sample covariance and subtracts the training global covariance. These leaf contrasts are passed directly to the commutant-spectral block estimator. No feature regression, response SVD, truth rotation, group label, validation outcome, or test observation enters the chart fit. Validation leaf covariances supply only held-out off-block loss and a hard failure check.

If the cell-averaged population covariance contrasts have irreducible inequivalent blocks, the existing commutant theorem identifies the unordered partition. With `B` leaves, residual dimension `q`, and minimum leaf count `m`, sub-Gaussian covariance concentration gives uniform contrast error of order `sqrt((q + log(B/alpha))/m)` up to the scale factor; the finite commutant perturbation proposition then bounds projector error. Chart fitting costs `O(n d log B + n q^2 + B q^4)` for fixed small `q`, and generation cost is unchanged.

## Development

Development uses only new seeds `500..504`, 12,000 training, 4,000 validation, and 8,000 sealed test observations. Candidate tree depths are 2, 3, and 4, giving 4, 8, and 16 balanced leaves. All other data laws, paired signed-permutation/Haar rotations, covariance heads, priors, information, ridge, metrics, and baseline controls remain those of the original rotation protocol.

A depth is eligible only if:

1. every arm has at least 200 training and 50 validation observations per leaf;
2. every commutant relative eigengap exceeds `0.50` and held-out off-block loss is below `0.05`;
3. the upper endpoint of the five-unit 95% Student interval for Haar projector error is below `0.07`, leaving margin below the confirmatory `0.10` limit;
4. JBD is equivalent to oracle block within `0.02` nat per residual dimension;
5. permutation-minus-JBD and diagonal-minus-oracle NLL intervals are positive;
6. full covariance is equivalent to oracle block.

Select the shallowest eligible tree. If none is eligible, reject balanced-cell registration.

## Confirmation

After the selected depth and implementation are frozen and pushed, run one 30-unit confirmation on untouched seeds `600..629`. The existing 14-component Holm family, `0.02` TOST margin, projector `<0.10`, leakage `<0.05`, oracle/full adequacy `<0.01`, and hard per-unit chart rule remain unchanged. The hard rule replaces the retired response eigengap with minimum train/validation leaf counts, commutant relative eigengap, and held-out loss.

Passing would establish finite-sample residual-subspace recovery and a quality separation from fixed-axis and diagonal fibers on this declared nonlinear/non-Gaussian latent law. It would not establish active-quotient discovery, image/video quality, or structured-chart speed.
