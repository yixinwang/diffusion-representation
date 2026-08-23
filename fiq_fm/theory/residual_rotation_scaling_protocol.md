# Residual-rotation sample-complexity repair

The first registered confirmation at 12,000 training observations failed the projector and one per-unit response-rank gate while passing all NLL, leakage, equivalence, and baseline-separation gates. This follow-up does not reinterpret or rerun that confirmation. It tests the finite-sample implication of the existing perturbation theorem with a new seed family and an unchanged v3 estimator.

## Frozen mechanism and data law

The nonlinear/non-Gaussian active law, tied-marginal residual covariance, signed-permutation and Haar arms, feature dictionary, two-split cross-response operator, commutant estimator, ridge, covariance heads, priors, metrics, and thresholds remain exactly those in `residual_rotation_protocol.md` and commit `c8ad680`. No failed confirmation seed enters fitting, selection, or inference.

## New development ladder

Development uses seeds `200..204` at training sizes 12,000, 24,000, and 48,000, with 4,000 validation and 8,000 sealed test observations per unit. Every method within a unit receives the same observations and paired rotations. The only variable is training sample size.

A training size is eligible for a new confirmation only if:

1. every signed-permutation and Haar chart passes the frozen response-gap, commutant-gap, and held-out-loss rule;
2. the upper endpoint of the five-unit 95% Student interval for Haar JBD projector error is below `0.08`, leaving margin below the unchanged confirmatory limit `0.10`;
3. Haar JBD remains equivalent to oracle block within `0.02` nat per residual dimension;
4. Haar permutation-minus-JBD and diagonal-minus-oracle NLL intervals remain strictly positive;
5. full covariance remains equivalent to oracle block.

The smallest eligible size is selected. If no size is eligible, the commutant-spectral estimator is rejected for scaling.

## Independent confirmation

If development succeeds, the chosen size and all existing gates are frozen and pushed before a single new 30-unit confirmation on seeds `300..329`. The 14-component Holm family, hard chart rule, `0.02` TOST margin, projector and leakage limits, and `0.01` adequacy ceiling remain unchanged. Failure again rejects the empirical rotation route; no third confirmation is allowed without a new estimator and a new seed family.

This study can establish finite-sample recovery for the declared nonlinear latent law at a dataset size comparable to common image benchmarks. It cannot establish image/video performance, structured-chart efficiency, or universal representation superiority.

## Outcome

No training size was eligible. At 12k one chart failed and the projector upper interval was `0.1346`; at 24k all charts passed but the upper interval was `0.1383`; at 48k all charts passed and the mean improved to `0.0521`, but the upper interval was `0.0917`, above the frozen `0.08` buffer. Seeds `300..329` remain unopened. The v3 cross-fitted commutant estimator is rejected for image/video scaling.
