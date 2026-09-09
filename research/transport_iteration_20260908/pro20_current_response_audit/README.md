# Current innovation response: scoped theory and corrected diagnostic

The [available-model review20](https://chatgpt.com/c/6aa1b515-9e74-83e9-9bec-c9d9334ecb93) correctly identified the current16-anchor/704-follower triangular response and the exact-copy limit. Its initial guessed implementation path returned404, which is retained here as a reported source-read failure; the actual code is qalt/src/qalt/innovation_response.py at3b8c0f7. Pro was explicitly disabled, and this was not Pro execution.

Its proposed anchor-permutation falsifier was mathematically invalid. The independent derivation in THEORY_AND_CORRECTION.md gives the missing KL terms, explains why a failed lower-confidence threshold is inconclusive, and substitutes an achieved predictive-log-score diagnostic with an honest target. It preserves the valid fixed-common-chart risk-gap inequality, including approximation, estimation and optimization requirements, without asserting these requirements hold for native training. No strictly better oracle class or equally cheap exact copy follows.

## Actual-code nonlinear witness

Use one actual720-coordinate response within3072coordinates, leaving other coordinates unchanged. One follower has mean tanh(anchor)^2 minus its Gaussian expectation and unit conditional variance. This normalized nonlinear joint law is non-Gaussian and has zero anchor/follower covariance. Two SiLU hidden units implement the mean exactly in real arithmetic; the existing graph-frame chart realizes the required coordinate.

The exact permutation-score formula is Var(mean), numerically **0.09752374 nats**, while mutual information is at most **0.04652825 nats** by the Gaussian entropy bound. These numerical constants use ordinary quadrature, not an interval certificate; the strict discrepancy follows analytically for every nonzero variance. The exact MI value was not computed. This is a counterexample to an information-estimator claim, not a learned-generation win.

Root ran the actual response and graph-frame implementation on64full-dimensional source draws. Forward discrepancy was2.22e-16, inverse discrepancy4.44e-16, and log determinants were zero. All64draws are whole observations; coordinates are not counted as independent samples. Source hashes match frozen3b8c0f7. A portable replay produced the identical saved-bank hash. No real data or fitted native checkpoint was accessed.

Run the portable check from any directory with the repository Python environment:

```sh
python check.py --repo /path/to/diffusion-representation --output /path/to/new-witness-output
```

The original workspace-specific invocation is retained separately for provenance. Parameter counts, full-dimensional output cost, training QR/Cayley work, common-chart restrictions and the lack of a usable native finite-sample bound are explicit in the theory note. Current real-image results remain pending; the earlier cached-innovation failure remains negative evidence.
