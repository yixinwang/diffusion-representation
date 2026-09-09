# Sources and attribution

## Frozen repository context actually inspected

Repository: yixinwang/diffusion-representation.
Requested branch: agent/observation-transport-audit-20260908.
Resolved frozen commit: ce2b6ba6b1075e62fe048b42e84110632688fabb.
Reads were through the connected GitHub tool, not a successful local clone.
The following Git blob identities were returned by the connector:

| File under research/transport_iteration_20260908 | Git blob SHA |
| --- | --- |
| README.md | 504999dd58aec7a1c45fd4d7bf2dc6f1a1f9fdf1 |
| psc_trapezoid_mechanism/environment.json | 896a7f70520b4679f743de73b8fb1439cf666baf |
| trapezoid_pair/learner.py | 3b9d9a8bb92c4a8ce2c2d67b39faffa2115e76f1 |
| trapezoid_mechanism/fixture.py | 29fb9ed9cc0b02bc0e749755ad1c4d2c5d04c9e1 |
| trapezoid_pair/reference.py | cdbe6de953b45d967c72d1228a57396ab9333f04 |

Also inspected commit metadata/diffs for ce2b6ba, effc2c5 and 024c1e3.
The complete resolved scientific source commit was
effc2c565b050193561978c9bce686b3a3f996eb, and the result-publication commit was
024c1e383e168fc3e2ace7373af66d50b66b53f2. These are not Pro15 publication commits.
The environment receipt reports 22 source-file SHA256 values; this review did
not download and independently hash all historical source or PSC payload bytes.
No historical Pro13 binary-state delivery is inferred from a text receipt.

model.py independently packages the inspected trapezoid formulas and retains
the important exact endpoint-affine identities from reference.py. The known
uniform first-root assumption and block-shared context functions come from the
actual inspected fixture. The parameter noise reduction therefore pools true
pairs within a block, not arbitrary independent phase-specific functions.

## Primary literature checked online

Cyrill Scheidegger, Julia Hörrmann and Peter Bühlmann (2022),
*The Weighted Generalised Covariance Measure*, Journal of Machine Learning
Research 23(273), 1–68.
https://www.jmlr.org/papers/v23/21-1328.html

The primary abstract describes weighted covariances of conditional-regression
residuals, including fixed/estimated weighting ideas. The general weighted
conditional-covariance idea is not claimed as new in Pro15. The full article
was not analyzed here; the proofs in THEOREM.md are self-contained adaptations
for this specified copula family, not claims to reproduce the paper's theorems.

Shalev Shaer, Gal Maman and Yaniv Romano (2023),
*Model-X Sequential Testing for Conditional Independence via Testing by Betting*,
AISTATS, Proceedings of Machine Learning Research 206, 2054–2086.
https://proceedings.mlr.press/v206/shaer23a.html

The primary abstract establishes prior conditional-independence betting work.
The implemented comparator is an elementary sample-split likelihood-ratio
specialization with exact conditional-uniform marginals, not a replication of
that paper's algorithm. No PDF was needed or analyzed for either attribution.
