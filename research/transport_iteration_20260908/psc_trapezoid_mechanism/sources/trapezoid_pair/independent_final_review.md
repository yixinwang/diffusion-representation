# Final independent scalar prototype review

The endpoint correction preserves the ideal continuous map. Its diff adds only the exact first/last flat-segment inverse formulas p/(1+kA) and1-(1-p)/(1+kA); it does not clip source values, repaired discriminants, recovered outputs or probabilities. All existing domain/discriminant checks remain.

Original source SHA256:99784d08f559636da143551488039f6dad76877d74454b3a91d23b5939c86529. That source is retained as reference_attempt1.py.txt, and its failing1,869,210-case results remain in independent_check.json with the original independent_review.md.

Corrected source SHA256:dc423b981eae11b757a4aefc90beb2a91ffec634c55f2c6635ba4b52f0a5baa1. The unchanged broad sweep now passes all1,869,210 cases; results are separately saved in independent_check_fixed.json. An expanded sweep also includes source-probability nextafter neighbors of both0 and1, in addition to all four conditional-CDF thresholds and their neighbors, and u knots/neighbors. It passes2,136,240 cases with no exceptions. Results and reproducible checker are independent_check_fixed_neighbors.json/.py.

In the expanded sweep, maximum probability roundtrip error is1.1102230246251565e-16; inverse/forward reported log determinants cancel exactly in this calculation; maximum disagreement with the separately derived quadratic inverse is3.3306690738754696e-16. The parameter grid has129 values covering both extrema-.45,+.45 andzero; u covers1,025 uniform positions plus knot neighbors. These are ordinary float64 checks, not exhaustive or interval certification.

The algebraic moment/normalization and Jacobian proofs from the original independent review still apply. The scalar conditional CDF is C1 in its transformed coordinate; the full pair map is piecewise differentiable because psi has corners, so determinants are a.e. statements. Very small subnormal source probabilities or upper-tail neighbors can still round under ordinary arithmetic; passing the tests does not establish exact bitwise preservation or an atomic-output forward-KL theorem.

The corrected scalar prototype is suitable to preserve with both its adverse initial result and this successful correction before any separately registered learning experiment. No fits, observations, model selection, timing benchmark or PSC jobs were performed for this review.
