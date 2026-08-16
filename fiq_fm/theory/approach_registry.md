# Approach registry

This registry records mathematical mechanisms, not wording variants. Routes were kept separate until their exact obligations were clear.

| Family | Mechanism | Concrete result | Status |
|---|---|---|---|
| A. Deterministic bottleneck | Generate `x=D(z)` with `d<D` | Lipschitz image has zero `D`-dimensional volume | Rejected for full-dimensional targets |
| B. Ambient-teacher distillation | Train full flow, then distill a latent model | Can inherit teacher quality | Rejected as a primary route: pays the full training cost first |
| C. Bottlenecked ambient field | Force a full field to depend only on low-dimensional features | Cheap only when normal dynamics are trivial | Blocked by circle/normal-contraction counterexamples |
| D. Tangent-only manifold flow | Integrate only tangent dynamics | Handles exactly singular manifolds locally | Blocked for noisy full-dimensional data because normal density matters |
| E. Moving reversible chart | Learn `H_t(x)=(z,r)` and triangularize the vector field | Exact conjugacy and teacher-free coordinate labels | Correct but too broad for a first verified implementation; chart can hide the generator without gauge/compute controls |
| F. Static orthogonal quotient | Use `Q^T x=(z,r)`, deep flow on `z`, cheap stochastic fiber on `r` | Exact loss isometry; raw-label moment recovery in spiked Gaussian models | Selected restricted theorem |
| G. Endpoint latent/fiber density factorization | Learn `q(z)` and `q(r|z)` under a reversible endpoint chart | Exact KL chain rule; one-shot fiber | Selected practical generator |
| H. Residual gauge discovery | Learn residual block structure from conditional covariance edges | Exact recovery under pair-block margin via maximum-weight matching | Added after empirical failure |
| I. Unrestricted nonlinear quotient discovery | Shallow coupling chart plus cross-fitted closure heads | Observable held-out quotient defect | Open extension; not needed for the verified claims |
| J. Atlas of local charts | Patch multiple quotients when topology forbids one chart | Avoids global-coordinate impossibility | Open extension |

## Why family F+G+H was selected

1. **Loss fidelity.** Orthogonality makes the coordinate squared error exactly equal to the induced ambient squared error.
2. **Source fidelity.** An isotropic Gaussian source is invariant under the chart.
3. **No hidden generator.** A fixed linear chart cannot absorb a deep nonlinear transport.
4. **Full support.** The stochastic fiber retains the complementary randomness that a deterministic autoencoder discards.
5. **Latent-order iteration.** Only `z` is integrated by the deep ODE; the fiber is sampled once.
6. **Discoverability.** The active subspace is estimated from raw flow labels, and the previously unidentified residual gauge is fixed from a held-out conditional-dependence objective.

## Blocked lemmas and exact gaps

- A universal low-dimensional quotient for arbitrary distributions is false.
- A general guarantee that a learned moving chart remains cheap is false without explicit architecture/FLOP restrictions.
- Exact recovery of residual blocks larger than pairs by the current greedy grouping rule is not proved.
- A universal representation dominance theorem over unrestricted VAEs is false; the verified comparison requires a rate constraint or conditional residual-family misspecification.
- The dense orthogonal chart used in the small experiments costs `O(D^2)` once. Image-scale latent-order complexity requires a structured chart such as lifting, wavelets, local coupling, or butterfly factors.
