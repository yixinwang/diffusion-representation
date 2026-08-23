# Adversarial audit

This file preserves the checkable audit record: candidate claims, counterexamples, failed experiments, repairs, and the exact surviving statement. It does not contain private reasoning traces.

## Independent audit roles

The work was repeatedly checked from four deliberately different perspectives:

1. **Geometry checker:** tests support, dimension, invertibility, topology, and gauge freedom.
2. **Probability checker:** tests path-law identities, KL decompositions, conditional families, and source distributions.
3. **Computation checker:** counts every iterative and endpoint operation, including the chart and fiber.
4. **Experimental checker:** tests data splitting, information parity, parameter matching, seed aggregation, and whether a promoted metric actually supports the mechanism.

No external multi-agent runtime was available. These are independent audit passes rather than a claim that 64 software agents were executed.

## Candidate claim 1: a deterministic `d<D` decoder can match a noisy full-dimensional target

**Verdict:** false.

For a locally Lipschitz decoder `D:R^d->R^D`, the image is `d`-rectifiable and has zero `D`-dimensional Lebesgue measure. A target with an ambient density assigns probability one to a set of positive ambient measure. The generated law is therefore singular with respect to the target. A stochastic complementary variable is not optional.

**Repair:** retain `R in R^(D-d)` and model `q(r|z)`.

## Candidate claim 2: active-subspace recovery is enough to fit a cheap block fiber

**Verdict:** false, discovered empirically.

The raw-label moment can recover the two-dimensional active subspace under its eigengap assumptions. A tied residual eigenspace would create an unresolved rotation gauge: a random rotation can turn a block-local fiber into a dense conditional covariance. The current graph only permutes fixed residual axes. Moreover, the synthetic benchmark uses unequal residual marginal variances, so it does not instantiate this tied-eigenspace failure mode.

This failure appeared in full seeds 0 and 1. It was not averaged away.

**Repair:** estimate conditional covariance edge strengths with train-only regression and validation scoring, then choose the maximum-weight pair matching. After this repair, the learned block-minus-diagonal NLL advantage matched the analytic KL gap.

## Candidate claim 3: the residual matching rule is exact for arbitrary block size

**Verdict:** not proved.

For pair blocks, a positive matching margin makes maximum-weight perfect matching exact. For blocks larger than two, the current greedy grouping is a practical heuristic. The digits experiment uses block size four under misspecification and makes no exact recovery claim.

## Candidate claim 4: orthogonal static charts give teacher-free ambient supervision

**Verdict:** proved.

Let `Y=Q^T X` and `B=Q^T V` for orthogonal `Q`. A coordinate predictor `a(Y,t)` induces ambient velocity `Q a(Q^T X,t)`, and

`||V-Q a||^2 = ||Q^T V-a||^2`.

Thus ordinary conditional flow labels train the quotient exactly without first fitting an ambient network. The theorem test checks this identity numerically.

## Candidate claim 5: exact endpoint parity in the triangular class

**Verdict:** proved under explicit assumptions.

If the target admits

`Z=F(U,C),  R=mu(Z,C)+L(Z,C)E`,

where `U,E` are independent Gaussians, `F` is represented/integrated exactly by the latent flow, and the block-Cholesky fiber represents `mu,L`, then sampling the latent flow followed by one fiber draw reproduces the target endpoint law exactly. Orthogonal decoding preserves equality of laws.

This is not a universal theorem for arbitrary image distributions.

## Candidate claim 6: equal active dimension implies representation superiority over every VAE

**Verdict:** false without restrictions.

An unrestricted VAE with a sufficiently rich stochastic decoder can represent the same conditional law. The rigorous separation used here is narrower:

- a finite KL rate creates a rate-distortion lower bound when the latent is forced toward a fixed prior; or
- a diagonal Gaussian residual decoder has a positive conditional KL gap when the true residual covariance contains off-diagonal blocks.

The synthetic benchmark instantiates the second case. The RAE block baseline removes the KL penalty and is correspondingly stronger.

## Candidate claim 7: current implementation has latent-order image-scale complexity

**Verdict:** not established.

The deep iterative field acts only in dimension `d`, and the fiber is one-shot. However, the present chart is a dense `D x D` matrix and costs `O(D^2)` once per sample. For small tabular/image-vector benchmarks this is inexpensive; for images, a structured reversible chart is required before claiming asymptotic latent-order complexity. The manuscript states this boundary explicitly.

## Experimental integrity audit

- Train/validation/test indices are disjoint and tested.
- Standardization is fit only on training data.
- Chart selection, gauge fixing, early stopping, and hyperparameters use only train/validation data.
- Test data are used once for final metrics.
- Every generative method sees the same labels and the same train/validation samples.
- Every iterative solver uses the same Heun NFE.
- The full flow receives a parameter-matched width.
- VAE and RAE baselines use the same latent dimension and latent vector-field architecture as FIQ.
- Reported inference time includes the endpoint fiber and chart.
- All five registered seeds are included.

## Surviving result

The surviving theoretical result is the static orthogonal quotient theorem plus pair-partition recovery under its matching margin. It is exact for the stated triangular family and exposes a strict diagonal-decoder KL gap. The archived empirical tables are not promoted: their referenced seed-level JSON files are absent, and the original digits aggregation command failed. Fresh common-random-number confirmation is required.
