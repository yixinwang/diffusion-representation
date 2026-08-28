# RMPF known-truth preregistration

Frozen before outcomes on 2026-08-27.

## Estimand, margin, units, and stop rule

The independent unit is a fitted model seed. Development seeds are 9000--9002; untouched confirmation seeds are 9100--9104. The positive arm uses 8,000 training, 2,000 validation, and 4,000 confirmation samples per seed. The visible dimension is 32; all coordinates are retained.

Primary quality estimand:

\[
\Delta_{\rm NLL}=R_{\rm no\ global}-R_{\rm RMPF},\qquad
R_m=-D^{-1}{\mathbb E}_{P}\log q_m(X).
\]

Promotion to genuine-data development requires, on the five confirmation seeds:

1. inverse error below 1e-10 and analytic/autograd log-Jacobian error below 1e-8;
2. mean NLL gain at least 0.05 nat/dimension with paired 95% interval above zero against no-global/MCQF;
3. global-bit correlation absolute error at least 0.20 lower than no-global and MCQF;
4. energy-score noninferiority to the validation-optimal full global baseline within 0.02;
5. batch latency ratio at most 0.75, single-sample ratio at most 0.90, peak allocated-memory ratio at most 0.75, and stored-byte ratio at most 0.75 against the full attention conditioner;
6. exact-copy sample and density mismatch below 1e-10;
7. no nonfinite/catastrophic seed;
8. the dependence-strength and rank perturbations follow the frozen smooth/threshold prediction.

If any fails, genuine confirmation remains sealed. One failed layer may be revised at a time on development only.

## Data-generating law

Let `D=32`, `A,B in R^16`, `Z_A,Z_B ~ N(0,I)`, and

`g(z)=sinh(0.6z)/0.6`, `g^{-1}(x)=asinh(0.6x)/0.6`.

An orthonormal Haar matrix `H` maps coefficients `Y=(A,B)` to visible `X=YH`. Coarse coordinates evolve through known positive diagonal affine maps. At each of 16 stages, details receive a local additive update. At stages `{3,7,11,15}`, they also receive a rank-four higher-order tree update of strength 12. The tree recursively combines signed tanh leaves by `h <- tanh(2.15 h_L h_R)`, yielding a parity-like global statistic with weak pairwise signal.

All transforms are bijective almost everywhere. Haar has determinant magnitude one, additive detail coupling has determinant one, and the exact log density is

\[
\log p_X(x)=\log\phi(z_A)+\log\phi(z_B)
-\sum_i\log\cosh(0.6z_{A,i})
-\sum_j\log\cosh(0.6z_{B,j})
-\sum_{k,i}\log s_{k,i}.
\]

## Methods and attribution

- B: full visible-coordinate generic global conditioner.
- B+S: exact Haar/coarse/local schedule with no structured global coupling.
- B+S+A: RMPF tensor-tree global coupling plus multirate compiler.
- A=0, random-feature, frozen-output, and sample-shuffled controls.
- parameter/operation-matched generic global features.
- dense full self-attention conditioner at all 16 stages, selected by validation NLL.
- full random-feature conditioner at all stages.
- MCQF-style coarse/local one-pass conditional model.
- equivalence copy, which must tie exactly.

All receive the same source-target pairs, stages, source prior, samples, conditioning, and ridge/tuning budget. Ridge grid: 1e-6, 1e-4, 1e-2. RMPF ranks: 1,2,4. Error thresholds: 0.0025,0.005,0.01. Full attention heads: 1,2,4. No confirmation value is used for selection.

## Mechanism, rivals, falsifier, and prediction

M: global dependence is sparse in update time and low in hierarchical communication rank, not sparse in coordinate dimension. All dimensions remain, but global nonlinear communication is evaluated four rather than sixteen times.

R1: any gain is only the Haar inductive bias. Distinguishing intervention: B+S versus B+S+A and zero/random/frozen/shuffled A.

R2: the full model is undertrained or capacity restricted. Distinguishing intervention: closed-form ridge convergence, validation-optimal full widths/heads, matched controls, and an exact equivalence copy.

R3: pairwise statistics suffice. Distinguishing intervention: parity-like targets with near-zero pairwise correlations and a characteristic/global-bit diagnostic.

Recovery requires the global increment to lie in the rank-r tree span at selected stages and omitted-stage residual energy to be below the controller threshold. If true rank `R>r`, the irreducible conditional-mean error is at least the omitted singular-feature energy. If the communication graph has depth below the leaf-graph diameter, it cannot convey an even-parity bit across disconnected components.

At fixed rank deficit, NLL/proper-score loss is predicted to grow quadratically near strength zero and then smoothly. At fixed strength, loss falls sharply when fitted rank reaches true rank, up to finite-sample smoothing.