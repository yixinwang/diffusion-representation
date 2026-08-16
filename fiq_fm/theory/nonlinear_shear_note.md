# Nonlinear, non-Gaussian quotient theorem and audit

## Model

Let `[U,U_perp]` be orthogonal, `Z_s in R^d`, `R in R^(D-d)`, and let `beta` be a
nonlinear `C^1` map. The paired source and target are

\[
X_s=U\{Z_s+\beta(R)\}+U_\perp R,\qquad s\in\{0,1\}.
\]

No Gaussian assumption is imposed on `Z_1` or `R`. The experiment uses a four-component
full-covariance active mixture and a warped two-component fiber mixture.

## Identification theorem

Assume `E Z_0=0`, `(Z_0,Z_1)` is independent of `R`, finite second moments, and
`Cov(Z_1-Z_0)` is positive definite. Then

1. `V=X_1-X_0=U(Z_1-Z_0)`, hence `range Cov(V)=span(U)`.
2. With `A_0=U^T X_0` and `R_0=U_perp^T X_0`,
   `E[A_0 | R_0=r]=beta(r)`.
3. The chart
   \[
   H(x)=\big(U^T x-\beta(U_\perp^T x),\ U_\perp^T x\big)
   \]
   is globally invertible with inverse
   \[
   G(z,r)=U\{z+\beta(r)\}+U_\perp r.
   \]
4. In rotated coordinates `DG=[[I,D beta],[0,I]]`, so `det DG=1`.
5. Therefore `H(X_s)=(Z_s,R)`. Any exact latent flow on `Z`, together with the identity or
   a cheap conditional fiber, gives the exact ambient endpoint law and density.

The proof is direct: the shared shear cancels from the paired velocity; orthogonal projection
recovers the conditional regression; composing `H` and `G` gives identity; block triangularity
gives determinant one.

## Strict baseline separation

Without nonlinear residualization, the retained coordinate is `A=Z_1+beta(R)`. If `beta(R)`
is nonconstant and the characteristic function of `Z_1` is nonzero near zero, then `A` and `R`
are dependent. Consequently

\[
\inf_{q_A,q_R} KL(P_{A,R}\|q_Aq_R)=I(A;R)>0,
\]

whereas the nonlinear chart has zero population factorization error under realizability. An
unrestricted stochastic decoder can remove this gap only by becoming the full conditional fiber;
its cost must then be counted.

## Finite-sample transition

Velocity-covariance concentration and Davis--Kahan give active-subspace error
`O(sqrt(D/n)/eigengap)`. Cubic regression has `p` features and prediction error
`O(dp/n)` up to conditioning and tail factors. The improvement becomes visible only when

\[
\mathcal I_{\rm eff}=\frac{n\alpha^2}{dp}
\]

crosses a constant threshold. This is not merely post-hoc: `alpha^2` is the misspecification
signal removed by the nonlinear chart and `dp/n` is its estimation cost.

## Executed audit

Five seeds, three sample sizes, and three shear strengths were run. At `n=2048, alpha=1`, the
nonlinear quotient lowers test NLL by 1.206 nats relative to the same-dimensional linear
flow-active representation (`p=1.9e-5`), by 1.244 relative to PCA (`p=1.2e-5`), and by 0.898
relative to a full ambient GMM (`p=1.0e-5`), while iterative dimension is 3 instead of 8.
The estimated active-projector error is `8e-4`.

The registered failure regime is retained: mean NLL gain over the linear chart is -1.795 at
`(n,alpha)=(256,.25)`, -0.150 at `(512,.5)`, and +1.206 at `(2048,1)`. Regressing gain on
`log2(n alpha^2/(dp))` gives slope 0.396 and `R^2=0.700`. The fitted zero-gain threshold is
1.46 for this estimator and data distribution.
