# Nonlinear non-Gaussian identification note

## 1. Conditional-variance identity

Let `C` be context and let `X, X'` be conditionally independent draws from the same law given `C`.
For every square-integrable representation `h`,

\[
\frac12\,\mathbb E\|h(X)-h(X')\|^2
=\mathbb E\,\operatorname{tr}\operatorname{Var}(h(X)\mid C).
\]

Conditional on `C`, expand the square. The two conditional second moments are equal, and conditional
independence gives `E[h(X)^T h(X')|C] = ||E[h(X)|C]||^2`. No Gaussian assumption is used.
With one future per context, a cross-fitted conditional mean estimates the same population risk.

## 2. General nonlinear identification theorem

Assume

\[
X=G(S,U),\qquad S=s(C),\qquad U\mid C\sim P_{U\mid S},
\]

where `G` is continuously differentiable and injective. For each state, assume the conditional support
of `U` is connected and has positive density. If continuous `h` has zero conditional-variance risk,
then there is `psi` such that

\[
h(G(s,u))=\psi(s)
\]

on every conditional support. If `dim h = dim S`, `h` has full differential rank on the data manifold,
and the state tangent is not annihilated, `psi` is a local diffeomorphism. Thus the deterministic state
is identified up to smooth reparameterization.

Proof: zero nonnegative risk gives zero conditional variance almost surely, so `h(X)` is constant under
each conditional law. The law depends on context only through `S`; positivity and continuity extend the
equality to the full connected support. The chain rule gives `D(h o G)=[D psi,0]`; the rank conditions
make `D psi` nonsingular, and the inverse-function theorem applies.

## 3. Exact triangular-shear theorem

Let

\[
X=(Y,U),\qquad Y=S+h_\star(U),\qquad S=s(C)+\epsilon_S,
\]

where `U` is arbitrary and non-Gaussian, `epsilon_S` is independent centered noise, and
`h_star(0)=0`. For reversible charts `H_f(Y,U)=(Y-f(U),U)`, paired futures obey

\[
\mathcal L_{pair}(f)=\frac12\mathbb E\|[Y-f(U)]-[Y'-f(U')]\|^2.
\]

If `U` is independent of context and its only zero-variance functions are constants, then

\[
\mathcal L_{pair}(f)=
\mathbb E\|[h_\star-f]-\mathbb E(h_\star-f)\|^2+\mathbb E\|\epsilon_S\|^2.
\]

Under the gauge `f(0)=h_star(0)=0`, the unique population minimizer is `f=h_star`. This is global
nonlinear chart identification for an arbitrary non-Gaussian innovation law.

## 4. Quadratic-variation regularizer

For

\[
dS_t=b_S(S_t,A_t)dt,\qquad
dU_t=b_U(S_t,U_t,A_t)dt+\Sigma_U(S_t,U_t,A_t)dW_t,
\]

and `X_t=G(S_t,U_t)`, Ito's formula gives represented diffusion coefficient
`D_U(h o G) Sigma_U`. Therefore

\[
\frac{d}{dt}\langle h(X)\rangle_t=
D_U(h\circ G)\Sigma_U\Sigma_U^T D_U(h\circ G)^T.
\]

If `Sigma_U Sigma_U^T >= lambda I`,

\[
\|D_U(h\circ G)\|_F^2\leq
\lambda^{-1}\operatorname{tr}(Dh\,A_X\,Dh^T).
\]

Penalizing represented quadratic variation therefore controls stochastic-fiber leakage. Zero penalty
forces deterministic coordinates to be constant along stochastic fibers.

## 5. Conditional generative parity

Let a reversible chart split a conditional target into `(S,Q,R)`. If the pushed-forward full
conditional flow has triangular form

\[
\dot S=f(S,C,t),\quad \dot Q=u(Q,S,C,t),\quad \dot R=w(R,Q,S,C,t),
\]

with `u` represented by the iterative model and `w` in a declared one-step/local family, coordinate
ODE conjugacy implies exact path and endpoint-law parity with the full ambient flow. Approximate field
error yields the standard Gronwall/Wasserstein endpoint bound.

## 6. Strict VAE/fiber separation

For true fiber `p(r|q,s,c)`, a baseline that makes the fiber independent of `q` has minimum gap

\[
\inf_{a(r|s,c)} KL\{p(q,r|s,c)\|p(q|s,c)a(r|s,c)\}=I(Q;R\mid S,C).
\]

The gap is positive whenever the fiber carries conditional dependence. A diagonal-Gaussian VAE
has an additional correlation/negentropy gap when the true fiber lies outside that family. A
deterministic decoder below ambient dimension is singular for a full-dimensional target. DIQ is exact
when its declared conditional fiber contains the true one; an unrestricted iterative decoder can close
the gap only by paying the omitted fiber complexity.

## 7. Audit boundary

The theorem does not assert that every natural image law has a small innovation quotient. The chart
class, stochastic support, state minimality, and cheap-fiber assumption are substantive and must be
tested by held-out closure defect, innovation spectra, and quality-versus-token curves.
