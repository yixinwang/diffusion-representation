# Exact joint cross-plane coupling and frozen falsifier

## Single changed layer

The exact parent flow supplies coarse/global coordinates, frozen binary state `s`, and an orthonormal rank-eight fine-stage projection. Split the projection as

\[
p=(c,t),\qquad c,t\in\mathbb R^4.
\]

The conditioner `c` and state `s` remain unchanged. With

\[
v=c/\|c\|,\qquad q=A_s v,\qquad
a_s(c)=0.72\,q/\sqrt{1+\|q\|^2},
\]

and `u=t/||t||`, apply the spherical Möbius map on `S^3`, retaining the target radius. In ambient projected coordinates,

\[
T_s(c,t)=\left(c,\ \|t\|M_{a_s(c)}(t/\|t\|)\right).
\]

The complement of the rank-eight projection is unchanged.

## Exact inverse and log-Jacobian

Because `c` and `s` are unchanged, the inverse uses the same conditional parameter and

\[
M_a^{-1}=M_{-a}.
\]

For a three-dimensional sphere (`t∈R^4`), the exact added log-Jacobian is

\[
3\left[
\log(1-\|a\|^2)-
\log\{1+\|a\|^2+2a^\top u\}
\right].
\]

The projection and orthogonal-complement operations have determinant magnitude one. Composing this layer with the unchanged parent therefore retains the exact standard-Gaussian change-of-variables density. `A_s=0` is exact ordinary-parent recovery.

## Finite aligned advantage

The registered teacher first draws the conditioner and state, then generates the target direction through the inverse conditional Möbius map. R14 is exact in this family. A state-only R13 map cannot remove the cross-plane conditional mutual information. Because the energy score is strictly proper,

\[
\mathbb E ES(Q,Y)-\mathbb E ES(P,Y)
=\tfrac12\mathcal E(P,Q)>0
\]

whenever the state-only and conditional joint laws differ.

The source gate required all five known-truth seeds to pass exact invertibility, finite log-Jacobian, copied-mechanism equality, and positive practical-margin contrasts against zero/identity, random, shuffled-pair, fixed-state Möbius, additive, and parameter-matched nonlinear additive controls.

## Frozen real-data activation and falsifier

For each unchanged state, the conditioner-target matrix is estimated on one deterministic half of the residual-fit role and evaluated on the other. It activates only when:

1. the state has the preregistered minimum number of examples;
2. held-out exact logdet gain exceeds the frozen BIC-per-sample charge;
3. numerical and systems gates pass.

No rank, state, strength, threshold, or post-outcome selector is tuned.

The real smoke fails the mechanism if either domain lacks a practically large proper-energy and cross-plane-dependence gain over the exact parent, or fails either full-flow/positive-rate-latent quality frontier under the unchanged downstream gates.

## First transfer-failure condition

The smallest exact failure conditions are:

- insufficient samples inside an unchanged state;
- cross-plane conditional risk too weak to pay the predeclared complexity charge;
- conditional multimodality outside one Möbius law;
- dependence outside the selected joint projection;
- cyclic fine-to-conditioner feedback, which would destroy triangularity.

The executed real result hits the first two conditions. Another rank, threshold, scalar strength, or activation retuning would repeat a rejected premise.