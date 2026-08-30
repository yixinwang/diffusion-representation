# R21 exactness, transfer bound, and second-harmonic falsifier

## Exact triangular map

For unchanged state `s`, conditioner angle `phi_c`, target angle `phi_t`, and retained target radius `r`, define

\[
h_2(\phi_c)=(\sin\phi_c,\cos\phi_c,\sin2\phi_c,\cos2\phi_c)^\top,
\quad m_s(\phi_c)=\theta_s^\top h_2(\phi_c).
\]

Let

\[
\eta=\operatorname{wrap}\{\phi_t-m_s(\phi_c)\},
\quad \xi=R_s(\eta),
\quad \phi_t'=\operatorname{wrap}\{\xi+m_s(\phi_c)\}.
\]

The conditioner and radius are unchanged. Therefore

\[
\eta=R_s^{-1}(\operatorname{wrap}\{\phi_t'-m_s(\phi_c)\}),
\quad \phi_t=\operatorname{wrap}\{\eta+m_s(\phi_c)\}
\]

is the exact inverse. In polar coordinates the target-plane Jacobian is triangular and radius preserving, so

\[
\log|\det DT|=\log R_s'(\eta).
\]

Summing over the two target planes gives the complete added log-Jacobian. Zero second-harmonic coefficients recover the parameter/compute-matched parent conditioner; an identity spline recovers the ordinary parent flow exactly.

## Finite aligned advantage

Suppose within state `s`

\[
\Phi_t=m_s^{(2)}(\Phi_c)+E_s\pmod{2\pi},
\]

where `E_s` has the fitted periodic-RQ law and the second-harmonic component is nonzero. The exact R21 map Gaussianizes/uniformizes the residual angle. A first-harmonic family has irreducible conditional KL

\[
\Delta_{2|1}
=
\inf_{b\in\mathbb R^2}
\operatorname{KL}\{P(\Phi_t\mid\Phi_c,s)\|Q_b(\Phi_t\mid\Phi_c,s)\}
>0
\]

whenever the second-harmonic conditional phase is not almost surely representable by a first harmonic modulo a statewise constant. Strict propriety of the energy score yields

\[
ES(Q_1;P)-ES(P;P)=\tfrac12\mathcal E(P,Q_1)>0.
\]

The experiment-aligned lower bound for an estimated R21 law is

\[
ES(Q_1;P)-ES(\widehat Q_2;P)
\ge
\tfrac12\mathcal E(P,Q_1)-2W_1(\widehat Q_2,P).
\]

The frozen known-truth gate requires the finite lower endpoint to exceed `0.002` versus the padded first-harmonic control and `0.005` versus identity.

## Source-to-development transfer error

For target law `P`, generated law `Q`, and Euclidean energy score,

\[
|ES(Q,P)-ES(Q',P')|
\le W_1(P,P')+2W_1(Q,Q').
\]

For the parent/child gap `Delta(P,Q0,Q1)=ES(Q0,P)-ES(Q1,P)`, source role `S`, development role `D`, and an `L_G`-Lipschitz inverse observation map `G`,

\[
|\Delta_D^x-\Delta_S^x|
\le
2L_G W_1(P_D,P_S)
+2L_G\{W_1(Q_{0D},Q_{0S})+W_1(Q_{1D},Q_{1S})\}.
\]

This explains why a source target-fiber pass is necessary but not sufficient. The same source identities and frozen state masks minimize the generated-law terms; role shift and nonlinear observation dilution remain the first transfer errors.

## Second-harmonic falsifier and no-headroom boundary

The second-harmonic layer has no headroom when the conditional phase is fully represented by the first-harmonic offset plus the unchanged periodic residual spline. It also fails when:

- the conditional phase requires third or higher harmonics;
- the two target-plane residuals have a copula not changed by coordinatewise splines;
- dependence lies in radii or the orthogonal complement;
- source and development conditional laws alias to the same first two harmonic moments;
- the observation inverse map dilutes the target-fiber effect below the pixel/video margin.

The frozen source falsifier is the joint requirement of incremental held-out log score above `log(n)/n`, padded-parent-minus-child proper-energy lower endpoint above `0.001`, and dependence improvement above `0.01`. The executed child failed the incremental parent comparison before development.