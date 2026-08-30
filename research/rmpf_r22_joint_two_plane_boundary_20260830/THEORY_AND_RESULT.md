# RMPF-R22 exact derivation and executed falsifier

## One-layer intervention

Within frozen state `s`, define the unchanged first-harmonic offsets

\[
m_s(c)=\beta_{s1}\sin c+\beta_{s2}\cos c,
\qquad x_j=\operatorname{wrap}\{t_j-m_s(c_j)\}.
\]

RMPF-R22 replaces independent target-plane maps by the joint shear

\[
h_\kappa(x)=\kappa_1\sin x+\kappa_2\cos x,
\qquad H_\kappa(x_1,x_2)=
(x_1,\operatorname{wrap}\{x_2-h_\kappa(x_1)\}).
\]

With the unchanged periodic rational-quadratic map `R`,

\[
T_{22}=H_\kappa^{-1}\circ(R\times R)\circ H_\kappa.
\]

Explicitly,

\[
u_1=x_1,\quad u_2=x_2-h_\kappa(x_1),\quad
y_j=R(u_j),\quad z_1=y_1,\quad z_2=y_2+h_\kappa(y_1).
\]

The inverse is

\[
y_1=z_1,\quad y_2=z_2-h_\kappa(y_1),\quad
x_1=R^{-1}(y_1),\quad x_2=R^{-1}(y_2)+h_\kappa(x_1).
\]

Both `H_kappa` and its inverse have unit determinant. Therefore the exact added Cartesian log-Jacobian is

\[
\log|\det DT_{22}|=\log R'(x_1)+\log R'(u_2).
\]

Radii and all orthogonal-complement coordinates are retained. Zero shear gives the parameter/compute-matched padded R19 parent. Identity splines give exact ordinary-flow recovery.

## Parameter and operation matching

Per state, R22 uses two first-harmonic coefficients, two shear coefficients, and five free RQ knot locations: nine learned degrees per state, eighteen total. The two shear coefficients replace the two unused R21 second-harmonic coefficients. Compact exact state is 314 bytes, matching R21. The frozen scalar conditioner/coupling operation budget is matched.

## Proper-energy transfer and alias boundary

Strict propriety gives the oracle observation-space gain

\[
\Delta_{\rm oracle}^X
=ES(Q_X;P_X)-ES(P_X;P_X)
=\tfrac12\mathcal E(P_X,Q_X)>0
\]

when the laws differ. For fitted law `Qhat`,

\[
\Delta_{\rm fit}^X
\ge \tfrac12\mathcal E(P_X,Q_X)-2W_1(\widehat Q_X,P_X).
\]

For a frozen inverse observation map `G` that is locally `L_G`-Lipschitz, the fitting term is at most `2 L_G W1(Qhat,P)` in angular coordinates. The source gate directly measures target-fiber and full-pixel proper energy, avoiding an unsupported latent-to-pixel inference.

No headroom exists when padded-parent residual planes are already conditionally independent. The first alias classes are symmetric or multibranch torus copulas with vanishing first circular shear score, reverse/cyclic feedback, dependence in radii or outside the selected planes, and observation-space dilution that exceeds the angular gain.

## Frozen gates and executed outcome

Known truth required:

1. identity-minus-R22 energy LCB at least 0.005;
2. padded-R19-minus-R22 energy LCB at least 0.002;
3. additive and compute-matched nonlinear shear LCBs strictly above zero;
4. positive NLL and dependence gains over padded R19;
5. exact inverse/Jacobian, parent fallback, copied-mechanism equality, and matched budget.

The exact algebra and the NLL/dependence gates passed. The proper-energy gates failed. In particular,

\[
\Delta_E(\text{padded R19},\text{R22})
=0.00392484\;[0.00023261,0.00761707],
\]

and the compute-matched nonlinear shear contrast was

\[
0.00014912\;[-0.00042798,0.00072622].
\]

The source-only CIFAR diagnostic then showed negative incremental held-out logdet gains in both states, effectively zero/adverse pixel energy changes, and worse dependence error. UCF had only 39/27 state examples. The frozen stop rule therefore kept development, replication, and confirmation sealed.