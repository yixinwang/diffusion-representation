# RMPF-R17 exact derivation and falsifier

## Preserved exact flow

The frozen R16/C4G parent maps each visible observation bijectively to an unchanged global conditioner `c`, state `s`, and target `t=r u` with `||u||=1`. R17 changes only the stored 4x4 direction matrix `B_s` in the existing conditional spherical Möbius coupling:

\[
q_s(v)=\gamma B_s v,\qquad
a_s(v)=\rho q_s(v)/\sqrt{1+\|q_s(v)\|^2},\qquad \rho=0.72.
\]

\[
M_a(u)=\frac{(1-\|a\|^2)u+2(1+a^\top u)a}
{1+\|a\|^2+2a^\top u}.
\]

The conditioner and state are unchanged, so

\[
M_a^{-1}=M_{-a}.
\]

The radius and orthogonal coordinates are unchanged. In target dimension four,

\[
\log|\det DT|
=3\left[\log(1-\|a\|^2)-\log\{1+\|a\|^2+2a^\top u\}\right].
\]

Changing the source-fit direction therefore changes no normalization algebra. Gamma zero or `B_s=0` exactly recovers the parent.

## Orthogonalized dependence score

At identity,

\[
\nabla_a\log|\det DT|\big|_{a=0}=-6u.
\]

Within each unchanged state, let `l` be the unit copied-local feature and `v` the unit global context. Define

\[
L=[\mathbf 1,l],\qquad
\Pi_L=L(L^\top L)^+L^\top,
\]

\[
E=(I-\Pi_L)U,\qquad L^\top E=0.
\]

Fit the residual conditional mean in the unchanged global feature space,

\[
\widehat W=(V^\top V)^+V^\top E.
\]

For a uniform direction on `S^3`, the identity Fisher information is

\[
I_a=\mathbb E[(-6u)(-6u)^\top]=9I_4.
\]

The fixed natural-gradient conversion is

\[
\widehat B_\perp
=-\frac{4}{2\cdot3\cdot0.72}\widehat W^\top
=-\frac{2}{3\cdot0.72}\widehat W^\top.
\]

This stores the same two 4x4 matrices and has the same nonzero-amplitude inference cost as R16. The unchanged source-only gamma/BIC rule evaluates `{0,0.5,1,1.5,2,3}`.

## Finite proper-energy condition

For an aligned teacher in the same conditional Möbius family, exact recovery gives a strict energy-score advantage over identity:

\[
\mathbb E ES(Q_0,Y)-\mathbb E ES(P,Y)
=\tfrac12\mathcal E(P,Q_0)>0.
\]

On the compact parameter ball `||a||<=rho<1`, the inverse Möbius map is Lipschitz in its parameter. Coupling estimated and oracle generators with the same source yields

\[
\Delta_E(\widehat B)
\ge \tfrac12\mathcal E(P,Q_0)
-2L_\rho\{\mathbb E\|a_{\widehat B}(V)-a_{B_*}(V)\|^2\}^{1/2}.
\]

Therefore the registered 0.005 margin can hold only if the oracle energy advantage exceeds 0.005 plus the finite direction-estimation term. The preregistered true-family oracle measured the first term under the unchanged R16 teacher.

## Impossibility and first failure

If

\[
\mathbb E[(I-\Pi_L)U\mid V,S]=0,
\]

then the residual cross-score operator is zero and every direction in the fixed global linear class has zero first-order source log-score gain.

Even with a nonzero operator, the practical gate is impossible on the frozen teacher if the exact true-family oracle does not clear the 0.005 proper-energy margin. Additional failures are dependence outside the four global coordinates, within-state multimodality outside one Möbius map, and source-log-score versus downstream-proper-quality target mismatch.

## Executed falsifier

The residual direction was nonzero, source-score orthogonality was `2.115e-16`, and gamma 1.5 was selected in all five seeds. It improved identity NLL and dependence, but identity-minus-R17 proper energy was only `0.001798 [0.001247,0.002349]`. R16 direct likelihood was better on NLL and dependence. The finite true-family oracle interval was `0.002092 [0.000033,0.004151]`, whose upper endpoint was below 0.005. The source gate therefore failed before real development.
