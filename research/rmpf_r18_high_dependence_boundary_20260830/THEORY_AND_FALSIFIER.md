# RMPF-R18 exact regime derivation and falsifier

R18 does not change the probability map.  For unchanged source-fit context `v`, state `s`, and target unit direction `u`, the frozen R16 coupling uses

\[
q_s(v)=B_s v,\qquad
a_s(v)=\rho q_s(v)/\sqrt{1+\|q_s(v)\|^2},\qquad \rho=0.72,
\]

and the four-dimensional spherical Möbius map

\[
M_a(u)=\frac{(1-\|a\|^2)u+2(1+a^\top u)a}{1+\|a\|^2+2a^\top u}.
\]

The conditioner is unchanged, so `M_a^{-1}=M_{-a}`.  The exact added log-Jacobian is

\[
3\left[\log(1-\|a\|^2)-\log\{1+\|a\|^2+2a^\top u\}\right].
\]

The complete Gaussian-base likelihood remains exactly normalized, and gamma zero recovers the parent.

## Source-only regime statistic

\[
H(G)=\|a_{S(G)}(V(G))\|_2.
\]

The state, context, and matrices are source-fit and frozen.  `H` is measurable with respect to the global/source coordinates and is invariant to arbitrary replacement of fine and target coordinates.  Conditioning on `A_q={H\ge\tau_q}` changes only the evaluation law.

## Exact oracle headroom

Let `P_q` be the true conditional law and `Q_{0,q}` the identity-coupling law inside `A_q`.  Strict propriety gives

\[
\Delta_q=\mathbb E ES(Q_{0,q},Y)-\mathbb E ES(P_q,Y)
=\tfrac12\mathcal E(P_q,Q_{0,q})\ge0.
\]

Across the five fixed seeds the frozen lower endpoint is

\[
L_q=\bar\Delta_q-t_{0.975,4}s_q/\sqrt5.
\]

The practical regime exists only when `L_q >= 0.005` and every seed retains at least 96 reference and generated examples.  Candidate masses are fixed at `{1/8,1/4,1/2}`; the full law is diagnostic only.  The selected regime is the smallest passing mass.

## Prediction and failure boundary

At fixed context the angular departure from identity is controlled by the Möbius parameter, so `H` is a natural source-only dependence-strength statistic.  It is not guaranteed to monotonically order the joint full-sample energy distance because cross-context distances enter the energy score.

The first failure conditions are:

1. `H` orders likelihood/dependence strength but not the registered energy benefit;
2. the high-`H` tail is too rare for finite certification;
3. the frozen Möbius family has less than 0.005 conditional proper-energy headroom;
4. source-fit thresholds are unstable.

Executed evidence supports the first and third explanations: NLL and dependence separation increased in smaller tails, while every proper-energy lower endpoint stayed below 0.005.  Under the frozen rule, failure of all candidates retires this coupling family.  It does not prove a universal impossibility for other statistics or other flow families.