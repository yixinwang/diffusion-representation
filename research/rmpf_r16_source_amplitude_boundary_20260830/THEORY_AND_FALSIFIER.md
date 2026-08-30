# RMPF-R16 exact amplitude derivation and falsifier

For frozen state `s`, unchanged normalized conditioner `v`, and unchanged fitted R15 matrix shape `B_s`, the only new degree of freedom is

\[
q_{s,\gamma}=\gamma B_s v,
\qquad
a_{s,\gamma}=\rho q_{s,\gamma}/\sqrt{1+\|q_{s,\gamma}\|^2},
\qquad \rho=0.72.
\]

The unchanged four-dimensional target direction uses

\[
M_a(u)=\frac{(1-\|a\|^2)u+2(1+a^\top u)a}{1+\|a\|^2+2a^\top u}.
\]

The conditioner is unchanged, so `M_a^{-1}=M_{-a}`. The exact added log-Jacobian is

\[
\ell_\gamma(u,v,s)=3\left\{\log(1-\|a_{s,\gamma}\|^2)-\log(1+\|a_{s,\gamma}\|^2+2a_{s,\gamma}^\top u)\right\}.
\]

Gamma zero exactly recovers the parent flow. Every visible coordinate and every pre-existing density term is retained.

## Finite source rule

The source-only candidate set is `{0, 0.5, 1, 1.5, 2, 3}`. The unchanged matrix shape is fitted on even-indexed residual-fit rows. Each candidate is evaluated on odd-indexed rows using the exact log-Jacobian. The unchanged statewise complexity charge is

\[
16\log(n_{val})/(2n_{val}).
\]

The global score is

\[
Q(\gamma)=\sum_s n_{val,s}\max\{\widehat g_s(\gamma)-\mathrm{BIC}_s,0\}.
\]

The smallest maximizer is selected; a nonpositive maximum selects gamma zero. States with fewer than 128 rows use exact identity fallback.

Around gamma zero under a teacher amplitude gamma-star,

\[
E_{p_{\gamma_*}}[\log p_\gamma-\log p_0]
=I\gamma\gamma_*-\tfrac12 I\gamma^2+O((|\gamma|+|\gamma_*|)^3),
\]

so the source log score has a smooth optimum near the teacher amplitude. Since the log-Jacobian is bounded for `||a||<=rho<1`, finite-candidate uniform concentration gives a sufficient source-selection margin. If the selected model equals the teacher, strict propriety yields a positive population energy-score advantage over the parent.

## Cost

All nonzero candidates execute exactly the same matrix-vector product, norm, Möbius map, sparse update, buffers, FLOPs, and parameter bytes. Only gamma zero uses the identity short circuit. The finite source search adds held-out log-Jacobian evaluations but does not rerun the optimizer per candidate.

## First failure conditions

1. Source-score separation is below finite noise or the frozen complexity charge.
2. A state has fewer than 128 source rows.
3. The source-NLL-optimal amplitude differs from the downstream proper-energy optimum.
4. The frozen matrix direction or single Möbius family is wrong; scalar amplitude cannot repair representation error.
5. Large amplitude saturates the bounded ball parameter and may worsen conditioning without changing asymptotic cost.

The executed result realizes conditions 2 and 3: aligned source NLL selects gamma 1 while gamma 0.5 has slightly better proper energy; real CIFAR has no BIC-adjusted headroom, and real UCF has insufficient state support.
