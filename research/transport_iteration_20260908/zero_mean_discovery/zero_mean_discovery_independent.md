# Phase-unknown discovery in the zero-mean trapezoid world

This is a prospective mathematical calculation, not a new fit, simulation, code change, or native experiment. Take independent complete arrays, observed $C_1\sim U(0,1)$, the declared trapezoid feature $\psi$, and true edge coefficient $\theta(c)=A\sin(2\pi c+\varphi)$ with $A=.075$ and unknown phase. Each of four720-coordinate blocks has an unknown perfect matching of360 pairs. Edges share phase within a block; the calculations below remain valid for any fixed phases. There are $H=1,035,360$ candidate edges and $M=1440$ true edges.

The exact feature moments are $E\psi=E\psi^3=0$, $E\psi^2=1$, $|\psi|\le\sqrt{3/2}$. No pair within an array is counted as an independent array. Context and feature knowledge are public and must be available to every comparator.

## Two observed context-weighted scores remove cancellation

For one candidate edge set $X=\psi(R_i)\psi(R_j)$ and

$$Y=X(\sin(2\pi C_1),\cos(2\pi C_1)),\qquad\widehat\mu=n^{-1}\sum_{a=1}^nY_a.$$

For a true edge $E[X\mid c]=\theta(c)$, hence

$$\mu=\frac A2(\cos\varphi,\sin\varphi),\qquad\|\mu\|=t=.0375.$$

For a false edge, its endpoints belong to different conditionally independent pairs and each has a uniform marginal, so $E[X\mid c]=0$ and $\mu=0$. Moreover $E[X^2\mid c]=1$ for **both** cases: the true-edge correction contains the vanishing third feature moments. Thus $E[YY^T]=I_2/2$ and $\operatorname{Cov}(Y)=I_2/2-\mu\mu^T$. Thresholding $\|\widehat\mu\|$ is phase-agnostic; it uses only observed context and residuals. It is not the unconditional graph statistic from the frozen Lane A learner.

## Honest simultaneous concentration

Choose a threshold $0<\tau<t$. Use a regular net of $J$ directions on the unit circle, $J\ge4$, with $c_J=\cos(\pi/J)$. If a false edge has $\|\widehat\mu\|\ge\tau$, one net direction has projection at least $x=\tau c_J$. Each projected observation has mean zero, variance $1/2$ and absolute bound1.5. Scalar Bernstein and a union bound therefore give false-edge probability at most (see the bounded-distribution Bernstein inequality in [Vershynin, *High-Dimensional Probability*, first edition, Section2.8](https://www.math.uci.edu/~rvershyn/papers/HDP-book/HDP-1.pdf))

$$J\exp\left(-\frac{nx^2}{1+x}\right).$$

For a true edge, project along its true mean direction **only in the proof**. The statistic need not know this direction. The projected mean is $t$, variance $1/2-t^2$, and its centered lower deviation is bounded by $1.5+t$. If the norm is at most $\tau$, this projection is at most $\tau$. With $d=t-\tau$, its failure probability is at most

$$\exp\left(-\frac{nd^2}{1-2t^2+\frac23(1.5+t)d}\right).$$

No independence between tested edges is used. Simultaneous exact graph recovery follows, including degree-one validity, except on an event of probability at most

$$\delta(n,\tau,J)=\min\left\{1,
J(H-M)e^{-n(\tau c_J)^2/(1+\tau c_J)}
+M e^{-n(t-\tau)^2/[1-2t^2+\frac23(1.5+t)(t-\tau)]}\right\}.\tag{1}$$

At $n=2000$, this bound is vacuous. The false-edge mean-vector root-mean-square norm is already $1/\sqrt{2000}=.02236$, compared with true signal norm .0375; the true-direction sample standard deviation is .015789. These moment calculations alone do not predict exact recovery probability, but show why a million-edge guarantee is demanding.

Ordinary floating evaluation of (1), preserved in `zero_mean_discovery_constants.py/.json`, gives:

| Net directions | Optimized sufficient n for error≤.05 | Optimized sufficient n for error≤.01 |
|---:|---:|---:|
|4|64625|71498|
|8|47507|52653|
|16|45214|50071|
|32|45333|50131|
|64|46004|50798|

These are sufficient counts from this bound and numerical threshold optimization, **not optimal sample-complexity lower bounds**. A less delicate rounded choice is $J=16,n=46000,\tau=.0219$, giving the unclipped bound .038593. For $n=52000,\tau=.0216$, it is .005316. The formulas are rigorous concentration inequalities; the printed numerical constants are ordinary floating calculations, not interval certificates. There is no claim that an implementation was run on these arrays.

## A separate information-theoretic limitation at n2000

There is also a genuine minimax obstacle for **exact recovery of all four unrestricted matchings**, not just a weakness of (1). Put a uniform prior on the catalog of four matchings, even reveal all phases to the estimator, and keep context law/feature/chart fixed and known. The catalog size satisfies

$$\log|\mathcal G|=4\log\frac{720!}{2^{360}360!}=8035.5075767.$$

For one true pair, let $Z=\psi(U)\psi(V)$ under independent uniforms. Symmetry and the positive even-power entropy expansion give

$$\mathrm{KL}(1+\theta Z\|1)
=\sum_{k\ge1}\frac{\theta^{2k}EZ^{2k}}{2k(2k-1)}
\le\frac{\theta^2}{2[1-(1.5A)^2]}.$$

Here $EZ^2=1$, $|Z|\le1.5$, and $|\theta|\le A$. Since $E_C\theta^2=A^2/2$, the per-array KL to the context-independent product residual reference is at most

$$K_0=\frac{M A^2}{4[1-(1.5A)^2]}=2.05095743.$$

For clarity, let $P_g$ be the full one-array law for graph $g$, and $Q$ the auxiliary law with the **same true root/context law** and all residual coordinates independent uniforms. Conditional independence of true pairs gives $\mathrm{KL}(P_g\|Q)\le K_0$ for every graph. The unknown root probabilities can even be revealed; they carry no graph information. For $L=|\mathcal G|$ and $\overline P_n=L^{-1}\sum_g P_g^{\otimes n}$, adding and subtracting $\log d\overline P_n$ yields exactly

$$\frac1L\sum_g\mathrm{KL}(P_g^{\otimes n}\|Q^{\otimes n})
=I(G;X^{1:n})+\mathrm{KL}(\overline P_n\|Q^{\otimes n}).$$

Thus $I(G;X^{1:n})\le nK_0$. Note that $\overline P_n$ is a mixture of product laws, not the product of the one-array mixture: the same unknown graph generates every training array. If a decoder has error probability $p_e$, encoding whether it is wrong and, on error, its catalog index bounds conditional entropy by $\log2+p_e\log L$. This is the standard Fano reduction; see [Yu (1997), *Assouad, Fano, and Le Cam*](https://web.stanford.edu/class/stats300a/REFS/yu1997assouad.pdf). It gives

$$\Pr(\widehat G=G)\le\frac{nK_0+\log2}{\log|\mathcal G|}.\tag{2}$$

At $n=2000$, (2) is at most .51056. Consequently **no estimator has uniform95% exact-all-matchings recovery over this unrestricted class from2000 arrays**, even with known phase. Necessary $n$ for that uniform target is at least3722 from this bound; this is not sufficient. An optional ordinary series calculation gives actual per-array KL to product about2.0276049, but the displayed analytic upper bound already establishes the conclusion.

This is an average-catalog/minimax exact-graph statement. It does not exclude easier preselected graphs, approximate recovery, better global matching algorithms, or useful generation without exact graph recovery. It is not a lower bound on generative KL. The Lane A learner reserves only2000 arrays for graph discovery; using all4000 for graph selection would be a newly declared estimator and would require revisiting its independent regression-split proof. No change to that frozen learner is proposed here.

## Squared dependence and a paired U-statistic

A phase-invariant unbiased squared-signal statistic is

$$T_n=\frac1{n(n-1)}\sum_{a\ne b}X_aX_b\cos(2\pi(C_a-C_b))
=\frac{\|\sum_aY_a\|^2-\sum_a X_a^2}{n(n-1)}.$$

It needs only the same two weighted sums and the squared-feature sum, not an actual quadratic-time pair loop. Its mean is zero for a false edge and $t^2=.00140625$ for a true edge. Negative values are possible and should not be silently clipped into supposed unbiased evidence.

Writing $m^2=\|\mu\|^2$, the first Hoeffding-projection variance is $m^2/2-m^4$, and the degenerate kernel variance is $\operatorname{tr}(I/2-\mu\mu^T)^2=1/2-m^2+m^4$. Hence exactly

$$\operatorname{Var}(T_n)=\frac4n(m^2/2-m^4)+\frac2{n(n-1)}(1/2-m^2+m^4).$$

For a false edge this is $1/[n(n-1)]$. At n2000 the null standard deviation is about .0005001 and the true-edge standard deviation is .0012852, close to the .00140625 signal. The degenerate null has a better variance rate than a generic bounded U-statistic, but its million-edge tails still need proof.

Naively applying bounded order-two U-statistic Hoeffding with kernel in $[-2.25,2.25]$ and threshold $t^2/2$ gives a sufficient count around690 million for a5% global error target. This is a deliberately retained **very loose bound**, not a suggested sample size or a fundamental requirement. Sharper degenerate inequalities or the original vector concentration avoid this amplitude-to-the-fourth penalty. Squaring the same weighted sums does not evade the catalog information limitation (2), and it adds no new observations or known phase. Global matching/pooling may improve the upper bound, but needs an explicit new estimator and analysis rather than a claim that cancellation is solved at no statistical cost.

## Independent conservative numerical cross-check

A second calculation using Python Decimal at 60 and 100 digits substituted the simpler lower bound cos(pi/16) > 0.98 into the false-edge concentration term. The resulting total upper bound at n=46000 and threshold 0.0219 is 0.03911727863, still below 0.04. The catalog information bound independently gives success at most 0.51055990776 at n=2000. These are ordinary high-precision evaluations of the stated inequalities, not interval certificates. The complete numerical outputs are in zero_mean_discovery_root_check.json. No new observations or fitted models were used.
