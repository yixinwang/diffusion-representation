# Pro11: finite learning of unknown matched, context-varying non-Gaussian dependence

## Status and scope

This is a standalone derivation and locally executed numerical inspection package. It is not a native image experiment, a PSC result, or a theorem for a learned Pro9 analysis network. Repository inspection was read-only at `f024cc56651da7dc161d555117d9373a4ac8fa99` of `yixinwang/diffusion-representation`. No repository file, branch, job, or protected dataset was changed or opened.

The positive result is an expected **joint** KL bound for a normalized full-dimensional generator whose root distribution, arbitrary pair matching, and four unknown nonlinear context functions are learned from observations. Its comparator lower bound is only for conditionally independent scalar decoders in the specified fixed chart. A same-information analytic implementation and a copied decoder can tie it exactly. No global-FM, native perceptual, semantic-representation, training-to-quality, or speed superiority is proved.

The copula and moment estimator are established constructions, not new families. The contribution of this note is the explicit composite finite-sample certificate, its observable prediction consequence, and a deliberately small changed-law failure. References and inspection limits appear in `PROVENANCE.md`.

## 1. Why pairwise structure alone is insufficient

Even give the learner the root law and the correct matching. Partition a uniform scalar context into K equiprobable cells. In each cell and on each of M pairs independently choose an unknown coefficient +b or -b, with 0<b<1/2, in the pair density defined below. With no regularity or parameter sharing, a cell not seen among N training arrays contains no information about its independent signs.

The Bayes predictive conditional density on that cell is the product uniform law. Its expected log-loss gap is M F(b), where F(b)>=b^2/2. A new context is unseen with probability (1-1/K)^N. For K>=2N this is at least 1/2. Consequently the minimax expected joint KL for this broader class is at least M b^2/4. This is a direct unseen-context construction; dimensionality alone, invertibility, or knowing a pair graph does not remove it.

The positive class below excludes this obstruction through smoothness and within-block sharing. It does not assume that all pooled sites are independent training arrays.

## 2. Positive class and public information

Work on the open unit cube in a **fixed public chart**, with coordinates

\[
 X=(C,U_1,\ldots,U_G),\qquad C\in(0,1)^c,\quad U_b\in(0,1)^{2m},\quad D=c+2Gm.
\]

A known coordinate permutation, public invertible preprocessing, or reshape can be incorporated without changing KL, with its cost and Jacobian included. An unknown learned analysis map is not included for free. In particular the known uniform residual margins here are an assumption, not a conclusion about Gaussianized native images.

The assumptions, available equally to all controls, are:

1. The N complete arrays are iid. Root coordinates are independent. Each root density is an unknown positive histogram on k0 equal bins. The first root density is at least r*>0. The learner estimates every root bin mass, including those of C1.
2. Each group has an **unknown arbitrary perfect matching** E_b of its 2m coordinates. The admissible search space is all matchings, not a teacher-supplied finite catalogue. All candidate within-group edges are inspected.
3. Given the root, groups and their matched pairs are independent. A matched pair has density
   \[
   p_{\theta_b(C_1)}(u,v)=1+\theta_b(C_1)\psi(u)\psi(v),\qquad
   \psi(u)=\sqrt 2\cos(2\pi u).
   \]
   Each unknown function theta_b is shared by the m pairs in its group. Its argument C1 is known; its values, shape, signs, and nonlinear variation are not supplied to the learner.
4. The functions obey |theta_b|<=kappa<1/2, Lipschitz constant at most L, and |E theta_b(C1)|>=a>0. Neither a sinusoidal form nor a finite-dimensional functional catalogue is assumed in the theorem.
5. The K equal context cells refine the root histogram bins: K is a multiple of k0. Thus the true context distribution is uniform within each fine cell. Each cell has probability at least p_min=r*/K.

The pair density is between 1-2kappa and 1+2kappa and integrates to one. Both scalar margins are uniform, even conditional on the whole root. This is a nonlinear, non-Gaussian dependence class. The root can induce higher-order dependence across different pairs through the shared context.

For different coordinates i,j,

\[
 E[\psi(U_i)\psi(U_j)\mid C]
 =\begin{cases}\theta_b(C_1),&\{i,j\}\in E_b,\\0,&\text{otherwise}.\end{cases}
\]

In contrast, both ordinary covariance and covariance after coordinatewise Gaussianization vanish. Indeed,

\[
 \int_0^1(u-1/2)\psi(u)du=0,
 \qquad\int_0^1\Phi^{-1}(u)\psi(u)du=0.
\]

The second identity follows from oddness of Phi-inverse around 1/2 and evenness of psi around 1/2. A Gaussian covariance fit does not identify this dependence; a same-information nonlinear pair control can.

## 3. Explicit estimator: no teacher graph or optimization oracle

Reserve ns of the N arrays for graph estimation and n=N-ns independently for conditional estimation. Root estimation may use all N arrays.

**Root.** Estimate each root bin probability by (count+1)/(N+k0).

**Pair graph.** For every candidate within-group edge form

\[
 \widehat S_{ij}=n_s^{-1}\sum_{t=1}^{n_s}\psi(U_{ti})\psi(U_{tj}).
\]

Retain an edge iff |S_hat_ij|>a/2. Accept a group's graph only if each node has degree one. Otherwise that group explicitly falls back to a product decoder with coefficient zero; an arbitrary canonical pairing then has no distributional effect.

**Nonlinear context.** For an accepted group and each parameter-estimation array form

\[
 T_{tb}=m^{-1}\sum_{\{i,j\}\in\widehat E_b}\psi(U_{ti})\psi(U_{tj}).
\]

Average T_tb within each observed C1 cell, clip the average to [-kappa,kappa], and use zero for an empty cell. This gives the K-bin estimate theta_hat_b. This is estimation of an unknown nonlinear function, not evaluation of a supplied teacher coefficient.

`model.fit()` accepts only observed arrays and the public dimensional/regularity parameters. It has no argument for a true graph, root mass, phase, sign, Gaussian training source, or hidden variable. Counts, matrix products, means, and clipping have no unreported optimization gap.

## 4. Scalar KL curvature

Let h=psi(u)psi(v) under the product uniform measure. Its distribution is symmetric, E h=0, E h^2=1, E h^3=0, and |h|<=2. Define

\[
 F(t)=\int(1+t h)\log(1+t h)du\,dv.
\]

For |s|,|t|<=kappa,

\[
 D(p_t\|p_s)=F(t)-F(s)-(t-s)F'(s).
\]

Symmetry gives

\[
 F''(t)=E\frac{h^2}{1-t^2h^2},\qquad
 1\le F''(t)\le\frac{1}{1-4\kappa^2}.
\]

Consequently, writing B_kappa=1/[2(1-4kappa^2)],

\[
 \tfrac12(t-s)^2\le D(p_t\|p_s)\le B_\kappa(t-s)^2,
 \qquad F(t)\ge t^2/2.
\]

These are global bounds on the stated parameter interval, not a second-order approximation at independence.

## 5. Finite joint-KL theorem

Define M=Gm and E=G binom(2m,2), the number of pairs and candidate edges. Set

\[
\begin{split}
 \delta_s&=\min\left\{1,2E\exp\left[-\frac{n_s a^2}{8\{1+(2+\kappa)a/6\}}\right]\right\},\\
 v&=\frac{L^2}{12K^2},\quad p_{\min}=\frac{r_*}{K},\\
 A_n&=\frac{K}{n+1}\left(1+\frac{3}{(n+2)p_{\min}}\right).
\end{split}
\]

Let Q_hat be the full fitted density above. Let Q_hat,J use the normalized continuous CDF-grid implementation in Section 9, with J grid intervals per pair. Then

\[
\boxed{
\begin{split}
E_{\mathcal T}D(P\|\widehat Q_{\mathcal T,J})\le{}&
\frac{c(k_0-1)}{N+1}\\
&+B_\kappa\{Mv+(G+Mv)A_n+M\kappa^2e^{-np_{\min}}\}\\
&+\delta_s M\log\frac{1+2\kappa}{1-2\kappa}\\
&+\frac{2\pi\kappa M}{J(1-2\kappa)}.
\end{split}}
\]

This is an expectation over independent fitting arrays, for a fresh target array. It is **not** a statement that every fitted seed has KL below the displayed value. It is not KL of the atomic machine-output distribution against a continuous target.

### Proof: root learning

For one root coordinate with bin masses p_j and add-one estimate q_j,

\[
 E\frac{1}{N_j+1}=\frac{1-(1-p_j)^{N+1}}{(N+1)p_j}.
\]

Applying Jensen to the log separately for every bin gives

\[
 E D(p\|q)\le\log\frac{N+k_0}{N+1}\le\frac{k_0-1}{N+1}.
\]

The constant histogram density multipliers cancel in the KL. Sum over c independent root coordinates.

### Proof: unknown graph

For a true edge, S has mean E theta_b(C1), variance at most one, and centered absolute value at most 2+kappa. For a false edge its mean is zero and variance is one. Bernstein's inequality at deviation a/2, followed by a union bound over E candidate edges, gives probability at most delta_s that any score violates its required accuracy. On the complementary event the thresholded graph is exactly the true matching in every group.

This probability counts ns independent **arrays**, not ns times the number of edges. No independence between edge scores is used in the union bound.

### Proof: unknown context and strengths

Conditional on the correct graph, the n parameter arrays are still independent of the graph-selection event. For the true matching,

\[
 E[T_b\mid C]=\theta_b(C_1),\qquad
 \operatorname{Var}(T_b\mid C)=(1-\theta_b(C_1)^2)/m\le1/m.
\]

The conditional independence of pairs justifies this variance reduction. It does not make their common contexts independent. Within each context cell the Lipschitz bound and uniform context law give

\[
 \operatorname{Var}(\theta_b(C_1)\mid\text{cell})\le L^2/(12K^2)=v.
\]

For B~Binomial(n,p), the elementary inequality

\[
 1/k\le1/(k+1)+3/[(k+1)(k+2)],\quad k\ge1,
\]

implies

\[
 E[1/B;B>0]\le\frac1{(n+1)p}+\frac3{(n+1)(n+2)p^2}.
\]

Multiplying by cell probabilities and summing bounds the inverse-count contribution by A_n. Conditional cell averaging has variance at most (1/m+v)/B. Clipping cannot increase squared error relative to any true coefficient in [-kappa,kappa]. Empty cells add at most kappa^2 exp(-n p_min). Therefore

\[
\sum_bm E\int(\theta_b-\widehat\theta_b)^2dP_{C_1}
\le Mv+(G+Mv)A_n+M\kappa^2e^{-np_{\min}}
\]

on the successful structural event. The Mv*A_n term is precisely the cost of random shared contexts that would be lost by falsely counting all sites as independent examples.

Apply Section 4's upper curvature bound and the conditional product factorization. On a bad graph event both true and fitted conditional densities remain between (1-2kappa)^M and (1+2kappa)^M, so conditional KL is at most M log[(1+2kappa)/(1-2kappa)]. Weight this by delta_s. The root/conditional chain rule finishes the ideal-density proof. Root estimation using all N arrays introduces no extra independence requirement: the expected risks add, and the conditional estimator uses observed C1, not an estimated root quantile.

Section 9 supplies the final continuous-grid term.

## 6. A nonvacuous D=3072 regime

Take

```
c=192, G=4, m=360, M=1440, D=3072,
N=4000, ns=2000, n=2000,
k0=8, K=32, r*=0.5,
kappa=0.45, a=0.35, L=0.5, J=2**32.
```

There are 1,035,360 candidate edges. The graph-error bound is 4.775210643113885e-6. The following are full-array nats, **not** per-coordinate KL:

| Contribution | Expected joint-KL upper contribution |
|---|---:|
| Unknown root histogram masses | 0.335916020995 |
| Nonlinear-context approximation | 0.077097039474 |
| Context/continuous-strength estimation | 0.185832260667 |
| Empty context cells | 2.0574e-11 |
| Unknown matching | 0.020246855546 |
| Continuous CDF-grid density | 0.000009479709 |
| **Total** | **0.619101656411** |

For every full-root-conditioned scalar-product decoder in the fixed chart,

\[
 D(P\|Q_{\rm product})\ge\sum_bm E F(\theta_b(C_1))
 \ge\frac M2 a^2=88.2.
\]

The lower bound already grants that comparator exact scalar conditional margins and the true root law. It has no force against a learned analysis that absorbs the dependence, an analytic pair model, a global flow, or a copied decoder.

A loose high-probability corollary follows from Markov's inequality: with probability at least 95%, joint KL is at most 12.3821 nats; probability that it fails to beat the 88.2 product floor is at most 0.619101657/88.2 < 0.00702. These are worst-case corollaries, not empirical confidence intervals.

This is a shape-compatible array regime, not evidence for real-image structure. The substantive assumptions are known uniform margins in a public chart, independent histogram roots, a known scalar sufficient context, within-block stationarity, conditional pair independence, and a strong graph signal. Arbitrary learned native analysis, unknown context selection, arbitrary per-pair nonlinearities, and the weak rho=0.01 problem are outside this certificate. In particular the graph bound becomes vacuous at a=0.01 with ns=2000.

Ignoring lower-order count terms and structural failure, balancing MK^-2 and GK/n gives K proportional to (M L^2 n/G)^(1/3), and risk scaling of order M^(1/3)L^(2/3)(G/n)^(2/3), plus root learning. This rate describes sharing under a one-dimensional smooth context, not high-dimensional nonparametric image estimation.

## 7. Observable representation/prediction consequence

For every residual coordinate j with true mate pi(j),

\[
 f_j^*(C,U_{-j})=E[\psi(U_j)\mid C,U_{-j}]
 =\theta_b(C_1)\psi(U_{\pi(j)}).
\]

Use the estimated mate and theta_hat to define f_hat. On a failed graph event, squared function error is at most 8kappa^2 per coordinate. The one-per-pair sum of excess functional prediction errors is bounded by

\[
 R=Mv+(G+Mv)A_n+M\kappa^2e^{-np_{\min}}+8M\kappa^2\delta_s.
\]

The average over all 2M residual coordinates has excess MSE at most R/M. Numerically R=0.111052745449 and R/M=0.000077119962. The zero predictor has MSE one, while its gap above the optimal prediction is E theta_b^2. Hence the learned predictor's mean per-coordinate MSE improvement over zero is at least

\[
 a^2-R/M=0.122422880038.
\]

This tests a function of **observed masked coordinates**, not correlation with inaccessible latent variables. A same-information analytic copy predicts the same function. Retaining all D stochastic coordinates does not prove semantic disentanglement, lossy compression, or downstream usefulness. For Gaussian Z, T(Z) and T(OZ) for orthogonal O have the same observational law, so latent rotation cannot be identified by this density result.

## 8. Exact normalized full-D sampling

Use one full Z~N(0,I_D). Transform each source coordinate to its uniform Phi(Z_j). Generate the root by the learned histogram inverses. For each learned pair retain its first uniform source and transform the second source by the inverse conditional CDF

\[
 F_t(v\mid u)=v+t\psi(u)\frac{\sqrt2}{2\pi}\sin(2\pi v),
 \qquad F'_t(v\mid u)\in[1-2\kappa,1+2\kappa].
\]

Use **generated** C1 in t=theta_hat_b(C1); never feed a real observed root into unconditional generation. Every Gaussian coordinate is used. The triangular map is a measurable bijection apart from irrelevant bin boundaries. Histogram/context boundaries need not be globally smooth; normalization follows by conditional integration, and Jacobian identities hold almost everywhere.

## 9. Compute and precision are not free

Dense graph moments cost ns*G*(2m)^2 = 4,147,200,000 multiply-adds (about 8.29 GFLOPs under two FLOPs per multiply-add), plus cosine evaluation, root counting, graph checks, context fitting, allocation, and I/O. The remaining estimator cost is O(Nc+nD). There are c(k0-1)+GK=1472 free continuous fitted parameters plus a discrete matching table; this is not a 192-dimensional latent-noise model.

A float64 4000x3072 array takes 93.75 MiB. Four full 720x720 score matrices take about 15.82 MiB; they can be processed one at a time. Temporary feature arrays and training/source banks add memory. The executed code is not claimed to attain a minimal streaming-memory bound.

For a precise continuous implementation, linearly interpolate F_t at J=2^b dyadic knots and invert that interpolant. Binary search finds the cell in b conditional-CDF evaluations without storing a J-entry table, then an affine formula locates the continuous output within the cell. The average derivative is evaluated stably using sinc and the cell midpoint rather than subtracting adjacent almost-equal CDF values.

The pair's density error satisfies

\[
 |p_t(u,v)-p_{t,J}(u,v)|\le2\pi\kappa/J,
 \quad |\log(p_t/p_{t,J})|\le\frac{2\pi\kappa}{J(1-2\kappa)}.
\]

Thus the full density-KL increase against any target law is at most M times this uniform log bound. For b=32, sampling costs roughly 1440*32=46,080 conditional-CDF evaluations per array. This may lose to an optimized low-NFE global FM. No inverse-CDF or compilation cost is hidden in a claim of constant-time sampling.

Returning a bisection midpoint would instead define an atomic output law and cannot justify continuous-density KL. Ordinary floating-point output is also atomic; its KL against a continuous target is not the theorem's object. The code implements and checks the continuous piecewise-linear density in float64, recording numerical round trips separately. Parameter rounding by at most eta adds a conservative 2M*eta/(1-2kappa) log-density budget, with a separate root log-probability budget if roots are rounded. No bound on an arbitrary neural optimizer or transcendental-library implementation is claimed.

## 10. Actual checks and interpretation

`run_checks.py --math` passed 20 checks: feature identities, normalization and margins, covariance blindness, entropy-series and direct quadrature agreement, the KL curvature bounds, wrong-matching KL, inverse-count and root bounds, CDF-grid density/round trips, and a 20-dimensional dense full-rank Jacobian. The D=3072 full-source check is separate; a dense 3072x3072 Jacobian was not computed.

Three fresh fixture seeds 1109101, 1109102, 1109103 were executed at N=4000,D=3072. In the positive fixture the four unknown functions are signed, random-phase curves

\[
 \theta_b(c)=s_b[0.35+0.075\sin(2\pi c+\alpha_b)],
\]

with unknown random within-group matchings and unknown root masses. The first root happens to be uniform in the fixture but is still estimated; the other 191 roots have independently randomized masses. The learner receives neither the sine formula nor truth parameters. The fixture lies inside the theorem's Lipschitz class because 2pi*0.075<0.5.

The changed fixture replaces only the intercept 0.35 by zero. Root law, pairs, signs, phases, amplitude, and the per-seed Gaussian banks are held fixed. This preserves conditional nonlinear variation but removes the marginal edge signal. It changes overall dependence strength; no matched-strength claim is made.

All positive fits recovered 1440/1440 true edges. Joint population-KL evaluations were 0.209583477348, 0.211227491032, and 0.211379408991. Their conditional-product floor was 95.874882737277. A fitted constant-context ablation gave joint KL approximately 3.08 nats, so the fitted nonlinear context is materially used. On the changed law every structural gate fell back to the product law, leaving conditional KL 2.028220434893 in each seed and zero masked-feature gain.

These are population evaluations of the **ungridded fitted model** against the ideal analytic fixture, using the scalar entropy series and independent context quadrature; they are not estimates from a reused repair sample. The actual fixture sampler used continuous 2^44 CDF interpolation, with a per-array uniform log-density discrepancy bound of 2.314382117591e-9 relative to the ideal fixture. This is disclosed rather than calling floating-point simulation mathematically exact. The fitted sampler used 2^32 interpolation. Quadrature refinements and source checks are stored in the JSON files. Finite numerical tests do not themselves prove the theorem.

Only a same-weight analytic density copy and the static-context ablation were checked. The independent tuned analytic control, global-FM training, fresh-repair model selection, and the matched-budget comparison in `PROTOCOL.md` have **not** run. No native model or protected image/video data were loaded.
