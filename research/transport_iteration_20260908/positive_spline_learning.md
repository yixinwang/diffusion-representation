# A finite learning guarantee for convex conditional spline densities

This note gives a learned, normalized, full-dimensional generative construction under explicit conditional smoothness and dependence assumptions. It does not establish those assumptions for realistic images or videos, or prove superiority over an unrestricted stochastic latent model. The construction came from the [fresh Pro4 design consultation](https://chatgpt.com/c/6aa0633a-e1f8-83e9-812d-73bdd9749c78) and received a separate algebraic check. It makes no novelty claim for spline bases, convex likelihood fitting, conditional quantile transports, or conditional-density learning rates.

The implemented primitive and wrapper must be checked against the hypotheses below. A neural spline model with a different objective or feasible class does not inherit this theorem.

## Model and observations

An observation is an entire independent image, video clip, or other array $X=(X_1,\ldots,X_D)\in(0,1)^D$. Training contains $n\ge2$ independent identically distributed arrays. Coordinates within an array may be arbitrarily dependent. They are not counted as independent samples.

Fix a directed acyclic context graph in coordinate order, and partition its coordinates into $G$ groups. Coordinate $j$ has an observed context $C_j$ containing no more than $s$ preceding coordinates. Its **full conditional given all predecessors** is a shared group density:

$$
 p(x_j\mid x_{<j})=p_{g(j)}(x_j\mid C_j).
$$

This equality is the structural assumption. Specifying only the marginal conditional given $C_j$ is insufficient. Graphs, groups, and any observation preprocessing are fixed independently of the $n$ fitting arrays. Group losses average over the group's sites within each image, then average equally over images. The population target uses identical weights. Every group has a fixed positive site count in this statement.

For every group, assume its conditional density $p(t\mid c)$ is twice continuously differentiable on the unit cube, integrates to one in $t$, satisfies $0<m\le p\le M$, and has joint context-response Hessian operator norm bounded by $H$. Constants $m,M,H,s$ are fixed as $D$ varies. There is no Gaussianity assumption. There is also no guarantee here that realistic image or video contexts have $s\le2$, adequate density lower bounds, or controlled derivatives.

## Convex density family

Choose an integer $b\ge1$. For a group with $s_g\le s$ context coordinates, use tensor-product degree-two open-uniform B-splines $B_v(c)$, with $b$ intervals per coordinate and $(b+2)^{s_g}$ basis functions. These nonnegative functions form a partition of unity and reproduce coordinates at their Greville points $g_v$.

Use linear response hats $H_k(t)$ at knots $t_k=k/b$, $k=0,\ldots,b$. Their integral weights are $w_0=w_b=1/(2b)$ and $w_k=1/b$ otherwise. Define

$$
 q_a(t\mid c)=\sum_v B_v(c)\sum_{k=0}^b a_{vk}H_k(t),
 \qquad \ell\le a_{vk}\le U,
 \qquad \sum_k w_k a_{vk}=1.
$$

Here $\ell=m/2$ and $U=2M$. These linear constraints define a nonempty compact convex set: the constant coefficients $a_{vk}=1$ are feasible. Each fitted density is normalized and lies between $\ell$ and $U$. Negative log likelihood is convex in the coefficients.

Fit each group using the stated image-averaged negative log likelihood. Let $\tau\ge0$ bound its empirical suboptimality. An exact Frank-Wolfe linear-minimization gap provides such an upper bound. Approximate solves and numerical errors must be included in the reported gap; a small gradient alone does not provide it.

## Finite learning theorem

Define the maximum coefficient count, approximation constant, loss range, variance constant, and entropy bound by

$$
\begin{aligned}
 P_b&=(b+2)^s(b+1), & A_s&=H(9s+1)/2+MH/6,\\
 B&=\log(U/\ell), & V&=2U^2/\ell^2,\\
 K&=8V+8B/3, & L_n&=P_b\log\!\left(1+8(U-\ell)n/\ell\right)+\log(G/\delta),
\end{aligned}
$$

where $0<\delta<1$ is the total failure probability. If $b^2\ge H/6$, then with probability at least $1-\delta$ over the training arrays, every group's fitted density satisfies

$$
 \mathbb E_{\text{target group sites}}\mathrm{KL}
       \bigl(p(\cdot\mid C)\,\|\,\widehat q(\cdot\mid C)\bigr)
 \le \frac{A_s^2}{\ell b^4}+\frac{K L_n}{n}+2\tau+\frac1n.
$$

The expectation averages fresh target images and their within-group contexts. The event holds jointly for all groups; groups may be dependent. Under the target graph factorization, the full joint KL divided by $D$ is the site-count-weighted average of these group risks and obeys the same upper bound.

For fixed constants, $b$ of order $(n/\log n)^{1/(s+5)}$, and optimization error $\tau$ of the resulting order or smaller, the bound is

$$
 \mathrm{KL}(P\|\widehat Q)/D
 =O\!\left((\log n/n)^{4/(s+5)}\right).
$$

This is a bound derived from observations, approximation, and optimization. It does not assume that the learned conditional errors are already small. The finite formula retains potentially large constants, the restriction on $b$, and the number of independent arrays.

## Proof

**Approximation.** At each Greville point set

$$
 Z_v=\sum_k w_k p(t_k\mid g_v),\qquad
 a_{vk}=p(t_k\mid g_v)/Z_v.
$$

The trapezoid error bound is $|Z_v-1|\le H/(12b^2)$. The assumption on $b$ gives $1/2\le Z_v\le3/2$, so these coefficients lie in the required box and have exactly normalized rows. This is an analytical witness; fitting never queries the true density.

Taylor expansion at $(c,t)$ has canceling first-order terms because Greville and response-knot averages reproduce coordinates. Active context Greville points differ by no more than $3/b$ in each coordinate; active response knots differ by no more than $1/b$. The unnormalized interpolation error is bounded by $H(9s+1)/(2b^2)$. Normalizing its coefficients adds no more than $MH/(6b^2)$. The uniform error is bounded by $A_s/b^2$; the inequality $\mathrm{KL}(p\|q)\le\int(p-q)^2/q$ and $q\ge\ell$ give the approximation term.

**Population curvature.** Let $q_*$ minimize population negative log likelihood in the feasible family. For a feasible $q$, write $f=\log(q_*/q)$ and $\mu=\mathbb E f$. Strong convexity of $-\log$ on $[\ell,U]$, together with first-order optimality of $q_*$, gives

$$
 \mu\ge\frac{\mathbb E(q-q_*)^2}{2U^2},\qquad
 \mathbb E f^2\le V\mu,\qquad |f|\le B.
$$

For each image, replace $f$ by its average over the group's sites. Jensen's inequality bounds the squared average by the average squared loss. The same population curvature argument holds under those weights. The displayed variance and range bounds hold for the image-level average despite arbitrary dependence within an image.

**Finite net and concentration.** Choose a coefficient-sup-norm net of radius $\ell/(4n)$ with its representatives inside the feasible set. Partitioning the ambient box and retaining a feasible representative from every nonempty cell gives cardinality bounded by

$$
 \log|\mathcal N|\le P_b\log\!\left(1+8(U-\ell)n/\ell\right).
$$

An unnormalized ambient grid is not a substitute for this feasible net. Partition-of-unity interpolation transfers coefficient radius to density radius. Log-loss distances are bounded by $1/(4n)$, pointwise and after averaging.

For a feasible net density, centered image losses have range bounded by $2B$ and variance bounded by $V\mu$. Bernstein's inequality gives

$$
 \Pr\{P_n f\le\mu/2\}\le e^{-n\mu/K},
$$

where $P_n$ averages the $n$ independent images. Union over the groups and their nets shows that every net point with $\mu>K L_n/n$ has $P_n f>\mu/2$.

Empirical $\tau$-suboptimality implies $P_n\log(q_*/\widehat q)\le\tau$. Its nearest feasible net point has empirical excess bounded by $\tau+1/(4n)$. On the simultaneous event, its population excess is bounded by $\max\{K L_n/n,2\tau+1/(2n)\}$. Transferring back to $\widehat q$ adds $1/(4n)$, yielding the stated conservative bound $K L_n/n+2\tau+1/n$. Add the approximation term. This proves the theorem.

## Full-dimensional sampling and cost

Use exactly $D$ independent standard Gaussian coordinates. Transform each to a uniform using the Gaussian CDF, then apply the fitted conditional inverse CDF in graph order. Finally return all $D$ response coordinates. No coordinate is discarded, and no VAE or uncounted decoder noise is introduced.

Response density is piecewise linear, so its CDF is piecewise quadratic and its inverse is obtainable by interval search and a scalar quadratic solve. Positivity gives a unique inverse. Context B-splines are continuously differentiable, and response density is continuous, so the conditional quantile transport is continuously differentiable on the open domain. Its triangular Jacobian supplies the exact normalized density.

There are $O(GP_b)$ stored coefficients. At a context point, no more than $3^{s_g}$ quadratic tensor-product basis functions are active. Sampling cost and sequential depth depend on exploiting this locality and on the context graph. A chain still requires $D$ sequential conditional steps. Convex fitting and its verified stopping rule supply no automatic training-time advantage over latent diffusion or flow matching.

## A strict separation from a restrictive independent model

Let $D=2J$ and make $J$ independent pairs with joint density

$$
 p_\rho(x,y)=1+\rho\cos(2\pi x)\cos(2\pi y),\qquad 0<|\rho|<1.
$$

Both marginals are uniform. The conditional is smooth, non-Gaussian, and nonlinear, with $m=1-|\rho|$, $M=1+|\rho|$, and joint conditional Hessian norm bounded by $H=4\pi^2|\rho|$. A one-parent graph and two shared groups, roots and children, satisfy the learning assumptions. Its conditional CDF is

$$
 F(y\mid x)=y+\frac{\rho\cos(2\pi x)}{2\pi}\sin(2\pi y),
$$

which is strictly increasing and yields a nonlinear Gaussian-input transport.

For any model with all $D$ coordinates independent, the per-coordinate KL is bounded below by

$$
 \mathrm{KL}(P\|Q_{\rm independent})/D
 \ge \frac12\mathrm{KL}(p_\rho\|1)
 \ge \frac{\rho^2}{16(1+|\rho|)}.
$$

The first inequality follows by separating the pair mutual information from nonnegative marginal KL terms. For the second, set $z=\rho\cos(2\pi x)\cos(2\pi y)$. The function $(1+z)\log(1+z)-z$ has second derivative at least $1/(1+|\rho|)$, its linear term integrates to zero, and $\int z^2=\rho^2/4$.

The learned spline bound eventually falls below this positive floor. That establishes a strict asymptotic log-score separation from the specified independent family. A model retaining these pairs, or a stochastic decoder using the same conditional model, avoids the floor. This comparison is not a theorem against latent diffusion, latent flow matching, or strong pair-aware baselines.

## Optional transport stability and its additional restrictions

Suppose the learned conditional quantiles have a nonnegative coordinate-influence matrix $A$, with $A_{ji}=0$ outside the preceding-coordinate graph, and $\|A\|_2\le\gamma<1$. Here $\|\cdot\|_2$ is the spectral operator norm. Spectral radius is insufficient. Same-uniform triangular coupling, the conditional lower density $\ell$, and Pinsker's inequality give

$$
 \frac{W_2(P,\widehat Q)}{\sqrt D}
 \le \frac{\sqrt{\mathrm{KL}(P\|\widehat Q)/D}}
               {\sqrt2\,\ell(1-\gamma)}.
$$

In detail, the same-context quantile squared error is bounded by conditional KL divided by $2\ell^2$. The coordinate error recursion is $\|X-Y\|_2\le\|e\|_2+\gamma\|X-Y\|_2$; square, average, and sum conditional KL by the chain rule. Normalized energy-score excess is bounded by twice this normalized Wasserstein distance.

Adjacent context-coefficient constraints

$$
 |a_{v+e_i,k}-a_{vk}|\le\ell A_{ji}/(2b)
$$

suffice for the influence condition: differentiating quadratic B-splines contributes a multiplier bounded by $2b$, integrating the resulting density derivative bounds the CDF derivative, and division by $q\ge\ell$ bounds the quantile derivative.

These constraints change the approximation class. The unconstrained theorem's approximation term does not automatically survive them. One sufficient target margin is the following. If $K_i$ bounds $|\partial p/\partial c_i|$, put $d_b=H/(12b^2)$. The normalized Greville witness is feasible for the adjacent constraints whenever

$$
 \frac{K_i}{1-d_b}+\frac{MH}{6b(1-d_b)^2}\le\ell A_{ji}/2.
$$

This follows by subtracting adjacent normalized coefficients, using Greville spacing bounded by $1/b$ and normalizer differences bounded by $2d_b$. A strict density-derivative margin makes this hold for sufficiently large $b$. A target quantile margin alone does not prove it. The independent-pair example above is a log-score example and is not asserted to meet these additional constrained-approximation conditions.

## Scope and remaining obligations

- If the selected contexts omit informative predecessors, a conditional-mutual-information floor remains. The finite fitting bound cannot remove it.
- Independent arrays, matching within-image weights, a fixed model class, and a verified optimization gap are necessary for the stated inference. Reusing fitting images to choose the graph or chart requires a new argument.
- The Wasserstein bound uses the modeled coordinate metric. An arbitrary nonlinear synthesis chart preserves log-density ratios but requires its Lipschitz factor to transfer Wasserstein or energy bounds to pixels.
- A stochastic latent generator using the same coarse variables, conditional tables, and remaining Gaussian decoder coordinates produces exactly the same joint law with the same information and cost. This exact-copy comparator ties the construction.
- The note establishes no VAE representation comparison, realistic image/video quality improvement, or cost advantage over a strong latent flow/diffusion system. Those remain separate empirical and theoretical questions.

Conditional-density learning is an established subject. For broader statistical context, see [Bilodeau et al., Minimax Rates for Conditional Density Estimation via Empirical Entropy](https://arxiv.org/pdf/2109.10461). The role of this construction is a checkable approximation, fitting, normalization, and failure analysis for the project, without a publication-novelty claim.
