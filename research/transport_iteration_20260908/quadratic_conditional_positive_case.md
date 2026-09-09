# An explicit finite-sample positive case with quadratic conditional transport

This is a deliberately specified one-parameter distribution class with observed contexts and known scalar features. It gives a finite, non-Gaussian, nonlinear positive example against a stated independence restriction. It does not establish superiority over a same-information conditional latent model, learned image representations, or general flow matching.

## Model, information, and exact generation

Fix $\rho=0.65$, $h(c)=\cos(2\pi c)$, $g(r)=2r-1$, and an unknown $\theta\in[-\rho,\rho]$. A single pair has density

$$
p_\theta(c,r)=1+\theta h(c)g(r),\qquad (c,r)\in[0,1]^2.
$$

Both marginals are uniform; the density is between $1-\rho$ and $1+\rho$. The learner is given the observed coordinate designated as context and the fixed feature functions and parameter interval. Every comparator receives those same observations, features, and bounds. The true parameter and source innovations are unavailable to fitting. There is no learned-summary assumption hidden in the construction.

For $a=\theta h(c)$, the conditional CDF and a stable inverse are

$$
F_\theta(r\mid c)=r+a(r^2-r),\qquad
F_\theta^{-1}(u\mid c)=\frac{2u}{1-a+\sqrt{(1-u)(1-a)^2+u(1+a)^2}}.
$$

The inverse follows by rationalizing the quadratic root and is valid also at $a=0$ without division by $a$. The discriminant is a convex combination of positive squares, each bounded below by $(1-\rho)^2$, avoiding subtraction of nearly equal quadratic terms. The denominator is positive. Conditional derivative $1+a(2r-1)$ lies in $[1-\rho,1+\rho]$, so the map is strictly increasing and onto $[0,1]$.

Given exactly two independent standard Gaussian source coordinates, set $C=\Phi(Z_0)$ and $R=F_\theta^{-1}(\Phi(Z_1)\mid C)$. This is a full-dimensional invertible triangular transport on the open cube. Its dependence on the observed context is nonlinear for $\theta\ne0$; the output is non-Gaussian. Encoding uses $Z_0=\Phi^{-1}(C)$ and $Z_1=\Phi^{-1}(F_\theta(R\mid C))$. The forward log determinant is

$$
\log\varphi(Z_0)+\log\varphi(Z_1)-\log p_\theta(R\mid C).
$$

Generation requires fixed counts of normal-CDF, cosine, arithmetic, and square-root evaluations per pair. There is no iterative CDF inverse or ODE integration. These are arithmetic-operation statements about the prescribed class, not measured superiority over an optimized competitor.

## An observation-only estimator and exact variance

From $n$ independent pairs, define

$$
T_i=6h(C_i)g(R_i),\qquad
\widehat\theta=\operatorname{clip}\left(\frac1n\sum_iT_i,[-\rho,\rho]\right).
$$

Under the uniform reference measure, $E h=E g=0$, $E h^2=1/2$, $E g^2=1/3$, and $E[h^3g^3]=0$. Consequently,

$$
E_\theta[h(C)g(R)]=\theta/6,\qquad
E_\theta[T_i^2]=6,\qquad
\operatorname{Var}_\theta(T_i)=6-\theta^2.
$$

Clipping is metric projection onto an interval containing the truth and cannot increase squared error. Thus

$$
E_\theta(\widehat\theta-\theta)^2\le\frac{6-\theta^2}{n}.
$$

The unclipped estimate is unbiased; clipping need not preserve unbiasedness. Fitting is one streaming sum and costs $O(n)$ operations with constant additional memory for this pair model.

## Finite KL constants

Write $X=h(C)g(R)$ under the uniform reference measure, and define $\psi(t)=E_0[(1+tX)\log(1+tX)]$. The distribution of $X$ is symmetric, and

$$
\psi''(t)=E_0\frac{X^2}{1+tX}
=E_0\frac{X^2}{1-t^2X^2}.
$$

For $|t|\le\rho$, this implies

$$
\frac16\le\psi''(t)\le\frac{1}{6(1-\rho^2)}.
$$

The KL is the associated Bregman divergence,

$$
\operatorname{KL}(p_\theta\|p_\eta)
=\psi(\theta)-\psi(\eta)-\psi'(\eta)(\theta-\eta).
$$

Taylor's integral remainder gives the two-sided bound

$$
\frac{(\theta-\eta)^2}{12}
\le\operatorname{KL}(p_\theta\|p_\eta)
\le\frac{(\theta-\eta)^2}{12(1-\rho^2)}.
$$

Hence the fitted population KL satisfies

$$
E\operatorname{KL}(p_\theta\|p_{\widehat\theta})
\le\frac{6-\theta^2}{12(1-\rho^2)n}.
$$

For a high-probability version, put $V=6-\theta^2$, $B=6+\rho$, and $\ell=\log(2/\alpha)$. Each centered $T_i-\theta$ is bounded in absolute value by $B$. Bernstein's inequality gives probability at least $1-\alpha$ that

$$
|\widehat\theta-\theta|\le b_n(V):=
\frac{B\ell}{3n}+\sqrt{\left(\frac{B\ell}{3n}\right)^2+\frac{2V\ell}{n}}.
$$

This is the positive solution of $n b^2=2\ell(V+Bb/3)$. It is a theoretical bound at each fixed true parameter; using $V=6$ gives a parameter-uniform version. The high-probability KL upper bound is $b_n(V)^2/[12(1-\rho^2)]$. The trivial parameter-distance limit $2\rho$ may also be imposed. No test data or fitting of a population KL is required for these analytic statements.

## The explicit restricted comparator and positive regimes

Consider all product laws $q_C(c)q_R(r)$ that prohibit context-dependent residuals, even though their fitting information is otherwise identical. Since both true marginals are uniform, the optimal such law is the uniform product and

$$
\inf_{q_Cq_R}\operatorname{KL}(p_\theta\|q_Cq_R)
=\operatorname{KL}(p_\theta\|1)\ge\theta^2/12.
$$

The equality follows by adding marginal KL terms to the true-marginal product gap. At $\theta=\rho=0.65$, the baseline population KL floor is $0.0352083333$ nats per pair. Explicit finite bounds are:

| Independent observed pairs | Expected fitted KL upper bound | 95% fitted KL upper bound | Independence floor |
| ---: | ---: | ---: | ---: |
| 256 | 0.0031438830 | 0.0271968127 | 0.0352083333 |
| 4000 | 0.0002012085 | 0.0015455250 | 0.0352083333 |

The high-probability column uses the exact $V=6-\theta^2$ in the preceding theorem. At 256 pairs it is below the restricted comparator's floor by about 22.8%; at 4000 pairs by about 95.6%. Expected bounds are smaller still. These compare population log-score error with an irreducible independence error. They are not empirical measurements or bounds for unrestricted latent diffusion. The same result holds at $\theta=-0.65$. Near zero dependence there is no uniform strict improvement over the independence model; the baseline already becomes exact at zero.

## Whole-array observations and dependent sites

Two explicit dimension extensions keep assumptions visible.

**Independent-pair arrays.** Each observed array contains $m$ independent pairs with the same unknown $\theta$, so $D=2m$. Across arrays, observations are independent. Define $T_i$ as the within-array average of the $m$ pair statistics. Its variance is exactly $(6-\theta^2)/m$ under this specified model; apply the preceding Bernstein argument to the $n$ array averages, using the same conservative bound $B$. The joint fitted KL is $m$ times the single-pair KL. Expected total-array KL is bounded by $(6-\theta^2)/[12(1-\rho^2)n]$, while the restricted coarse-block/residual-block independent model has floor $m\theta^2/12$. Here even arbitrary dependence within each comparator block does not help: both true block marginals are product uniforms. That comparator still forbids any dependence between the blocks.

**One observed context shared by $m$ residuals.** Draw $C\sim U(0,1)$ and then conditionally independent $R_j$ from the stated conditional law. The dimension is $D=m+1$. Again use the within-array average $T_i=m^{-1}\sum_j6h(C_i)g(R_{ij})$. Conditional on $C$,

$$
E[T_i\mid C]=2\theta h(C)^2,\qquad
\operatorname{Var}(T_i\mid C)=\{12h(C)^2-4\theta^2h(C)^4\}/m.
$$

Using $E h^4=3/8$ gives

$$
V_m=\operatorname{Var}(T_i)=\frac{6-\tfrac32\theta^2}{m}+\frac{\theta^2}{2}.
$$

The nonvanishing term is the shared-context contribution. Treating all $nm$ residual pairs as independent would omit it. Apply clipping and Bernstein to the $n$ independent array averages with variance $V_m$, then multiply the single-pair KL bound by $m$. The expected joint KL bound is $mV_m/[12(1-\rho^2)n]$. For the fully factorized comparator $q_C\prod_jq_j$, the floor is $m\theta^2/12$. A stronger comparator $q_Cq_{R_1,\ldots,R_m}$ can model induced dependence among residuals, and its floor is instead mutual information $I(C;R_1,\ldots,R_m)$; the $m\theta^2/12$ claim must not be assigned to it.

Both extensions fit in $O(nD)$ time and generate in $O(D)$ operations from exactly $D$ standard Gaussian source coordinates. Bounds use whole arrays as units. If conditional independence, shared parameter, uniform context, fixed feature, or designated-context assumptions fail, these explicit guarantees do not automatically survive.

## Provenance and limits

The pair copula CDF is

$$
\mathcal C_\theta(u,v)=uv+\theta\frac{\sin(2\pi u)}{2\pi}(v^2-v).
$$

It belongs to the established separable perturbation class $uv+f(u)g(v)$ associated with generalized Farlie–Gumbel–Morgenstern copulas. This class is explicitly discussed in the primary research article [Jung, Kim, and Kim, 2008](https://www.tandfonline.com/doi/full/10.1080/03610910701711091), which identifies the Rodríguez-Lallena–Úbeda-Flores family. [Amblard and Girard's bivariate FGM extension](https://arxiv.org/abs/1103.5921) supplies further primary evidence that such function-based generalizations are established. No novelty claim is made for the copula, quadratic inversion, moment method, or concentration argument.

A stochastic conditional latent decoder with the same $\widehat\theta$, designated context, inverse formula, and full Gaussian source produces exactly the same law with the same mathematical operation count. It must tie. General conditional flow matching can also represent this distribution; the independence floor does not apply to it. The example demonstrates nonvacuous finite learning bounds for one known nonlinear conditional family. It does not solve learned analysis, general images or videos, VAE representation comparisons, or strict quality-and-efficiency dominance over strong latent models.
