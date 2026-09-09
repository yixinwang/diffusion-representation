# Pro8 finite-dictionary learning and locked-Heun comparison

This check assumes the exact specified distribution family: 192 independent root coordinates, 2,880 conditionally independent residual coordinates, one unknown root sign, and one unknown summary index among 16 public functions. It does not assert that realistic images satisfy these assumptions. The comparator fixes the exact canonical Gaussian-reference velocity and a specified Heun grid. Other velocity parameterizations, time changes, exact inverses, and learned or biased fields are outside that restricted comparison.

## Definition and root-sign estimation

Write $\psi(x)=2\Phi(x)-1$ and

$$
p_e(x)=\varphi(x)[1+e\psi(x)],\qquad
F_e(x)=\Phi(x)+e[\Phi(x)^2-\Phi(x)].
$$

The map $Q_e$ is the Gaussian-to-$p_e$ map $F_e^{-1}\circ\Phi$, not the ordinary uniform-input quantile. Explicitly,

$$
Q_e(z)=\Phi^{-1}\!\left(\frac{2\Phi(z)}{1-e+\sqrt{(1-e)^2+4e\Phi(z)}}\right).
$$

Tail stability of an implementation requires its own checks; this note concerns the real-arithmetic map. Under $p_e$, $\psi(X)$ has density $(1+eu)/2$ on $[-1,1]$, so $E_e\psi(X)=e/3$.

For independent $C_i\sim p_\gamma$, $\gamma\in\{-1/2,1/2\}$, the sign of the mean $\psi(C)$ from 32 arrays of 192 independent roots misclassifies with probability bounded by

$$
P_\gamma\le\exp[-32\cdot192/72]=8.71373225\times10^{-38}.
$$

Hoeffding applies to 6,144 independent bounded scalars of range length two and mean magnitude $1/6$. This use of within-array independence is valid because it is an explicit root-model assumption, not because arbitrary pixels are independent.

## Finite summary dictionary

Let $O$ have 16 orthonormal rows in $\mathbb R^{192}$ and define

$$
h_j(C)=2\Phi((OQ_\gamma^{-1}(C))_j)-1,\qquad
 e_j(C)=0.5+0.1h_j(C).
$$

With the correct root sign, the 16 projected variables are independent standard Gaussians, and their transformed $h_j$ values are independent uniforms on $[-1,1]$. Dense rows of $O$ provide dependence on all root coordinates; this is a public finite dictionary, not a learned unrestricted representation. For every pair of distinct indices,

$$
P(|h_j-h_k|\ge1/2)=(1-1/4)^2=9/16.
$$

All $e_j$ lie in $[0.4,0.6]$. For $|e|,|f|\le a=0.6$, the Hellinger affinity obeys

$$
1-\int\sqrt{p_ep_f}
=\frac{(e-f)^2}{2}\int\frac{\varphi(x)\psi(x)^2}
{(\sqrt{1+e\psi(x)}+\sqrt{1+f\psi(x)})^2}\,dx
\ge\frac{(e-f)^2}{24(1+a)}.
$$

The last step uses $E_\varphi\psi^2=1/3$ and denominator bounded by $4(1+a)$. Given a context, the 2,880 conditionally independent responses multiply affinities. Integrating over contexts and using the event above gives an upper bound on each pairwise full-array affinity:

$$
\rho_*=1-\frac9{16}\left[1-\exp\left\{-\frac{2880(0.1)^2(0.5)^2}{24(1.6)}\right\}\right]
=0.9038288789764752.
$$

Use 256 independent arrays, separate from root-sign estimation, for exact maximum likelihood over the 16 candidates. Under the correct sign, Markov's inequality applied to the square root likelihood ratio bounds each wrong candidate's win probability by $\rho_*^{256}$. A union bound gives

$$
P_j\le15\rho_*^{256}=8.59333950\times10^{-11}.
$$

This requires exact likelihood evaluation and maximization, with numerical failures accounted for separately. Conditional independence of the residuals and independent full arrays are essential. Separate samples ensure conditioning on the correct root decision does not alter the distribution used for summary estimation.

## Expected KL accounting

For any $e,f\in[0.4,0.6]$, KL is bounded by chi-square:

$$
\operatorname{KL}(p_e\|p_f)
\le (e-f)^2\int\frac{\varphi\psi^2}{1+f\psi}
\le\frac{(e-f)^2}{1.2}.
$$

Thus any wrong summary incurs conditional residual KL bounded by $2880(0.2)^2/1.2=96$, even if the root sign is also wrong: the two summary-induced coefficients remain in the prescribed interval. A wrong root sign incurs additional root KL bounded by $192/[3(1-0.5)]=128$. On a correct sign and index, the exact generative model equals the truth. Consequently,

$$
E\operatorname{KL}(P\|\widehat Q_{\rm exact})
\le96P_j+(96+128)P_\gamma
=8.24960593\times10^{-9}.
$$

The extra 96 on the root-error event must be retained; writing only $96P_j+128P_\gamma$ would omit its possible residual error. It has negligible numerical impact here but matters logically. The result is a finite-family learning calculation, not a statistical guarantee for an arbitrary learned analysis model.

## Exact Gaussian-reference conditional field

For independent $Z\sim N(0,1)$ and $R\sim p_e$, set $X_t=(1-t)Z+tR$, $s^2=(1-t)^2+t^2$, and $Y_t=X_t/s$. Let

$$
d^2=2(1-t)^2+t^2,\qquad k=t/d.
$$

Gaussian conditioning under the unperturbed endpoints gives the reference path density

$$
p_t(y)=\varphi(y)[1+e(2\Phi(ky)-1)].
$$

Here $k'=2(1-t)/d^3$ and $1+k^2=2s^2/d^2$. Integrating $\partial_tp_t$ from $-\infty$ to $y$ uses

$$
\int_{-\infty}^y u\varphi(u)\varphi(ku)\,du
=-\frac{\varphi(y)\varphi(ky)}{1+k^2}.
$$

The continuity-equation velocity $w=-\partial_tF_t/p_t$ is exactly

$$
\boxed{w(t,y;e)=\frac{2e(1-t)}{s^2d}
\frac{\varphi(ky)}{1+e(2\Phi(ky)-1)}.}
$$

At $t=0$ it is a constant translation field; at $t=1$ it is zero. The claimed formula is correct.

## An analytic global derivative bound below 0.55

Put $z=ky$, $D_e(z)=1+e\psi(z)$, and $A=2e(1-t)/(s^2d)$. Then

$$
|\partial_yw|=Ak\,\varphi(z)
\frac{|zD_e(z)+2e\varphi(z)|}{D_e(z)^2}.
$$

The prefactor satisfies $Ak=2et(1-t)/(s^2d^2)\le0.9$, using $e\le0.6$, $t(1-t)\le1/4$, $s^2\ge1/2$, and $d^2\ge2/3$.

For $z\ge0$, $D_e(z)\ge1$, giving a remaining factor bounded by

$$
\sup_z z\varphi(z)+1.2\sup_z\varphi(z)^2
=\frac1{\sqrt{2\pi\exp(1)}}+\frac{0.6}{\pi}<0.433.
$$

For $z=-u\le0$, $D_e(-u)\ge0.4+1.2\overline\Phi(u)$. The inequality

$$
\frac{\varphi(u)}{0.4+1.2\overline\Phi(u)}\le\frac12
$$

holds for every $u\ge0$: the derivative of $0.4+1.2\overline\Phi(u)-2\varphi(u)$ is $\varphi(u)(2u-1.2)$, so its minimum is at $u=0.6$. Using $\overline\Phi(0.6)\ge1/2-0.6/\sqrt{2\pi}$ and $\exp(-0.18)<0.84$, the minimum exceeds $1-2.4/\sqrt6>0$.

Since the two terms inside the absolute value now have opposite signs, $|a-b|\le\max(a,b)$ for $a,b\ge0$ gives

$$
\varphi(u)\frac{|-uD_e(-u)+2e\varphi(u)|}{D_e(-u)^2}
\le\max\left\{\frac{1}{0.4\sqrt{2\pi\exp(1)}},\,0.3\right\}.
$$

Combining cases establishes the uniform real-arithmetic bound

$$
|\partial_yw|\le\frac{2.25}{\sqrt{2\pi\exp(1)}}<0.544435<0.55.
$$

Thus this claim does not require an interval optimizer. With two or more Heun steps, $h\le1/2$ and the sufficient residual-Lipschitz bound $hL+h^2L^2/2$ is below one even with $L=0.55$. Each numerical scalar map is a globally increasing bijection, making its median the output from source zero. This also justifies a continuous numerical output density in ideal arithmetic.

## Median bias, claimed interval results, and restrictions

Let $H_N(0;e)$ be the endpoint of the fixed Heun rule started at zero and define

$$
d_N(e)=1/2-F_e(H_N(0;e)).
$$

The numerical model assigns probability $1/2$ to $(-\infty,H_N(0;e)]$. If $d_N(e)>0$, true and numerical probabilities on that event differ by $d_N(e)$. Pinsker gives conditional scalar KL at least $2d_N(e)^2$. Conditional product structure then gives joint residual KL at least $2m\inf_{e\in[0.4,0.6]}d_N(e)^2$.

Ordinary scalar numerical checks give minima at $e=0.4$ on the searches performed. They produce the following approximate joint lower-bound expressions for 4/8/16/32/64 actual velocity calls: 0.06222855, 0.01404301, 0.0009457283, 0.0000606899, and 0.00000383489. The proposed interval lower bounds 0.0621, 0.0140, 0.000940, 0.0000593, and 0.00000351 are all smaller and numerically consistent. These numerical searches do **not** prove their uniform minima. The interval code, endpoint handling, elementary-function enclosures, and rounding must be reviewed before treating those five reported constants as certified lower bounds.

If the interval bounds are established, the expected exact-model KL bound above is smaller than each locked-Heun residual lower bound on the correct-learning event. The expected locked-Heun bound can be multiplied by $1-P_j-P_\gamma$, since other events have nonnegative KL. This is a comparison with the **specified exact canonical velocity plus fixed Heun solver**, rather than with every flow-matching learner. The population-optimal velocity for continuous-time regression need not minimize the finite-step output error; another field can compensate solver bias. An exact conditional stochastic copy matches the exact method.

## Uniform summary-error robustness

Assume the root model is correct and $|e(C)-\widehat e(C)|\le\delta$ uniformly, with both coefficients in $[0.4,0.6]$. The exact decoder incurs conditional joint KL bounded by $m\delta^2/1.2$. At the numerical median produced using $\widehat e$,

$$
|F_e(x)-F_{\widehat e}(x)|
=|e-\widehat e|\Phi(x)(1-\Phi(x))\le\delta/4.
$$

If a uniform $d_N(\widehat e)\ge d_{\min}>0$ has been proved, the locked numerical decoder therefore has true joint KL bounded below by

$$
2m[d_{\min}-\delta/4]_+^2.
$$

The proposed upper and lower expressions are correct with this definition of $\delta$. A strict separation follows when

$$
\delta<\frac{d_{\min}}{1/4+1/\sqrt{2.4}}.
$$

A bound on summary error $|h-\widehat h|$ must first be multiplied by 0.1 to obtain this coefficient error. Pointwise error bounds that fail on a set of contexts require additional probability and KL accounting. No error guarantee for an unrestricted learned summary follows from the finite dictionary result.
