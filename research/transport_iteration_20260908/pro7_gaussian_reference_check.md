# Gaussian reference path and Heun defect: independent check

This note independently derives the scalar identities discussed in the [Pro7 consultation](https://chatgpt.com/c/6aa09906-7f18-83ea-8420-838be6f8f2f6). All calculations use an ideal standard Gaussian law and real-arithmetic numerical integration. No observations, trained weights, or fitting outcomes enter the calculations.

## Conditional flow-matching target

Let $Z,R$ be independent $N(0,I_d)$ variables, $X_t=(1-t)Z+tR$, and $U=R-Z$. Write

$$
s(t)^2=(1-t)^2+t^2,\qquad a(t)=\frac{2t-1}{s(t)^2}.
$$

The covariance identities are

$$
\operatorname{Var}(X_t)=s^2I_d,\quad
\operatorname{Cov}(U,X_t)=(2t-1)I_d,\quad
\operatorname{Var}(U)=2I_d.
$$

Gaussian conditioning gives

$$
v^*(x,t)=E[U\mid X_t=x]=a(t)x,
\qquad
\operatorname{Var}(U\mid X_t)=\left(2-\frac{(2t-1)^2}{s^2}\right)I_d
=\frac{1}{s^2}I_d.
$$

The last equality follows from $2s^2-(2t-1)^2=1$. The unweighted population squared-loss minimum per coordinate at time $t$ is $1/s(t)^2$. A nonzero training loss is present even when the conditional mean is represented exactly.

Since $s'/s=a$, the exact ODE solution of $\dot x=a(t)x$ with initial value $z$ is $x(t)=s(t)z$. Both endpoints have $s=1$, so the exact endpoint map is the identity.

## Reference coordinates and the loss weight

Define $Y_t=X_t/s(t)$ and

$$
W_t=\frac{U-a(t)X_t}{s(t)}.
$$

Pathwise differentiation gives $\dot Y_t=W_t$. For the Gaussian law, $Y_t\sim N(0,I_d)$ at every time, $E[W_t\mid Y_t]=0$, and

$$
\operatorname{Var}(W_t\mid Y_t)=s(t)^{-4}I_d.
$$

For any fitted field $w_\theta$ define the corresponding original-coordinate field

$$
v_\theta(x,t)=a(t)x+s(t)w_\theta(x/s(t),t).
$$

There is an exact pointwise identity

$$
\|v_\theta(X_t,t)-U\|^2
=s(t)^2\|w_\theta(Y_t,t)-W_t\|^2.
$$

This identity holds for arbitrary endpoint distributions for which the quantities are defined; The zero conditional mean above uses Gaussianity. In the Gaussian case, conditioning gives

$$
E\|v_\theta-U\|^2
=E[s^2\|w_\theta(Y_t,t)\|^2]+d/s^2
$$

at fixed $t$. More generally the original-coordinate excess risk is exactly the $s^2$-weighted reference-coordinate excess risk. Omitting the factor $s^2$ changes the time weighting. It may still have the same unrestricted conditional optimum, but is a different finite-capacity estimation objective. The transformation supplies an exactly zero Gaussian reference field; it does not by itself prove faster learning on non-Gaussian observations.

## Exact finite-Heun endpoint multipliers

For $N$ Heun steps of size $h=1/N$, the scalar endpoint multiplier is

$$
P_N=\prod_{i=0}^{N-1}\left[1+\frac h2\left\{a(i/N)+a((i+1)/N)[1+h a(i/N)]\right\}\right].
$$

All factors are rational. Evaluation with Python integer fractions, without floating arithmetic in the product, gives:

| Actual velocity calls | Heun steps | Endpoint multiplier |
|---:|---:|---:|
| 4 | 2 | 0.9375000000000000 |
| 8 | 4 | 0.9900000000000000 |
| 16 | 8 | 0.9987060970974352 |
| 32 | 16 | 0.9998369634930868 |
| 64 | 32 | 0.9999795807418543 |

The first two exact values are $15/16$ and $99/100$. All rational numerators and denominators are saved in `work/pro7_gaussian_reference_scalars.json`. These agree with the consultation's rounded values. In dimension $d$, the numerical output law for the exact population field is $N(0,P_N^2 I_d)$. This is a solver bias calculation with a known field. It does not quantify the error of a learned field.

## General third-order local defect

Take a sufficiently smooth time-dependent vector field $v(t,x)$ and start both updates at the same $(t,x)$. All derivatives below are evaluated there. Put

$$
A=v_t+D_xv\,v,\qquad
B=v_{tt}+2D_xv_t\,v+D_x^2v[v,v].
$$

Taylor expansion of the Heun predictor evaluation gives

$$
v(t+h,x+hv)=v+hA+\frac{h^2}{2}B+O(h^3).
$$

The numerical update is $x+hv+h^2A/2+h^3B/4+O(h^4)$. Differentiating the ODE along its solution gives

$$
x'''=B+D_xv\,(v_t+D_xv\,v).
$$

Subtracting the exact Taylor expansion from the numerical update yields

$$
H_h(x)-\Phi_{t,t+h}(x)=h^3d(t,x)+O(h^4),
$$

$$
d(t,x)=\frac1{12}\left(v_{tt}+2D_xv_t\,v+D_x^2v[v,v]\right)
-\frac16D_xv\,(v_t+D_xv\,v).
$$

The sign is **numerical minus exact**. For an autonomous linear field $v=Ax$, this reduces to $-A^3x/6$, agreeing with $I+hA+h^2A^2/2-e^{hA}$. For a state-independent time field $v=b(t)$, it reduces to $b''(t)/12$, the trapezoidal-rule local error. For scalar time-linear velocity $v=a(t)x$, the mixed terms cancel and give $d=(a''/12-a^3/6)x$. These checks confirm the displayed local-defect formula. The remainder requires derivatives bounded on the local solution neighborhood; it is not a uniform statement over every unbounded state for arbitrary nonlinear fields.

## The Gaussian endpoint cancels the usual leading global term

A local $O(h^3)$ defect generally produces an $O(h^2)$ endpoint error after $1/h$ steps. That leading coefficient vanishes in this Gaussian example. A more precise expansion also determines the next coefficient.

For the scalar Heun multiplier $m(t,h)$, direct expansion of its logarithm relative to the exact step integral gives

$$
\log m(t,h)-\int_t^{t+h}a(u)\,du
=h^3c_3(t)+h^4c_4(t)+O(h^5),
$$

$$
c_3=\frac{a''}{12}-\frac{a^3}{6},\qquad
c_4=\frac{a'''}{24}-\frac{(a')^2}{8}-\frac{a^2a'}4+\frac{a^4}{8}.
$$

The rational function $a$ is smooth on a neighborhood of $[0,1]$, so these expansions can be summed with uniform remainders. The left-grid Euler–Maclaurin formula gives

$$
\log P_N
=h^2\int_0^1c_3(t)\,dt
+h^3\left[\frac{c_3(0)-c_3(1)}2+\int_0^1c_4(t)\,dt\right]+O(h^4).
$$

Here the exact endpoint log multiplier $\int_0^1a=0$. Since $a(1-t)=-a(t)$, $c_3$ is odd about $1/2$, and its integral vanishes. Using the endpoint symmetries and integrating the derivative terms in $c_4$, the bracket simplifies to

$$
\frac18\int_0^1\{a(t)^4-a'(t)^2\}\,dt.
$$

With $u=2t-1$, $a=2u/(1+u^2)$ and $a'=4(1-u^2)/(1+u^2)^2$, this integral equals

$$
\int_{-1}^{1}\frac{2u^2-1}{(1+u^2)^4}\,du
=-\frac38-\frac{3\pi}{32}.
$$

For example, substituting $u=\tan\theta$ reduces the integrand to $2\cos^4\theta-3\cos^6\theta$ on $[-\pi/4,\pi/4]$, giving the stated value. Consequently,

$$
P_N=1-\left(\frac38+\frac{3\pi}{32}\right)N^{-3}+O(N^{-4}).
$$

The asymptotic constant is approximately $0.6695243113$. The observed third-order endpoint behavior has an explicit symmetry-based explanation and is not a generic order improvement for Heun or for learned nonlinear velocities.
