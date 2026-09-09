# Independent Pro10 forward-KL calculation

This note and `pro10_forward_kl_check.py/.json` are independently adapted mathematics and ordinary float64 quadrature. The portable checker resolves the adjacent frozen Pro8 reference and accepts an optional new --output JSON path, refusing overwrite; it otherwise prints without writing. The original scratch result is preserved separately as pro10_forward_kl_check_original.json. No fits, observations, timing study or jobs were run. The independent implementation uses the unchanged Pro8 reference only for parity checks, after verifying its SHA25638cb7d9b79111b99e3af961aa28cdf2ca6f5a2b427245434e2cea407c7b6c73b. Numerical convergence is not an interval certificate or a rigorous quality-eligibility upper bound.

## Direction and the nonnegative integrand

Let p_e(y)=phi(y)[1+e(2Phi(y)-1)], e in[0.4,0.6], and let H be an increasing onto differentiable numerical Heun map with J(z)=H'(z)>0. Its output density is q(H(z))=phi(z)/J(z). Thus

ell(z)=log[p_e(H(z))J(z)/phi(z)]

is log(p_e/q), evaluated at a point drawn from q. Changing variables y=H(z) in the *forward* divergence gives

KL(p_e || q)=E_phi[exp(ell) ell].

Since H is onto, E_phi exp(ell)=integral p_e=1. Subtracting this normalization identity yields the proposed exact identity

KL(p_e || q)=E_phi[exp(ell) ell-expm1(ell)].

This is forward KL, not reverse KL. Reverse KL would be E_phi[-ell]. The normalizing subtraction is essential to obtain a nonnegative and numerically stable integrand without subtracting large positive and negative quadrature contributions. Define F(u)=exp(u)u-expm1(u). It satisfies F(0)=0, F'(u)=u exp(u), so F(u)>=0 for every realu. Nearzero,

F(u)=sum_{n>=2}(n-1)u^n/n! = u^2/2+u^3/3+u^4/8+u^5/30+... .

The checker uses the degree12 polynomial for |u|<0.02 and the explicit exponential expression elsewhere. This is ordinary stable evaluation, not interval-controlled evaluation. It also computes ell's normal quadratic difference as -(H-z)(H+z)/2, avoiding subtraction of two near-equal squares.

## Exact derivative of the numerical sampler

Put a=1-t, s2=a^2+t^2, d=sqrt(2a^2+t^2), k=t/d, u=ky, B=1+e(2Phi(u)-1). The canonical Gaussian-reference field and its derivative are

w(y,t,e)=[2ea/(s2 d)] phi(u)/B,

w_y(y,t,e)=-k w(y,t,e)[u+2e phi(u)/B].

For one Heun step of widthh, define v=w(y,t), A=w_y(y,t), predictor p=y+h v, and B1=w_y(p,t+h). The discrete derivative multiplier is

D_step=1+(h/2)[A+B1(1+hA)].

Update y <- y+(h/2)[w(y,t)+w(p,t+h)] and log J <- log J+log1p((h/2)[A+B1(1+hA)]). This differentiates the actual finite numerical map, not the continuous ODE's divergence integral. There are n=NFE/2 steps of width1/n; “4” in the results below means four velocity evaluations and two Heun steps.

The independent endpoint implementation matches the pinned reference to at most1.78e-15 on a grid of sources[-9,9] and tilts0.4/0.5/0.6. A separate central finite-difference comparison to the reference gives Jacobian errors at most3.43e-10. These are fabricated scalar checks, not a proof based on a grid.

## Three-resolution quadrature

Use probabilists' Gauss-Hermite nodes for E_phi, with weights divided bysqrt(2pi), and Gauss-Legendre nodes transformed to e=0.5+0.1v, weights divided bytwo. The latter computes the uniform average over[0.4,0.6], not an unnormalized integral of width0.2. Multiply by2,880 for the conditionally independent residual coordinates. Under the correct root/dictionary index, their common e has exactly this marginal distribution, so the conditional product KL adds even though coordinates share e.

| Actual field calls | Joint forward KL, 512x128 quadrature |
|---|---:|
|4|0.22372470305395012|
|8|0.032062905859440424|
|16|0.0021465017757271786|
|32|0.00013774795185687397|
|64|0.000008709605496461353|

The128x32,256x64 and512x128 results agree to approximately3.3e-14 or better in absolute joint KL. They agree with every rounded Pro10 value supplied. Quadrature normalization residuals are of order1e-17. None of these observations bounds the unobserved quadrature error or floating arithmetic rigorously.

## Global growth bounds and their actual implication

The elementary global derivative bound |w_y|<0.55 was independently derived in `research/transport_iteration_20260908/pro8_finite_dictionary_audit.md`, in the section “An analytic global derivative bound below0.55.” It does not rely on the finite grid above. Since h<=1/2, the step residual derivative is bounded by b_h=0.55h+(0.55h)^2/2<=0.3128125<1. Hence each step is a globally increasing bijection and H is onto.

Uniformly in y,t,e, B>=0.4, s2>=1/2 and a/d<=1/sqrt2 give

0<=w<=3/sqrt(pi)<1.7.

The positive Heun weighted increments therefore imply 0<=H(z)-z<=1.7. For the Jacobian, n b_h<=0.625625 and b_h<=0.3128125. Thus

|log J| <= n[-log(1-b_h)] <= 0.625625/(1-0.3128125) <1.1.

Using |log(1+e(2Phi(H)-1))|<=-log0.4 and the difference of squares gives

|ell(z)| <=1.7|z|+1.7^2/2-log0.4+1.1 <1.7|z|+3.5.

So the proposed tail-envelope implication is valid. With A=1.7,b=3.5, the nonnegative integrand is bounded by exp(A|z|+b)(A|z|+b+1)+1. For T>=0, a valid analytic bound on its Gaussian tail integral is

2 exp(b+A^2/2)[A phi(T-A)+(A^2+b+1) barPhi(T-A)] +2 barPhi(T).

Multiply by2,880 for the joint residual KL tail. This proves integrability and offers a tail bound for a future rigorously truncated integral. It does not bound the central quadrature error, the quadrature discretization, or outward rounding. Gauss-Hermite convergence plus this envelope alone is not a certified upper bound.

## Wrong-learning contribution: a rigorous conservative bound

The proposed17280/17408 coefficients can be justified without a quadrature upper bound. For *any* true e and fitted f in[0.4,0.6], let q_f be the finite-Heun density and write z=H_f^(-1)(y). The previous global bounds imply |z-y|<=1.7 and |log J_f(z)|<=1.1. Therefore

log[p_e(y)/q_f(y)] <=1.7|y|+1.7^2/2+log1.6+1.1.

The odd tilt cancels in the absolute first moment: E_{p_e}|Y|=sqrt(2/pi). It follows uniformly that

KL(p_e || q_f) <=1.7 sqrt(2/pi)+1.7^2/2+log1.6+1.1 =4.3714073826... <6.

Thus arbitrary wrong-summary residual KL is at most2,880*6=17,280. A wrong root sign contributes at most128 root KL: the per-coordinate chi-square bound for gamma=+/-0.5 is (gamma-gamma')^2/[3(1-|gamma'|)] =2/3, and there are192 roots. Adding residual risk gives17,408 on a wrong-root event.

Let p_gamma bound root-sign failure, and p_j bound index failure conditional on correct root. Conditional on all training randomness, the same uniform bounds apply on a fresh array. Wrong-learning events contribute at most

17280 p_j +17408 p_gamma.

Using the previous finite-family bounds p_gamma<=exp(-32*192/72) and p_j<=15 rho^256, rho=1-(9/16)[1-exp(-2880*0.1^2*0.5^2/(24*1.6))], this additive contribution is at most1.4849290665e-6 (displayed arithmetic is floating; the symbolic bound is rigorous).

If R_oracle is the exact mathematical forward-KL average for the correct learned root/index and the locked Heun solver, total expected learned-Heun risk is at most R_oracle+17280p_j+17408p_gamma. The quadrature above estimates R_oracle but does NOT rigorously upper-bound it. Therefore combining the table with this rigorous correction does not produce a certified numerical upper bound. A finite exact-model learning guarantee remains separate from this solver-risk calculation, and no claim about arbitrary learned FM fields, other solvers, or native image/video quality follows.
