# Pro12 mathematical certificate

## 1. Fixed law, comparison and conditioning

Let phi and Phi be the standard normal density/CDF, and psi(r)=2Phi(r)-1.
For e in [2/5,3/5],

    p_e(r) = phi(r) [1 + e psi(r)],
    F_e(r) = Phi(r) - e Phi(r)(1-Phi(r)).

The original class has 192 coarse coordinates, an unknown root sign gamma in
{-1/2,1/2}, and an unknown summary index j among 16 known orthonormal rows of
O. The true coarse Gaussian source U has independent standard normal entries.
The residual tilt is e(C)=1/2+(1/10)psi(O_j U); therefore it is uniform on
[2/5,3/5] under the true root. Given C, all 2880 residual coordinates have
density p_e and are independent. A common, exactly invertible Haar/sigmoid
chart maps the complete source to the open RGB32 cube. This is the exact
finite-catalog class of Pro8, not a new data law.

Both generators use the same training procedure and fitted parameters. The
exact comparator uses F_e^{-1}(Phi(z)). The restricted numerical comparator
uses the specified population field and uniform Heun, N in {4,8,16,32,64}.
An exact-copy decoder is allowed to tie outside that restricted solver class.

On the event of correct root AND summary recovery, define K_N as the full-array
KL from truth to Heun. Conditional here refers to that learning event; the
fresh-array covariates and residuals are still integrated over their true law.
Expected training risk below is E_train KL(P || Q_fitted), for each fixed true
catalog member. It is not KL(P || E_train Q_fitted), not an empirical average
over the three timing sources, and not a claim about rounded ML decisions.

## 2. Discrete map and exact discrete Jacobian

For t in [0,1], put a=1-t, s2=a*a+t*t, d=sqrt(2*a*a+t*t), k=t/d. The field is

    w(y,t;e) = 2 e a/(s2*d) * phi(k*y)/(1+e*psi(k*y)).

At t=0, w=e/sqrt(pi), independent of y; at t=1, w=0. Put h=2/N and perform
N/2 predictor/corrector steps:

    v_i = w(y_i,t_i;e)
    y_(i+1) = y_i + h/2 [v_i + w(y_i+h*v_i,t_i+h;e)].

The implemented endpoint shortcuts preserve this exact real map. There are N
mathematical stages but N-2 nontrivial exponential/CDF field kernels.

At an interior stage, with q=k*y,

    w_y = -k*w * [q + 2*e*phi(q)/(1+e*psi(q))].

If d1=w_y at the first state and d2 at the predictor state, a full step has
Jacobian factor 1+h/2[d1+d2(1+h*d1)]. The final factor is 1+h*d1/2. Multiplying
these factors gives J=dH_N/dz. The program differentiates this discrete map and
this discrete Jacobian in its interval derivative recurrences; it does NOT
substitute the continuous flow Jacobian.

## 3. Global positivity, bijectivity and tail bounds

These lemmas are independent of compact-domain quadrature success.
Let lambda=2a/(s2*d). Since d>=sqrt(2)*a and s2>=1/2,
lambda<=2sqrt(2); a=0 is immediate. Consequently

    0 <= w <= 3/sqrt(pi) < B = 1.7,
    0 <= H_N(z;e)-z <= B.

Also lambda*k<=3/2. Setting r=t/a reduces this inequality to

    3r^4-4r^3+r^2-4r+6 >= 0.

For 0<=r<=2 this polynomial is
(r-1)^2(3r^2+2r+2)+4-2r. For r>=2 it is
r^3(3r-4)+(r-2)^2+2. Both are nonnegative on their stated domains; endpoints
follow by continuity.

Since lambda*e*k<=0.9, w_y is bounded using

    q*phi(q)/(1+e*psi(q)) + 2e*phi(q)^2/(1+e*psi(q))^2.

For q>=0 both terms are nonnegative and their sum, multiplied by 0.9, is at most

    0.9 [1/sqrt(2*pi*exp(1)) + 1.2/(2*pi)] < 0.390.

For q<0 the terms have opposite signs, so take the maximum of their absolute
values, not their sum. The first is at most
0.9/[0.4 sqrt(2*pi*exp(1))] < 0.545. For -1<=q<0, the second is bounded by
0.9*1.2/[2*pi*(1-0.6*psi(1))^2] < 0.494. For q<=-1 it is bounded by
0.9*1.2/[2*pi*exp(1)*0.16] < 0.396. Thus |w_y|<=L=0.6 uniformly.
`global_bounds.cpp` outwardly checks every transcendental constant inequality.

A step's Jacobian factor lies in [1-r_h,1+r_h], where
r_h=hL+(hL)^2/2<=0.345. It is therefore positive. The sum of r_h over all steps
is at most 0.69. Hence

    |log J| <= 0.69/(1-0.345) < 1.1.

The bounded displacement and positive Jacobian make H_N a global increasing
bijection of R onto R. Every density used below therefore exists. Compact
interval evaluations separately require every denominator and logarithm
argument to stay positive. No complex continuation or logarithm branch choice
is used; Arb's analytic-callback conditions do not enter this certifier.

## 4. Forward KL identity and nonnegative integrand

At y=H_N(z;e), define

    ell = log[p_e(y) J / phi(z)]
        = -(y-z)(y+z)/2 + log(1+e*psi(y)) + log J,
    G(ell) = exp(ell)*ell - expm1(ell).

Changing variables in forward KL gives E_phi exp(ell)*ell. Normalization gives
E_phi[exp(ell)-1]=0, so

    KL(p_e || H_N#phi) = E_phi G(ell).

G(0)=0 and G'(x)=x exp(x), so G is nonnegative on R. Near zero G has expansion
sum_(k>=2) (k-1)x^k/k!. The program does NOT use a finite unchecked truncation:
it evaluates G's zeroth coefficient with directed exp/expm1 at high precision,
and higher normalized derivatives through G'(x)=x exp(x). This handles
cancellation without replacing forward KL by reverse KL or signed-error
integration. A point interval can have a tiny negative lower endpoint from
rounding; enclosing it is valid and no unsafe positivity clipping is used.

By conditional independence and the uniform tilt law,

    K_N = 14400 * integral_(2/5)^(3/5) integral_R phi(z) G(ell(z,e)) dz de.

The common exact root contributes zero on correct recovery; the common
invertible chart preserves KL.

Write |ell|<=B|z|+A, with B=1.7 and A=3.5. Indeed
B^2/2-log(0.4)+1.1<3.5. Using
G(ell)<=exp(|ell|)(|ell|+1)+1, the full-array tail beyond |z|>T is at most

    2880 * [2 exp(A+B^2/2) *
      {B phi(T-B)+(B^2+A+1) Phi(-(T-B))} + 2 Phi(-T)].

At T=12, its outward upper endpoint is 7.144081089334284e-18. Only this
nonnegative tail upper bound is added; it is not a floating cutoff assumption.

## 5. Validated real two-dimensional integration

On each rectangle [z0,z1] x [e0,e1], the program evaluates univariate interval
jets in z and, separately, in e. A jet coefficient k encloses f^(k)/k! at
EVERY point of that rectangle for f(z,e)=phi(z)G(ell). Addition, multiplication,
division, exp, log and erf use finite exact derivative identities with directed
interval arithmetic. For erf, the recurrence uses erf'(x)=2 exp(-x^2)/sqrt(pi).
These are derivative enclosures, not polynomial approximations to the function
with an omitted Taylor remainder.

For an n-node Gauss-Legendre rule on an interval of width d, the Hermite
interpolation remainder and the norm of the monic Legendre polynomial give

    |integral f - Q_n f| <= c_n * d^(2n+1) * sup |f^(2n)/(2n)!|,
    c_n = (n!)^4 / [(2n+1) ((2n)!)^2].

For completeness, the error polynomial is the square of the monic polynomial
whose roots are the Gauss nodes. Interpolation is exact through degree 2n-1.
Integrating that square gives
(d^(2n+1)*(n!)^4)/[(2n+1)*((2n)!)^2]; the remaining derivative factor is
f^(2n)/(2n)!. Positivity of Gauss weights also provides the tensor bound below.

Let M_z and M_e enclose the corresponding normalized 2n-th partial derivatives
on the rectangle, and let its widths be dz,de. Then the tensor error is at most

    c_n [dz^(2n+1)*de*M_z + dz*de^(2n+1)*M_e].

This follows from I_z I_e - Q_z Q_e = (I_z-Q_z)I_e + Q_z(I_e-Q_e), bounding
both terms uniformly and using positive weights summing to dz. No assumed
mixed-derivative bound and no difference-of-resolutions heuristic is needed.

The primary run uses n=6, order-12 derivatives, 128-bit MPFR endpoint arithmetic,
and an initial 48-by-4 rectangle partition of [-12,12]x[2/5,3/5]. Rectangles
are subdivided until the error allowance is met. The exact mathematical
partition uses rational endpoints; outward arithmetic encloses its endpoints,
widths, centers and evaluation points. The returned bound sums the actual
outward error enclosures, independently of any heuristic splitting choice.

Ordinary-double Newton iterations only PROPOSE Gauss root locations. Interval
Legendre sign changes, bisection and disjoint brackets verify all n roots;
separate checks put them in (-1,1). Weights use the interval formula
2/[(1-x^2)P_n'(x)^2], and positivity is checked. Degree n plus n disjoint root
brackets establishes completeness. Node proposals and adaptive priorities are
not trusted numerical answers.

Every endpoint operation uses MPFR directed rounding, including pi, rational
constants, exp, expm1, erf, erfc, sqrt and log. Final endpoints are converted to
binary64 outward and padded one further ULP before 17-significant-digit decimal
printing, covering decimal serialization. MPFR/GMP, the compiler and the small
interval implementation remain the trusted computational base. The result is
not proof-assistant verified and is not a theorem about machine/library bugs.

The enclosure of the quadrature SUM alone is not an integral certificate. Only
`conditional_KL`, which includes the summed Gauss remainder and tail, is used.
An independent order-8/192-bit/changed-mesh N4 run also encloses the same value,
but neither certificate relies on this agreement.

## 6. Finite learning probabilities

The unchanged Pro8 exact-learning theorem uses disjoint samples of 32 root
arrays and 256 head arrays. For each fixed true gamma and j,

    p_gamma = exp(-32*192/72),
    rho = 1 - (9/16) [1-exp(-2880*(1/10)^2*(1/2)^2/(24*(8/5)))],
    p_j = 15*rho^256.

p_j bounds head error CONDITIONAL ON correct root recovery. No independence
of root/head error events is assumed. Fresh generation arrays are independent
of training. The root Hoeffding argument uses the explicitly independent coarse
coordinates; the head affinity argument uses the explicitly conditional-product
residual law. Those sample counts are not valid for arbitrary dependent pixels.
The exact sampler satisfies

    E_train KL(P || Q_exact) <= 96 p_j + 224 p_gamma
                             <= 8.2496059247963795e-9.

For the rate calculation, psi is uniform on [-1,1] under phi and its tilted
expectation is gamma/3. Hoeffding at separation 1/6 over 32*192 observations
gives p_gamma. For scalar candidates, the squared root-density difference
gives Aff(p_e,p_f)<=1-(e-f)^2/[24*(1+0.6)]. For each wrong catalog row, the two
standard-normal projections are independent; their transformed uniforms differ
by at least 1/2 with probability 9/16. Conditional-product affinities multiply
over 2880 residual coordinates. Averaging over the common root gives affinity
at most rho per array; the likelihood-ratio square-root bound over 256 arrays
and a 15-way union bound give p_j. Root/head sample independence justifies using
this argument conditional on correct root selection.

Finally, chi-square dominates KL and E_phi psi^2=1/3, so
KL(p_e||p_f)<=(e-f)^2/[3*(1-max|f|)]. This is at most 1/30 per residual
coordinate, hence 96 for all residuals. For root signs +/-1/2, the analogous
root bound is 192/(3*0.5)=128. Thus 96 and 128+96=224 bound the two exact-sampler
failure events used above.

Floating QR, finite-precision likelihood selection and output rounding are not
silently included in this exact-real statistical theorem.

## 7. First tighter failure bound: stochastic order

This valid bound is retained in primary receipts and used for root-wrong events.
For any e,f in [2/5,3/5], write r=H_f(z) and v=r-z in [0,B]. Then

    log[p_e(r)/q_f(r)]
      = log(1+e*psi(r)) - r*v + v^2/2 + log J_f(z).

Convexity in v implies -r*v+v^2/2 <= B(B/2-r)_+.
Since F_e<=Phi, p_e stochastically dominates the standard normal. The decreasing
hinge has no larger expectation under p_e. Also
KL(p_e||phi)<=chi-square(p_e||phi)=e^2/3<=0.12. Therefore, uniformly in e,f,N,

    KL(p_e || q_f) <= M
    M = B[(B/2)Phi(B/2)+phi(B/2)] + 0.12 + 1.1
      <= 2.8519519316104041.

The root-wrong full-array risk is at most 128+2880M. This needs neither teacher
information nor assumptions about the wrongly recovered summary's distribution.
Using it for all failures already improves the old 1.484929e-6 correction to
7.058244e-7, but the next argument is much sharper for head errors.

## 8. Final sharper correction, from certified matched KL

For fitted tilt f, write q_f=H_f#phi, L_f=log(p_f/q_f) and
kappa_f=KL(p_f||q_f). For any e,f in [2/5,3/5], p_e/p_f<=3/2 and
KL(p_e||p_f)<=(e-f)^2/[3*(1-0.6)]<=1/30. Consequently

    KL(p_e||q_f)
      = KL(p_e||p_f) + E_(p_e) L_f
      <= 1/30 + (3/2) E_(p_f)(L_f)_+.

The negative part satisfies

    E_(p_f)(-L_f)_+
      <= integral_(q_f>p_f)(q_f-p_f)
      = TV(p_f,q_f) <= sqrt(kappa_f/2),

using log u<=u-1 and Pinsker's inequality. For this use of Pinsker, apply
log-sum/data processing to the set {p_f>q_f}; Bernoulli KL is at least twice
the squared probability difference because its second derivative in the first
probability is 1/[x(1-x)]>=4 and its value and first derivative vanish at equality.
Thus the displayed TV bound follows directly, with no additional model assumption.
Thus
E_(p_f)(L_f)_+ <= kappa_f + sqrt(kappa_f/2).

On a correct-root/wrong-head training event, condition on the selected catalog
index k. For a fresh array O_k U is still standard normal, so the fitted f(C)
is uniform on [2/5,3/5], even though the training event depended on observed
training outcomes. This uses independence of fresh data from training, NOT
independence of learning-error events. It does not need the true index, teacher
outputs, or independence between the true and fitted tilts. Hence

    E_C kappa_(f(C)) = K_N/2880.

Jensen then gives the wrong-head full-array bound

    W_N(K_N) = 96 + (3/2)K_N + (3/2)sqrt(1440 K_N).

For a certified correct-recovery enclosure [a_N,b_N], the final rigorous
expected-training-risk enclosure is

    (1-p_gamma-p_j) a_N
       <= E_train KL(P||Q_Heun,N)
       <= b_N + p_j W_N(b_N) + p_gamma(128+2880M).

The expression W_N is increasing, so using b_N is valid. All its numerical
arithmetic is evaluated outward at 192 bits by `risk_bounds.cpp`. For N64 the
learning penalty is at most 8.2640427995153425e-9, rather than 1.484929e-6.
The expected KL upper endpoint is 8.7180979953948377e-6, safely below 1e-5.

## 9. Scope of rounding and of superiority

The verifier encloses its own roundoff, elementary/special-function evaluation,
quadrature remainder, Gaussian tail and statistical correction. It certifies
the underlying exact-real maps with exact catalog learning. It does NOT make
rounded floating-point output distributions continuous. In particular, literal
finite-float outputs define a discrete law; continuous-truth forward KL to
that discrete law is not the finite KL certified here. No unproved bound is
added for library quantiles, sigmoid saturation, QR or likelihood-score rounding.
Saved numerical parity remains a separate implementation audit.

Upper bounds admit target quality; lower bounds exclude it; a straddling
interval remains unresolved. The fixed grid is not all Heun step counts,
fields, solvers or decoders. No eligible member of this grid means no finite
measured eligible comparator, not infinite speedup. Exact-copy ties, native
negative results, and the absence of a traced-memory advantage remain intact.
