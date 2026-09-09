# Pro10 mathematical audit and next quality calculation

Inspected source commit: e7ee7938a9b146aa5b167a1176e53f99f9ccd4bd.
This is an independent review and standalone local implementation, not a repo
change or a PSC run. Original Pro8 files are not included or overwritten here.
The 83 reported PSC payload hashes and 324 timings were NOT independently
rehashed/recomputed in this review. The supplied completed PSC replication is
separate evidence; local results here do not replace it.

## Combined risk statement

Fix any true gamma in {-0.5,+0.5} and any true j in the public 16-element catalog.
Assume the exact real model, exact orthonormal dictionary, invertible common
chart, independent training arrays, specified within-array conditional
independence, disjoint 32-array root and 256-array head training, and exact ML.
Let pg=exp(-32*192/72), pj=15*rho^256, where
rho=1-(9/16)(1-exp(-2880*0.1^2*0.5^2/(24*1.6))).
Then E_train KL(P||Q_exact)<=96*pj+224*pg=8.249605924796479e-9.
The errors need not be independent: pj bounds head error conditional on a
correct root, and root/head data are disjoint. On root-correct/head-wrong
outcomes KL<=96; on root-wrong outcomes root KL<=128 and conditional KL<=96.
The expectation is over training datasets for each fixed truth, not a mixture
of all trained generators and not a Monte Carlo estimate from 35 recoveries.

The derivative lemma, |w_y|<0.544435, guarantees global increasing Heun maps.
The interval propagation covers all e in [.4,.6] on the median trajectory;
it does not interval-propagate all real Gaussian trajectories. These two
separate statements suffice for the binary-median event KL lower bound.
For N=4,8,16,32,64 mathematical field stages, denote the recorded full-array
lower bound by L_N. E_train KL(P||Q_Heun,N)>=(1-pg-pj)*L_N.
Therefore every declared point has a strictly larger expected ideal-law KL
than the exact sampler. No quality magnitude or perceptual benefit on native
pixels follows. The fields, fixed uniform Heun solver, fitted root/summary,
complete source, chart and training are shared and restricted exactly as in
Pro8. Changing fields, solvers, distillation, or decoder class leaves the
comparison. An exact-copy decoder ties.

Floating QR orthogonality, likelihood rounding, sigmoid saturation, and rounded
output distributions are numerical issues, not silently covered by the
real-arithmetic risk theorem. Small roundtrip error is not a KL certificate.

## Algebraically identical hybrid quantile

Let s=-1 for z<=0, +1 otherwise; p=Phi(-|z|), b=1+s*e.
Set D=b+sqrt(b*b-4*s*e*p). Then
Q_e(z)=-s*Phi^{-1}(2*p/D).
For |z|<=8 use ndtr and ndtri directly on this folded tail probability.
For |z|>8 calculate lp=log_ndtr(-|z|) and
Q_e(z)=-s*ndtri_exp(log(2)+lp-log(b+sqrt(b*b-4*s*e*exp(lp)))).
There is no clipping, source rejection, resampling, or tail truncation.
For the scoped |e|<=.6, the central quantile argument lies between
Phi(-8)/1.6 and .639, avoiding the dangerous subtraction from one.

The inverse similarly uses, with s determined by the sign of r and
p=Phi(-|r|), the probability p*(1+s*e*(1-p)); its tail log is
log_ndtr(-|r|)+log1p(s*e*(1-exp(log_ndtr(-|r|)))).

The supplied implementation explicitly rejects unrepresentable extreme
finite special-function outputs; no arbitrary-float-range promise is made.
Tiny downward steps (maximum 4.72e-16 in the local grid) occur from rounding.
The underlying exact real map remains strictly increasing.

## Fair uniform-Heun simplification

w(0,y;e)=e/sqrt(pi) is state independent, and w(1,y;e)=0.
Cache all time-only k and A=2(1-t)/(s^2*d*sqrt(2*pi)) once at setup.
Inside each sample cache e once, as already permitted by Pro8.
Evaluate interior w=A*e*exp(-(k*y)^2/2)/(1+e*(2Phi(k*y)-1)).
The final step is y += h*w(t,y;e)/2, with no final predictor allocation.
The first stage uses the broadcast constant. Repeated interior times have
DIFFERENT states and their field values cannot be reused.
Report N mathematical/equivalent stages AND N-2 nontrivial field kernels.
No old 'actual N expensive calls' label should be attached to the new code.
This is the same exact real map, so its original interval lower bound remains
valid; floating differences must pass a separate parity gate.

## Full KL without numerical inverse solves

Let H=H_N(z;e), J=dH/dz>0, and
ell=log(p_e(H)*J/phi(z))
   =-(H-z)*(H+z)/2+log1p(e*(2Phi(H)-1))+log(J).
Then
KL(p_e || H_N#phi)=E_{Z~phi}[exp(ell)*ell-expm1(ell)].
This is forward KL, not reverse KL. It follows by changing variables r=H(z)
in the forward KL integral and adding the exactly zero integral of p-q.
The integrand is nonnegative; near zero use its Taylor series
sum_{n>=2}(n-1)*ell^n/n! to avoid cancellation.
Propagate J by differentiating EACH Heun step, not the continuous ODE.
On correct catalog recovery, e is uniform on [.4,.6], so
K_N=(2880/.2)*integral_.4^.6 E_phi[g(ell)] de.

The local 256/512/1024-node z quadratures, with 32-node e quadrature, give
K_4≈.2237247031, K_8≈.03206290586, K_16≈.002146501776,
K_32≈.0001377479519, K_64≈.000008709605496 nats/full array.
Resolution agreement is not validated quadrature. These numbers MUST NOT be
labelled certified full-KL bounds. `local_quality.json` retains all resolutions.
The next computational task is outward-ball/interval integration on the compact
domain, with absolute full-array enclosure width <=2e-8 nats.

## Explicit tail bound for that certificate

For e in [.4,.6], 0<=w<=3/sqrt(pi)<1.7, hence 0<=H(z)-z<=B=1.7.
Using L=.6 and h<=.5, the step Jacobian is between 1-hL-(hL)^2/2
and 1+hL+(hL)^2/2. Summing logarithms gives |log J|<1.1.
Consequently |ell(z)|<=B*|z|+A, with A=3.5.
For g(ell)=exp(ell)*ell-expm1(ell), g(ell)<=exp(|ell|)(|ell|+1)+1.
The joint residual tail outside |z|<=T is thus at most

2880 * [2 exp(A+B^2/2) *
  {B phi(T-B)+(B^2+A+1) Phi(-(T-B))} + 2 Phi(-T)].

At T=12 this expression is approximately 7.15e-18 nats/full array.
Evaluate it outwardly for a formal bound; the current Python value is an
ordinary-float evaluation of a rigorous symbolic bound. The missing part of
the proposed full certificate is compact-domain quadrature enclosure, not
an uncontrolled Gaussian tail cutoff.

## Expected-risk upper bounds for a matched-quality frontier

Conditional-on-recovery K_N alone must not be called unconditional risk.
A conservative failure-event bound is available. For any true/fitted e,f in
[.4,.6], set z=H_f^{-1}(r). Then |z-r|<=1.7, |log J_f|<=1.1,
E_{p_e}|r|<=1.6*sqrt(2/pi), and log(1+e*psi(r))<=log(1.6).
Therefore KL(p_e||H_f#phi) is below
1.7*1.6*sqrt(2/pi)+1.7^2/2+log(1.6)+1.1 < 6.
Thus after obtaining K_N in a validated interval [a_N,b_N],

(1-pg-pj)*a_N <= E_train KL(P||Q_Heun,N)
              <= b_N + 17280*pj + 17408*pg.

The conservative additive upper-bound penalty is approximately 1.485e-6 nats.
Report both conditional and unconditional intervals. In particular, do not
assert that an approximately 8.71e-6 conditional error certifies an expected
1e-5 target when this chosen conservative upper bound crosses that threshold.

## Generalization falsifier, a separate changed-law experiment

Keep the root and scalar conditional marginals but pair the residual Gaussian
innovations: W_(2k-1)=V_(2k-1),
W_(2k)=rho*V_(2k-1)+sqrt(1-rho^2)*V_(2k), rho=.01,
then R_l=Q_e(W_l). No coordinates or randomness are added or discarded.
An unchanged conditional-product decoder has an irreducible conditional KL
floor -m/4*log(1-rho^2)=.07200360024 nats/full array, even with correct root,
summary and perfect scalar marginals. This is Gaussian pair mutual information,
preserved by separate invertible marginal transforms. Compare it to a
same-information exact pair-coupling control, with all estimation/setup/cost
charged. Do not reuse the original independent-head learning theorem for
this changed source. This is a separate next falsifier, NOT a change to the
current sampler-optimization law or its historical positive result.
