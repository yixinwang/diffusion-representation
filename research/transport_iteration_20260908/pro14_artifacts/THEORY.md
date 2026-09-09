# Pro14: one nested innovation-response experiment

## Scope

This is a mathematical and implementation review of the native cached-innovation
pilot, not a new native result. Native numbers are the user's provisionally
reported source-matching results, not independently recomputed here. No native
arrays, checkpoints, features, or pixels were read. The executable checks use
fabricated inputs and an independently written surrounding flow, not qalt's
native analysis or training runner. The proposal does not replace Pro13's
separate synthetic dependence work.

## 1. What the frozen block can and cannot express

For a fixed complete prefix H=(C,R_<b), the inspected decoder samples

    R = B_H(Z) = mu(H) + D(H) M(H) g_H(Z),
    M = I + U diag(exp(alpha)-1) U^T,   U^T U = I,
    Z ~ N(0,I_720).

Here g acts coordinatewise. Its integrated-linear derivative has eight equal
input bins in [-4,4], exact identity tails, and slopes in [.1,6.4]. Both D's
scales and M's nontrivial eigenvalues lie in [.5,2]; the explicit shifts mu
lie in [-4,4]. These are inspected architecture constraints, not inferred fits.

Conditionally on H this is an affine independent-components family. It is
not generally a Gaussian copula and need not have Gaussian marginal laws.
A higher rank alone does not turn one affine mixture of independent components
into arbitrary value-dependent nonlinear dependence.

Let V=diag(Var(g_i(Z_i)|H)) and K=M-I=U Lambda U^T. Then

    Cov(R|H) = D [V + K V + V K + K V K] D.

The correction to D V D has columns in span(D U, D V U), hence rank at most 2r,
or 32 for r=16. This is a diagonal-plus-rank bound. It is NOT a bound on the
rank after zeroing a covariance matrix's diagonal. It applies to one block
conditional on its entire prefix; it does not bound unconditional pixel
covariance, covariance after the nonlinear inverse analysis, or covariance
created through the root and earlier blocks.

The graph/Stiefel chart is locally restricted and may become ill-conditioned
near excluded subspaces. But graph charts cover a dense open set of subspaces;
excluded singular leading minors do not by themselves establish a positive
approximation floor. Signed eigenvectors remove orientation ambiguity in an SPD
representation. The Cayley chart similarly has excluded boundary rotations.
Native fitted conditioning, spectrum, and gradients were not inspected, so
attributing the failure specifically to either chart is unsupported.

At alpha=0, the old mixer is I for every frame, so its derivative with respect
to frame parameters vanishes. This is a genuine initialization fact, not evidence
that frame learning caused the native failure. The new zero-output response head
likewise has a short initial gradient gating stage.

## 2. Single architectural change

Keep B_H, its frame U, analysis, exact root, bins, and four block masks unchanged.
Precompose B_H with one small conditional affine coupling F_H in its innovation
coordinates. Within each 720-vector choose 16 anchors at zero-based positions
0,45,...,675 and let the remaining 704 coordinates be followers. These positions
are a fixed source-independent design, not selected on data.

Write the existing Gaussian block as (u,w), with u in R^16 and w in R^704. Retain
the conditioner's existing 16-number global summary h=h(H), rather than running
another conditioner. Let U_f be the 704 follower rows of the same existing frame.
Define

    (a,b) = L2 tanh( L1 [h, tanh(u), tanh(u)^2] + b1 ) + b2,
    m(h,u) = U_f a,
    s(h,u) = log(2) tanh(U_f b),
    F_H(u,w) = (u, m(h,u) + exp(s(h,u)) * w).

L1 is 48->32 and L2 is 32->32. There are exactly 2,624 new parameters per block,
10,496 total. With the user's frozen counts this gives 210,932 residual and
539,120 whole-model parameters. This count must be checked against the actual
native integration before launch. A no-mixer scalar control of width42 has
544,080 parameters by the frozen count algebra. The existing strong width32,
four-layer RQS has 589,712 parameters and is retained, not narrowed.

The zero-initialized L2 makes F the identity. Thus the candidate includes the
old model in exact arithmetic. The standalone fabricated zero-head test also
has identical floating outputs/determinants on its tested bank. This does not
claim universal bitwise equivalence for every possible floating-point input.

### Capacity-matched control

The prefix-only control P has exactly the same active parameters, operations,
frame, and zero initialization, but feeds [h,tanh(h),tanh(h)^2] into the response
head. The candidate I feeds [h,tanh(u),tanh(u)^2]. P gains prefix-head capacity
and shifts/scales before g, but retains conditional independence of the scalar
innovations before the affine mixer. Consequently I-vs-P is more informative
than I-vs-old-M alone about *innovation-dependent* effects. These are different
information routes inside the same generative model, not additional observations
or additional noise. The RQS control already has own-layer masked-value access.

### Inverse, determinant, and density

If e=(u,v)=F_H(u,w),

    F_H^-1(u,v) = (u, (v-m(h,u))*exp(-s(h,u))),
    log det J_F = sum_j s_j(h,u).

The Jacobian is block triangular, with identity on the anchor block and positive
exp(s) on the follower diagonal. Derivatives through h and u belong to off-
diagonal triangular blocks; det formulas do not license detaching those inputs.

Encode R first through B_H^-1, then F_H^-1; decode in the opposite order.
The complete block logdet is the sum of both maps' logdets. The response density
in e-coordinates is the normalized full-dimensional law

    q_H(u,v) = phi_16(u) product_j N(v_j; m_j(h,u), exp(2s_j(h,u))).

No latent variable is marginalized or discarded; the full generator still has
192+4*720=3,072 Gaussian coordinates. No root context is supplied from real data
at unconditional generation. There is no rejection, resampling, clipping, or ODE
solver in this proposal. The inherited scalar has identity tails; the complete
response composition need not have identity tails. The complete conditional
follower response has positive scale in [.5,2].

These are standard affine-coupling/change-of-variables principles, not a claim
to have invented a new universal class of flows. The question is whether this
particular cached, thin response allocates native finite-budget capacity better.
See Dinh et al., Real NVP, https://arxiv.org/abs/1605.08803 and Durkan et al.,
Neural Spline Flows, https://arxiv.org/abs/1906.04032 for the established families.

## 3. An exact approximation decomposition

Let P be the population law of dequantized pixels in the open unit cube. Fix a
candidate's entire fitted parameter vector. Its invertible logit/analysis map
sends X to (C,R_1,...,R_4). Let H_b=(C,R_<b) and pull each true conditional R_b
back through that candidate's B_H^-1 to obtain E_b=(U_b,V_b). Different candidates
induce different pulled-back targets. These are not common coordinates in which
to compare isolated residual NLLs across different learned analyses.

Assume the relevant densities, KL divergences, second moments, and conditional
entropies exist and are finite, and the marginal conditional variances below
are positive. Disintegrations are taken on standard Euclidean spaces. Write
T_b=(h(H_b),U_b), and let mu*_j(T),tau_j(T)^2 be the true conditional mean and
variance of V_j given T.

The chain rule and invariance of KL under each fixed invertible coordinate map
give the exact equality

    KL(P_X || Q_X) = KL(P_C || Q_C) + sum_b epsilon_b,

with

    epsilon_b = anchor_b + missing_context_b + TC_b + shape_b + fit_b,

    anchor_b = E_H KL(P_(U|H) || phi_16),
    missing_context_b = I_P(V ; H | h(H), U),
    TC_b = E_T KL(P_(V|T) || product_j P_(V_j|T)),
    shape_b = E_T sum_j KL(P_(V_j|T) || N(mu*_j,tau_j^2)),

    fit_b = (1/2) E_T sum_j [
        (tau_j^2 + (mu*_j-m_j)^2)/exp(2s_j)
        - 1 + log(exp(2s_j)/tau_j^2)
    ].

Proof. Condition first on H and split U from V. Since q(V|H,U) depends on H
only through h(H), inserting P(V|h(H),U) contributes exactly the conditional
mutual information term. Split that conditional joint law into its marginals,
which contributes conditional total correlation. For each marginal insert its
moment-matched Gaussian. The difference of the two Gaussian cross entropies
depends only on the true marginal's first two moments and gives fit_b. The
root plus conditional chain rule sums these expressions. The outer pixel/logit
and analysis Jacobians cancel in KL because the same candidate map is applied
to both distributions in this argument, not because they may be omitted from
likelihood evaluation.

All terms are nonnegative. The assumptions needed for a small bound are now
explicit: reasonably Gaussian anchors after the fitted base map; little useful
remaining context lost by the 16-summary response; little follower dependence
left after conditioning on those anchors and summary; sufficiently Gaussian
conditional followers; and mean/scale functions approximable and learnable in
the retained rank16 loading/head/bounded-scale family. None is established by
the native covariance statistic or by this fabricated check.

The frame, rank, scalar family, conditioner, and optimizer can all affect these
terms via B_H and the response functions. This is an analysis framework, not an
identification theorem saying one observed failed metric selects one term.

### Finite fitting is not free

For a predeclared complete-model class, if a uniform population/empirical
cross-entropy deviation were bounded by delta and the optimizer achieved
empirical excess eta over that class, then KL of its output is at most
inf_class KL + 2delta + eta. This elementary inequality requires those
assumptions. No small delta or eta has been proved for the 4,000-image,
roughly 539k-parameter pilot, and a wall-time cap or training loss trajectory
does not supply either one. Repair is reused development data.

## 4. Observable bridge and its limits

Let epsilon=KL(P_X||Q_X) in nats per whole image, and
v=min(1,sqrt(epsilon/2)). Pinsker and bounded-observable integration imply:

* Every f(X) in [0,1] has |E_P f-E_Q f| <= v. This includes the existing
  horizontal/vertical grayscale squared-gradient means.
* For two quadrant means a,b in [0,1],
  |Cov_P(a,b)-Cov_Q(a,b)| <= 3v (also bounded trivially by 1/2).
  Bound the ab moment by v and the product of means by 2v.
* The normalized paired pixel energy score differs from its population-correct
  counterpart by at most 2v; this follows by comparing its P-Q expectation and
  its Q-Q expectation separately, since pixel distance/sqrt(D) is at most one.
* If the fixed feature kernel obeys sup_x k(x,x)<=K, its population KID/MMD^2
  is <=4K v^2<=2K epsilon. This follows by bounding the RKHS norm of the
  difference of kernel mean embeddings by 2 sqrt(K) TV.

For the pinned finite Inception evaluated on the compact pixel cube, finite
feature bounds exist mathematically, but no useful K is certified here. These
bounds are population statements, not inequalities for the signed, full-bank
unbiased KID estimator. Negative sample KID values must remain unchanged.

These bridges are intentionally qualified. A small *change* in NLL/D is not a
small *absolute* KL(P||Q), entropy is unknown, and multiplying by 3,072 matters.
For example a sufficient worst-case KL for an absolute gradient error of .0008
is only about 1.28e-6 nats per image. Nothing in the pilot certifies such an
absolute accuracy. Lower complete NLL therefore does not certify the 10%
gradient gate, KID ordering, or energy margin. The native gates remain the
operational falsifiers, not decorations on a likelihood result.

## 5. Why the new head can detect a missed energy mechanism

In the pulled-back coordinates, the conditional Gaussian negative log density
is, apart from a constant,

    sum_j [s_j + .5*(v_j-m_j)^2*exp(-2s_j)].

Writing z_j=(v_j-m_j)exp(-s_j), its derivatives are

    dL/dm_j = -z_j exp(-s_j),   dL/ds_j = 1-z_j^2.

Thus a response direction weighted by a nonlinear anchor feature f(u) can learn
from E[(1-z_j^2) f(u)] even when ordinary Cov(u,v_j)=0. The reused frame projects
these mean/energy scores into its rank16 loading directions; missing directions
remain a genuine restriction.

A fully normalized two-coordinate instance of the actual proposed response head
has U,W independent N(0,1) and

    V = exp(s(U)) W,
    s(u) = log(2)*tanh(tanh(tanh(u)^2)).

The head weights realizing this are explicitly set in checks.py, not fitted.
Cov(U,V)=0 exactly, while Cov(U^2,V^2)>0. The best independent Gaussian with the
correct marginal variances has KL gap

    .5 log E exp(2s(U)) - E s(U) > 0,

by strict Jensen. Numerical quadrature gives about .02260215 nats and energy
covariance .53640147. This only demonstrates a missing nonlinear energy
mechanism. It is not an image-quality result, a win over RQS, or proof the old
fully trainable base cannot re-express this particular law. A generic coupling
with access to U can copy it, and exact-copy ties are retained.

A deliberately retained failure has a fixed identity base, independent anchors,
and two correlated Gaussian followers with rho=.6. Every diagonal-follower
response incurs conditional TC at least -.5 log(1-rho^2)=.22314355 nats. The
trainable full model can change its base, so this is a failure of the fixed-base
approximation assumptions, not an unconditional model-class floor.
