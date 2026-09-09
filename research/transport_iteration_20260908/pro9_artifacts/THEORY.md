# Pro9: cached conditional global innovations — derivation and scope

Status: a proposed construction and mathematical review at repository commit
`bb024437a761c096375a042239f77262724dca14`. No native model was fitted in this
review. The reference module has fabricated-input CPU tests, not GPU or image
quality evidence. Statements of exactness below concern real arithmetic.

## 1. Complete normalized model

Let x be an open-cube dequantized RGB image with D=3072 coordinates. Use the
same outer logit and its full Jacobian as the frozen native protocol. Let A be
the retained invertible analysis on logits. Its output is a coarse C in R^192
and a losslessly packed residual R in R^(45*8*8). Partition R into four sets
I_b={(channel,row,column): (channel+row+column) mod 4=b}, each of size 720.
The prefix H_b=(C,R_{I_0},...,R_{I_(b-1)}) never contains the current block.

The already implemented `GlobalInnovationFlow` supplies an exact learned root
T_C with 192 Gaussian inputs. This proposal replaces only its residual module:

    C = T_C(Z_C)
    E_b = g_b(Z_b ; h_b(H_b))
    R_b = mu_b(H_b) + D_b(H_b) M_b(H_b) E_b
    logits = A^{-1}(C,R),       x = sigmoid(logits).

All 192+4*720 Gaussian coordinates remain in the model. There are no auxiliary
latent draws, discarded coordinates, VAE, rejection decisions, observation-valued
sampling contexts, or sample corrections. The prior remains N(0,I_3072).

For each block, D_b=diag(exp(ell_b)), |ell_bi|<=log 2, and

    M_b = I + U_b diag(expm1(alpha_b)) U_b^T,
    U_b^T U_b=I_16,                 |alpha_bj|<=log 2.

The mean is bounded by mu=4*tanh(raw_mean) in this first experiment. Bounds on
means, scales and eigenvalues are capacity restrictions, not native-data facts.

The inverse and determinant are

    M_b^{-1} = I + U_b diag(expm1(-alpha_b)) U_b^T,
    log det M_b = sum_j alpha_bj.

M is positive definite with eigenvalues in [1/2,2]. Applying M or its inverse
requires two matrix multiplications through rank 16, not a 720-by-720 matrix.
Since each H_b is fixed during the block transform, the full residual Jacobian
is block triangular. Each block's forward log determinant is

    sum_i log g'_bi(Z_bi;h_bi) + sum_i ell_bi + sum_j alpha_bj.

Adding all blocks, the root determinant, A^{-1}'s determinant, and the sigmoid
Jacobian gives the complete density. This is not a likelihood assigned to a
finite-step Heun sampler.

### Orthonormal frame parameterization

The reference uses U=Q(W) O(v), where Q(W) is the reduced QR frame of [I;W]
with positive R diagonal, and O=(I-S)^(-1)(I+S) is a Cayley rotation with skew
S parameterized by its strictly upper-triangular entries. [I;W] has full rank;
I-S is invertible for every finite real skew S. The internal rotation matters:
a subspace alone does not specify eigenvectors when alpha entries differ.
The graph chart excludes subspaces with a singular leading minor; the Cayley
chart excludes rotations having eigenvalue -1. No claim of coverage of all
frames is made. A target frame must be representable or pay approximation error.

QR and the small solve occur once per block per training transform, never per
example. Frozen inference caches U. Cache creation, stored parameters, duplicate
frames and validity checks are charged. Training and checkpoint loading invalidate
frame caches. Arbitrary weight mutation during cached evaluation is forbidden.

## 2. Cached conditional features

Each block computes its context exactly once from the unchanged prefix:

* Mask all current/future residual coordinates inside the conditioner.
* Concatenate visible residuals, the fixed visibility mask, the generated/observed
  coarse tensor, and two fixed spatial coordinates.
* Apply two local 3x3 convolutional feature layers (width 32).
* Four learned queries attend to all 64 spatial tokens. Four values per query
  give 16 global summary scalars.
* Combine the active coordinate's local feature, a 12-dimensional channel
  embedding, and the shared 16-vector; output only that block's active heads.
* A separate small head maps the 16-vector to the block's 16 log eigenvalues.

Local features and prefix inputs are NOT claimed to be 16-dimensional. The
cache is recomputed after completing a block, not reused across changed prefixes.
Likelihood encoding conditions on actual preceding residual coordinates, not
previously whitened Gaussian coordinates. No current-block value may affect a
head or matrix parameter.

## 3. Direct real-line scalar transform

The proposed head is NOT the nine-height density on the normal-probability
scale. It is a monotone piecewise-quadratic map of real Gaussian coordinates.
Set K=8, B=4, Delta=2B/K=1, and fixed abscissae x_j=-B+j*Delta. Derivative
heights are

    h_0=h_K=1,
    h_j=m+(1-m)(K-1) softmax(a)_j,  j=1,...,K-1,  m=0.1.

Thus all derivative heights are positive, sum of interior heights is K-1,
and the integral over [-B,B] is 2B. For K=8 every derivative lies in [0.1,6.4].
There are seven shape logits, plus a mean and a log scale: nine outputs per
active coordinate. The separate rank head adds 16 outputs per block.

Define y_0=-B and y_(j+1)=y_j+Delta*(h_j+h_(j+1))/2. Inside bin j, with
0<=t=x-x_j<=Delta and beta_j=(h_(j+1)-h_j)/Delta,

    g(x)=y_j+h_j*t+beta_j*t^2/2,
    g'(x)=h_j+beta_j*t.

For |x|>=B set g(x)=x. The endpoint derivatives equal one, so g is a C^1
bijection of the real line. It is generally not C^2; log-density derivatives
at internal knots are interpreted piecewise/almost everywhere.

For inverse input y in [y_j,y_(j+1)], d=y-y_j,

    t = 2*d / [h_j + sqrt(h_j^2+2*beta_j*d)],
    x = x_j+t.

The discriminant equals (h_j+beta_j*t)^2 and is strictly positive. This formula
has the correct linear-bin limit without division by beta or cancellation of
nearly equal roots. Forward log derivative is log(g'); inverse is -log(g').
Implicit inverse derivatives satisfy

    d g^{-1}(y)/dy = 1/g'(x),
    d g^{-1}(y)/d theta = -(partial_theta g)(x)/g'(x).

The reference tests autograd against finite differences in both directions.
It does not evaluate Phi, Phi^{-1}, or a probability clipped away from 0/1.
Inactive tail quadratics receive a finite scratch value; the actual tail input
is returned unchanged. Invalid numerical discriminants are flagged and rejected,
not accepted via clamping. Float32 is used for flow arithmetic; float64 is a
reference check. No finite-precision implementation is globally exact over all
representable numbers; nonfinite log densities or outputs fail the run.

This head sacrifices movable knots and flexible innovation tails. Its possible
speed gain is an implementation hypothesis. It does not repair dependence by
itself. Heavy or strongly asymmetric innovation tails remain an approximation
risk to be reported, not a reason to truncate Gaussian source draws.

### A scalar approximation bound

Suppose a true scalar map g_* and a fitted g are increasing real-line
bijections, both with derivative >=m>0, and g_*' is L-Lipschitz. Assume
uniform errors ||g-g_*||_infinity<=e_g and ||g'-g_*'||_infinity<=e_d.
For P=g_*#N(0,1), Q=g#N(0,1), put delta=e_g/m. Then

    KL(P||Q) <= sqrt(2/pi)*delta + delta^2/2 + (e_d+L*delta)/m.

Proof: at X=g_*(Z), let V=g^{-1}(X). The inverse Lipschitz property gives
|V-Z|<=delta. The log-density ratio is
(V^2-Z^2)/2 + log g'(V) - log g_*'(Z).
Bound its expectation using E|Z|=sqrt(2/pi) and Lipschitzness of log on
[m,infinity). Integrability follows from these assumptions. The bound includes
all tails only when the stated uniform errors hold on the whole real line.
For a target with different tails, these assumptions must be justified or the
uncontrolled scalar KL retained. K=8 does not imply small approximation error.

## 4. Exact decomposition of the native structural problem

Fix all fitted functions and the analysis chart. Let P be the true distribution
on complete images. Under its induced variables define the demixed residual

    V_b=M_b(H_b)^(-1) D_b(H_b)^(-1) [R_b-mu_b(H_b)].

Let h_bi=h_bi(H_b) be the actual features available to scalar head i. Suppose
regular conditional densities exist, the transformations are measurable
bijections, all model conditionals are positive, and the displayed KL terms
are finite. Then the model density Q satisfies the exact identity

 KL(P_X||Q_X) = KL(P_C||q_C)
   + sum_b E_H TC(P_(V_b|H))
   + sum_(b,i) I(V_bi ; H_b | h_bi)
   + sum_(b,i) E_(h_bi) KL(P_(V_bi|h_bi) || q_bi(.|h_bi)).       (1)

Here TC(P_(V|H=h))=KL(P_(V|h)||product_i P_(V_i|h)).

Proof: use KL invariance under the complete chart and the root/block chain
rule. For each block change variables conditionally from R_b to V_b. Add and
subtract sum_i log p(V_bi|H_b), producing conditional total correlation. Then
add and subtract log p(V_bi|h_bi) in each marginal term. Since h_bi is a function
of H_b, the first difference is conditional mutual information and the second
is the displayed scalar KL. These are exact identities, not pixel independence
assumptions. Jointly learned A changes all the induced terms; separate NLL
pieces in different charts are not individually comparable population KLs.

Interpretation: the mixer and learned analysis can reduce the dependence term;
the cached representation can reduce conditional information loss; scalar heads
reduce marginal density error; the root addresses its own distribution. Merely
making a sampler invertible guarantees none of those errors is small.

### A strict structural separation, without restricting an FM solver

At a fixed analysis A and fixed block partition, a scalar-only decoder has
conditionally independent block outputs given the entire H_b. Even arbitrarily
powerful scalar heads cannot improve its optimum below

    sum_b E_H TC(P_(R_b|H)).

Conditional location/scale changes do not remove this floor. For bounded scalar
features f_i,f_j in [0,1], Pinsker and the product-of-marginals distribution give

    TC(P_(R_b|H=h)) >= 2 Cov(f_i(R_bi),f_j(R_bj)|h)^2.          (2)

Indeed, the expectation of f_i*f_j differs by that covariance, so TV is at
least its absolute value; KL is at least twice TV squared. Marginalization
cannot increase KL. This lower bound is for the scalar factorization, NOT for
ordinary global FM, RQS couplings, or an exact-copy decoder.

A nonlinear non-Gaussian positive family is obtained with an unknown non-Gaussian
root, unknown nonlinear mu/ell/alpha functions of local and global prefix
features, an unknown representable spread-out U, and conditionally independent
non-Gaussian E_i=g_i(Z_i;h_i). Generate R=mu+D M E and then apply an unknown
nonlinear invertible synthesis to logits. Full-dimensional light-tailed
innovations, conditional heteroscedasticity and long-range dependence coexist.
No globally log-concave image distribution is assumed.

For example, take a rank-one M=I+(exp(alpha)-1)uu^T, and identical symmetric
non-Gaussian innovations of variance v>0. At a fixed prefix, for i!=j,

    Cov(R_i,R_j | H) = exp(ell_i+ell_j)*v*(exp(2*alpha)-1)*u_i*u_j.

This is nonzero for a nonzero alpha and suitable distant coordinates. Bounded
truncation functions witness dependence as well (finite second moments suffice
to approach the raw covariance); they are mathematical test functions, not
clipping operations in the generator. A scalar-only decoder pays positive TC.
The proposed demixing removes TC for this family when the functions, frame,
root, and scalar laws are representable. All these components are unknown and
must be learned. The current native images are NOT asserted to belong to this
family. Rank, chart, context and scalar-tail mismatch remain terms in (1).

## 5. Learning and observation-quality bridge

Let ell_theta(X)=-log q_theta(X) be the loss per COMPLETE independent image.
For n iid training images, assume a uniform population/empirical deviation
G_n >= sup_theta |P ell_theta-P_n ell_theta|, finite entropy/risk, and a learner
whose empirical risk is within eta_opt of the class infimum. Then

    KL(P||Q_hat) <= inf_theta KL(P||Q_theta) + 2 G_n + eta_opt.  (3)

Proof: empirical optimality between two applications of the deviation bound;
subtract H(P). Equation (1) decomposes the approximation term of (3).
This statement does not supply small G_n or a certified eta_opt for the neural
Adam implementation. Pixel/site pooling does not change n from 4000 to
4000*2880. The earlier convex head optimizer certificate is not transferable.

On the image cube with d(x,y)=||x-y||_2/sqrt(D)<=1, the population energy score
obeys the repository's stronger squared-TV bridge

    0 <= S(Q;P)-S(P;P) <= TV(P,Q)^2 <= min(1, KL(P||Q)/2).     (4)

For bounded quadrant means a,b in [0,1],

    |Cov_P(a,b)-Cov_Q(a,b)| <= 3 TV(P,Q).

Thus a finite joint distribution error controls both a proper observation-space
score and bounded dependence measurements. The right side uses JOINT KL, not
per-coordinate KL. The bound does not certify KID/FID, PRDC, human quality, or
speed. A bounded continuous feature kernel could provide an additional MMD
bound, but no useful numeric constant is established for Inception here.

## 6. Training and attribution

The main arm uses the same 360-second nominal whole-fit budget as competitors,
with absolute phase boundaries at 90,180,270,360 seconds: Gaussian-analysis
warm start, exact-root fit, decoder fit, then joint analysis/decoder full-model NLL with root weights fixed. Preparation
costs consume the appropriate phase budget; crossing-update overruns are charged.
The frozen-analysis mixer arm uses the last 180 seconds for decoder-only fitting.
The joint arm must invalidate fit-code caches, clear the persistent analysis
freeze flag, re-enable analysis/root gradients, and re-encode every training
minibatch. No covariance/perceptual score or repair label enters generator loss.

A Gaussian-trained frozen A is a whole generative model already. Adding another
prior can overfit its residual error or undo useful regularization. Joint NLL is
proposed to learn a chart suited to the structured innovation law, not to force
a cheap head onto a permanently fixed Gaussian chart. It may still hurt sample
quality and must pass the separately frozen native metrics.

A semantic representation advantage does not follow from invertibility or small
likelihood. Equal-dimensional, fixed-readout probes are separate tests. Frozen-A
arms have literally the same A code; their different decoder scores cannot be
credited as an A-representation improvement.

## 7. Closest primary sources and novelty limits

The construction combines established ingredients. No priority or new general
normalizing-flow theorem is claimed.

* Lu and Huang, Woodbury Transformations for Deep Generative Flows, NeurIPS 2020:
  https://arxiv.org/abs/2002.12229 — directly relevant efficient global low-rank
  interactions and determinant/inverse identities.
* Mueller et al., Neural Importance Sampling, 2019, section 4.2:
  https://arxiv.org/html/1808.03856v5 — piecewise-quadratic maps obtained by
  integrating positive piecewise-linear functions.
* Durkan et al., Neural Spline Flows, NeurIPS 2019:
  https://arxiv.org/abs/1906.04032 — analytic rational-quadratic spline inverses.
* Yu, Derpanis and Brubaker, Wavelet Flow:
  https://arxiv.org/abs/2010.13821 — multiscale conditional generative factorization.
* Guth et al., Conditionally Strongly Log-Concave Generative Models, ICML 2023:
  https://proceedings.mlr.press/v202/guth23a.html — structural conditional learning
  and sampling theory beyond globally log-concave laws. Its assumptions are not
  established for this native experiment.
* Zhai et al., Normalizing Flows are Capable Generative Models (TarFlow), ICML 2025:
  https://arxiv.org/abs/2412.06329 — end-to-end autoregressive transformer pixel
  flows. Its augmentation/denoising/guidance recipe cannot be inherited for free,
  nor can postprocessed samples automatically inherit the raw flow's density.

An unrestricted stochastic latent decoder can copy this model, its law, and its
cost exactly. No theorem here separates this model from such a copy or from
arbitrary global FM. The contribution of this review is the specific repair,
explicit structural attribution, implementable head and a falsification plan.
