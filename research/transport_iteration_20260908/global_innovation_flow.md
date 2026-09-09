# Exact coarse and residual innovation flow

**Proposal.** Replace the coarse numerical FM map with four invertible globally attending coupling layers. Retain the learned analysis and the conditional residual coupling flow. This change gives the complete generator an explicit inverse and density in real arithmetic. It supplies no generation-quality, runtime, or representation advantage by itself.

Let y in R^D denote an image in the common logit coordinates. Let A(y)=(c,r) be the invertible learned analysis, including its fixed residual packing, with c in R^d and r in R^(D-d). Let T_C map R^d onto itself through the coarse coupling layers. For each fixed c, let T_R(.;c) map R^(D-d) onto itself through the conditional residual layers. All scales are positive and each scalar rational-quadratic spline has positive derivative and real-line tails. Sampling uses one full standard Gaussian vector z=(z_c,z_r):

    c = T_C(z_c),   r = T_R(z_r;c),   y = A^{-1}(c,r).

The source blocks are independent. Both decoders retain every coordinate. The coarse decoder receives a constant-zero compatibility channel; the weights on that channel are counted. No new randomness, clipping or ODE solver enters either transformation.

**Complete-density proposition.** The displayed map is a bijection of R^D. Its inverse is z_c=T_C^{-1}(c), z_r=T_R^{-1}(r;c), where (c,r)=A(y). Its normalized density is

    q_Y(y) = phi_D(z) |det D A(y)|
             |det D T_C^{-1}(c)| |det D_r T_R^{-1}(r;c)|.

Here phi_D is the D-dimensional standard Gaussian density and each determinant is evaluated at the argument displayed. The Jacobian of (z_c,z_r) -> (c,r) is block triangular: c depends only on z_c. The diagonal blocks are the coarse and fixed-context residual Jacobians. They are invertible by scalar monotonicity and reverse coupling order. Multiplying determinants and applying change of variables proves the formula and integral one. Applying the common sigmoid afterward defines a density on the open pixel cube, with its separate logit Jacobian included when evaluating pixel density.

This proposition removes the coarse finite-Heun qualification from the complete density. The implementation's finite-precision inverse still requires numerical checks. It does not establish useful global conditioning constants: a conditioner whose shift depends on unbounded inputs can have an unbounded derivative. Complete density and complete source retention therefore leave approximation, learning, quality and efficiency unresolved.

Native defaults use D=3072, d=192, four coarse and four residual layers, and width 32. They contain 255,952 analysis parameters, 72,236 coarse parameters and 261,524 residual parameters, totaling 589,712. The 1,152 coarse zero-context input weights are included. The code directly constructs the retained analysis modules and creates no unused FM heads. Three fabricated tests check nonidentity round trips, a dense 16-dimensional Jacobian, the complete density identity, finite gradients, unchanged source draws, exact copied outputs and saved freezing state.

This is a standard triangular composition related to Wavelet Flow, conditional normalizing flows and rational-quadratic spline couplings. Its scientific purpose is to test whether replacing repeated field evaluations with explicit conditional transports can improve measured quality and cost. The known exact stochastic latent copy still ties it. The retained coarse representation is the same analysis representation until a separately specified training objective changes it; no improved semantic representation has been shown.
