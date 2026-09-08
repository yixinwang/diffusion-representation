# Solver options for the unchanged convex density family

This design uses the solver source and mathematical examples. Its additive implementation is `positive_spline_optimizer.py`, checked on fabricated optimization problems. No saved validation observations or fitted outcomes were used to select the new solver, and no study was repeated. The reported 500-step failures concern the registered empirical optimization-gap requirement. They do not establish a density-family failure or a poor population score.

## Objective and a computable smoothness bound

For one shared head, flatten its coefficients into a and write its sparse feature matrix as F. Each row of F is nonnegative, sums to one, and has at most r=2*3^s nonzero entries, with s in {0,1,2}. Let N be the pooled response count. The objective and derivatives are

f(a) = -(1/N) sum_i log((Fa)_i),
g(a) = -(1/N) F^T [(Fa)^(-1)],
H(a) = (1/N) F^T diag((Fa)^(-2)) F.

The feasible set C has one constraint w^T a_row=1 per context-basis row and coefficient limits ell<=a<=U. Every feasible prediction lies in [ell,U]. The objective is convex and smooth on this set, with

L <= ||F||_2^2/(N ell^2)
  <= max_j sum_i F_ij/(N ell^2)
  <= 1/ell^2.

The middle bound follows from ||F||_infinity=1 and ||F||_2^2<=||F||_1||F||_infinity. Column sums give a safe training-data-based constant with one sparse pass. For ell=0.175, the final conservative bound is approximately 32.6531. Computing a tighter spectral bound costs additional work and must be charged. No assumption of independent pooled sites enters these deterministic statements.

With the current b=4 and one-parent sharing, the root head has five coefficients and the conditional head has thirty. Their equality constraints reduce the respective free dimensions to four and twenty-four. This is a small constrained optimization problem even when many observed responses contribute to its objective. Unobserved context directions can make the Hessian singular, so strict or strong convexity cannot be assumed.

## Exact weighted-row Euclidean projection

For a trial row v, the Euclidean projection onto its box and normalization hyperplane has the form

a_j(lambda)=clip(v_j-lambda*w_j, ell, U),
sum_j w_j a_j(lambda)=1.

The multiplier solves a continuous nonincreasing piecewise-linear equation. Its breakpoints are (v_j-U)/w_j and (v_j-ell)/w_j. Sorting the 2(b+1) breakpoints identifies the active interval. Within that interval,

lambda = [sum_active w_j v_j + U sum_upper w_j + ell sum_lower w_j - 1]
         / sum_active w_j^2.

The unique projected row follows after clipping. If an interval has zero slope, it cannot contain a crossing unless its constant already equals one; that case has a constant projected row and can be returned directly. A breakpoint scan costs O(b log b) per row and gives the exact projection in real arithmetic. Bisection is an alternative with an explicit tolerance and iteration charge, but an approximate projection does not inherit the exact accelerated proof without error control. Floating-point code should report row-integral residuals, bound violations, and projection KKT residuals. Normalizing after arbitrary clipping is not this Euclidean projection and can violate the coefficient limits.

## Recommended first candidate: feasible accelerated projected gradients

Keep the density basis, coefficient bounds, training arrays, grouping, uniform initialization, and requested final gap unchanged. Change only the numerical solver. Use a feasible accelerated scheme with a proved global L, not unchecked FISTA extrapolation.

The feasibility qualification matters. Ordinary extrapolation y=a_k+beta(a_k-a_{k-1}) can leave C and make Fa negative. For a b=4 response row, a feasible first coefficient can move from 3.3 to 0.175 while the other four coefficients preserve normalization. Extrapolating this coordinate with beta=0.9 gives -2.6375. A response near that endpoint then has a negative extrapolated density. Positivity of the accepted iterates alone does not justify evaluating the logarithmic objective there.

A one-projection accelerated construction avoids this problem. Initialize x_0=z_0=ones and A_0=0. At each iteration define

alpha = [1+sqrt(1+4 L A_k)]/(2L),
A_next = A_k+alpha,
y = (A_k*x_k+alpha*z_k)/A_next,
g = gradient f(y),
z_next = Projection_C(z_k-alpha*g),
x_next = (A_k*x_k+alpha*z_next)/A_next.

Every gradient point y and both state sequences remain feasible. The identity L*alpha^2=A_next and the projection three-point inequality give, for any optimizer a_star,

A_next [f(x_next)-f(a_star)] + (1/2)||z_next-a_star||^2
<= A_k [f(x_k)-f(a_star)] + (1/2)||z_k-a_star||^2.

To verify this potential, smoothness bounds f(x_next) around y; x_next-y=(alpha/A_next)(z_next-z_k). Convexity replaces the old x_k and the comparison a_star by their supporting planes at y. The projection inequality bounds alpha*<g,z_next-a_star> and cancels the squared step from smoothness. Summation gives

f(x_k)-f_star <= ||x_0-a_star||^2/(2A_k)
              <= 2L||x_0-a_star||^2/(k+1)^2, for k>=1.

This is an objective bound. It is not an O(k^-2) bound on the Frank–Wolfe gap. Acceleration also does not guarantee monotonic raw objectives. A separate incumbent with the smallest observed feasible training objective has a monotonic recorded objective and is no worse than x_k. Record the actual trajectory as well. Restarting after increases is a possible later rule, but arbitrary restarts do not automatically retain the displayed accelerated rate.

This update belongs to the established accelerated similar-triangles family; it introduces no optimization novelty. See [Gasnikov and Nesterov, Universal fast gradient method](https://arxiv.org/abs/1604.05275).

## Retain the existing stopping certificate

For every feasible candidate a, compute the same rowwise bounded-knapsack minimizer

s(a)=argmin_{s in C} <gradient f(a),s>,
G_FW(a)=<gradient f(a),a-s(a)>.

Convexity gives 0<=f(a)-f_star<=G_FW(a). The certificate depends on the objective and feasible set, not on whether Frank–Wolfe generated a. The proposed accelerated solver can compute it at y using the gradient already obtained. If it passes, y is a valid return point. At the resource cap, return the chosen feasible incumbent, recompute its gradient and linear minimizer, and report its own fresh gap. A stale gap from another point is invalid. Optimizer status messages or small steps cannot replace the gap test. See [Jaggi, Revisiting Frank–Wolfe](https://proceedings.mlr.press/v28/jaggi13.html).

A useful limit on the rate argument follows from smoothness. Let d_C be the diameter of C and delta=f(a)-f_star. A step toward s(a) gives

G_FW(a) <= max{2 delta, sqrt(2 L d_C^2 delta)}.

If G_FW<=L d_C^2, optimize the quadratic descent bound over its step fraction to obtain delta>=G_FW^2/(2L d_C^2). Otherwise the full step gives delta>=G_FW-L d_C^2/2>G_FW/2. An O(k^-2) objective bound consequently yields only an O(k^-1) generic gap bound. Standard Frank–Wolfe also has gap guarantees of order 1/k under its corresponding iteration-selection conditions. There is no theorem here that the accelerated method reaches the requested gap faster in this problem. Its practical effect must be measured prospectively.

The current per-head tolerance of 1e-4 can remain unchanged. Joint per-coordinate objective suboptimality is bounded by sum_g |sites_g|*G_g/D. Preserve both the headwise pass flags and this weighted quantity. Ordinary floating-point gap calculations remain numerical certificates; rigorous outward error bounds would require additional implementation work.

## Alternative solvers and costs

| Option | Mathematical support | Work and limitation |
|---|---|---|
| Existing Frank–Wolfe with exact scalar line search | Convexity and its explicit gap; preserves feasibility | Two sparse matrix/vector products for a gradient, another for the search direction, repeated N-vector line-search scans, and rowwise sorting. Boundary vertex mixtures can progress slowly. Existing failures do not prove that this mechanism caused them. |
| Feasible accelerated projection, recommended first candidate | Explicit O(k^-2) objective proof above; same final gap | Two sparse products per gradient, row projections, and the same rowwise linear minimizer for certification. Evaluating f(x_next) for a complete objective trace adds another sparse product. No active-vertex store is required. |
| Plain projected gradient with step 1/L | Feasible monotonic descent and O(1/k) objective convergence | Simpler fallback for verifying projections and derivative code. It provides no acceleration guarantee and still requires the same gap before success. |
| Away-step or pairwise Frank–Wolfe | Feasible steps; stronger rates under additional curvature/error-bound and polytope assumptions | Must store and update active vertices and charge their scans. Strong-convexity-based linear convergence cannot be asserted for a rank-deficient empirical design. |
| Damped constrained Newton or a small dense convex solver | Local curvature can improve practical convergence; independent final FW gap remains valid | The present heads are small enough to form Hessians. Active constraints, singular directions, line searches, and all linear solves must be handled and charged. A damped step may preserve the original objective; adding a fitted regularization penalty changes it and would require a separate estimator definition. |

For N pooled responses, m=R(b+1) coefficients, and at most r local features per observation, each sparse product costs O(Nr); a gradient requires two. A row projection or linear minimizer costs O(R b log b). A dense Hessian can be formed in O(N r^2) work and stored in O(m^2), followed by solves up to O(m^3). These operations are cheap or expensive according to actual head sizes and response counts; their costs cannot be omitted from a solver comparison.

## Prospective verification and accounting

Before any repeated development batch, independently verify the projection against a generic constrained quadratic solver on fabricated rows; finite-difference the objective gradient; check positivity and row constraints at every evaluated accelerated point; and recompute final gaps through the existing linear minimizer. Tiny fabricated response sets can be compared with an independent convex optimizer. Include a singular/unused-context case, a known uniform optimum, and a capped nonconverged case. These checks must not use the previously evaluated 36-cell quality outcomes for solver selection.

Charge feature construction and storage, curvature-bound construction, every gradient and likelihood evaluation, every projection and linear minimization, line-search calls, restart work if introduced, final independent certification, peak memory, and elapsed CPU time. Report iterations as an additional quantity; equal iteration caps do not imply equal computation. Select a compute or gradient-pass budget before rerunning. Keep the same statistical sample counts and do not relabel pooled responses as independent arrays.

A cold start at the existing uniform coefficients gives the clearest prospective comparison. If a run instead continues a frozen Frank–Wolfe checkpoint, charge all preceding 500 steps and checkpoint construction; it is a continuation, not a fresh low-cost fit. Preserve the original results. A successful new empirical optimization certificate would repair the unresolved fitting requirement for this fixed density family. It would not establish a new population learning rate, density-family adequacy, image/video quality, or superiority over a same-information stochastic decoder.
