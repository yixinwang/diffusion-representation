# Conditional CDF observation-only development validation

Status: frozen experimental specification awaiting execution.
Root review and a committed source revision precede the first execution.

This study checks the construction in
`research/transport_iteration_20260908/conditional_learnability.md`. It is a
synthetic density-learning experiment. It makes no comparison against real
image/video models or trained latent diffusion. No real-data import, archive
extraction, or test split access occurs.

## Frozen configuration

Use theta 0.65, dimensions 8 and 32, independent training-vector counts 256,
1024 and 4096, response/context bins `ceil(n**0.25)` (4, 6 and 8), and seeds
7101, 7102 and 7103. Combine every configuration with the two worlds below: 36 cells.
Every cell has 4096 independently generated evaluation vectors and 256 fresh
Gaussian sampling inputs. Derive three separate NumPy SeedSequence streams
from `[seed, world_id, dimension, n]`, in train/evaluation/sampling order.
Neither training samples nor evaluation samples are nested across sample sizes.
Each full vector is one independent unit; coordinates are never counted as
independent training or evaluation units.

1. **Balanced cosine tree:** root density is uniform, and parent of coordinate
   j>0 is floor((j-1)/2), with zero-based indexing. The conditional density is
   `1+theta*cos(2*pi*parent)*cos(2*pi*response)`. The learner receives this
   fixed graph, the bin rule, and observed training vectors only.
2. **Omitted distant dependence:** joint density is
   `1+theta*cos(2*pi*x[0])*cos(2*pi*x[D-1])`; other coordinates are independent
   uniform. Fit the predeclared local chain with parent j-1. It omits the
   final node's distant informative predecessor. Every local population
   conditional is uniform, so the joint KL approximation floor is at least
   `theta**2/(8*(1+abs(theta)))`, or this number divided by D per coordinate.
   No correct-parent competitor is fitted or given privileged graph access.

Generate both worlds by applying Gaussian CDFs to D independent standard
Gaussian inputs and inverting each specified scalar conditional CDF by 52
bisection steps. The density generator necessarily knows its own world;
the fitting API receives only observations, graph and bins. It never receives
theta, true density evaluations, true conditional probabilities or latent
inputs. True joint log density is used only after fitting for evaluation.

## Estimator and endpoints

Use the unchanged `ConditionalCDFFlow.fit` add-one histogram estimator with
linear interpolation between context-bin centers. Report each cell separately:

- Mean of `(log p(X)-log q(X))/D` over 4096 independent evaluation vectors and
  its vector-level Monte Carlo standard error. Keep negative MC estimates;
  do not truncate them. The population KL is nonnegative, but its finite
  sample estimate can be negative.
- One fit wall time and three decode times for the same 256 Gaussian inputs,
  including the first call; report all repetitions and their median. These
  are CPU implementation measurements. Asymptotic complexity is separate.
- Maximum Gaussian roundtrip error (limit 1e-9), observed-vector roundtrip
  error (1e-10), forward/inverse log-determinant cancellation (1e-8), and
  Gaussian change-of-variables density identity error (1e-8).
- A stochastic latent-copy identity check: name the first Gaussian coordinate
  coarse noise and the remaining D-1 coordinates decoder noise, then use
  the identical fitted tables and graph in the same conditional decoder.
  Samples and inverse log determinants must be bitwise equal. This is an
  algebraic equality control. It is not fitted independently.

No accuracy threshold or monotonic trend is a promotion requirement. Three sample
sizes and three seeds do not establish a convergence rate. The positive
world examines a proved class; the omitted-context world checks a known
structural limitation. Do not interpret its D-normalized floor shrinking
with D as recovery of the distant dependence: the joint floor is unchanged.
MC errors from one fitted model do not cover training variability; retain all
three independent fits rather than treating evaluation vectors as training
replicates. No significance claim or empirical superiority claim is planned.

## Reproduction, resource limits, and failures

Run only after the model, runner, experimental specification and launch script are committed
and unchanged. Require the full source revision on the command line and
refuse an existing output directory. Save source hashes, configuration,
software/thread environment, Slurm job ID, and hashes of train/evaluation
vectors, Gaussian inputs, samples, graph and fitted tables. Save one strict
JSON scalar record per cell durably, then a summary and hashed completion
record. No training arrays are retained. Failures preserve completed records
and produce a failure record when Python retains control.

Limit the allocated job to one CPU, 2000 MB and ten minutes. Set all BLAS
thread counts to one. The runner checks a 570-second soft deadline before
each cell; Slurm enforces the ten-minute hard deadline. A hard kill may leave
partial records without a failure/completion marker; this is not success.
Stop after any numerical/equality failure without adapting bins, seeds,
graphs, bounds or sample counts. No GPU is used.

Companion mathematical tests check the known conditional inversion and
normalization by deterministic values/quadrature. They do not fit a model or
evaluate experimental seeds. Root review precedes any experimental training.
