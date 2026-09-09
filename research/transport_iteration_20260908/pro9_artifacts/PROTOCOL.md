# Pro9 native RGB32 experiment — proposed freeze, not executed

Scientific base: yixinwang/diffusion-representation,
agent/observation-transport-audit-20260908,
`bb024437a761c096375a042239f77262724dca14`.
This package does not change that checkout, submit a job, or provide a native
study runner. A reviewed implementation/source closure must be frozen before
execution. Do not use pending KID or kernel results to change this design.

## Question and staged decision

Does a cached, explicitly globally mixed conditional innovation model repair
native long-range dependence, and does joint endpoint likelihood then improve
actual image quality enough to justify its whole-pipeline cost?

There is one head (8 fixed real-line bins), one rank (16), one partition
(4 interleaved blocks), and one main candidate width (32). No architecture,
head, rank, solver, optimizer or seed search. Three seeds: 77201,77202,77203.

Stage I trains S, M, J and RQS for every seed. A failed mechanism gate ends this
iteration as a negative; no scale-up or practical-superiority claim is allowed.
If the mechanism gate passes, Stage II runs all five stronger control
configurations at all three seeds with the already frozen settings below.
This gives a fail-fast 12-fit first stage, at most 27 trained configurations.
Sharing measured A/root prefixes physically is allowed; their entire cost is
charged to each resulting standalone model. There is no early exit within the
three Stage-I seeds based on a favorable result.

## Data and information

Use exactly the frozen canonical 4000 fitting images and all 1000 reused repair
images, with the existing ID/partition/dequantization/hash ledger. Original
40k/5k/5k membership rules remain binding. Official test and excluded discovery
stay closed. Do not relabel repair as untouched or construct a new partition by
sampling those protected pools. Use the existing strict float64 dequantization
and outer logit before float32 model arithmetic. No resizing to an easier grid.

Generator fitting uses fitting images only. No repair covariance, KID, labels,
external feature extractor, pretrained generator, VAE, generated teacher samples
or guidance enters its loss. Inception weights are evaluation-only and shared.
The exact root receives 192 source coordinates; all residual methods retain the
other 2880. Standalone controls receive all 3072. No source temperature, clipping,
resampling, calibration on repair, or post-generation denoising.

## Model arms and whole-model parameter matching

P is the complete instantiated main J model count: retained analysis + exact
192-coordinate root + the default cached rank-16 residual decoder. The latter
has 200436 parameters in the tested standalone reference; this is NOT P.

Before opening data, instantiate all complete models and save their module and
parameter censuses. Diagnostic M has the identical count as J. For all other
retrained controls, choose widths by shape only so the COMPLETE parameter count
is at least P and no more than 1.05 P. The one-sided allowance deliberately
favors controls. Count frozen learned modules and zero-context compatibility
weights; count buffers/cache bytes separately. No unused padding parameters.
If the width family cannot achieve this interval, fail the design gate rather
than silently weaken or match only the residual network. `integration.py`
supplies a generic closed-failure shape matcher, not finalized native widths.

| ID | Complete model | Purpose |
|---|---|---|
| S | Shared A + exact root + cached scalar decoder, mixer absent; widened active conditioner | Same-head, same-context, conservatively parameter-matched dependence ablation |
| M | Shared A + exact root + rank-16 cached decoder; A/root frozen after prefixes | Direct dependence mechanism |
| J | Same architecture as M; final phase jointly trains A and decoder, with root weights fixed | Main repair and learned-chart test |
| RQS | Shared A + exact root + existing four global RQS couplings; shape-matched width | Strong exact-flow/control for cheap head versus current globally attending decoder |
| LFM | Shared A + exact root + globally attending ordinary residual FM | Learned-analysis stochastic latent comparator, all D sources |
| LFM_GR | Same but Gaussian-reference residual FM | Strong path control |
| FULL_FM | Standalone full-D globally attending ordinary FM, no A/root | Full standalone budget and whole-parameter control |
| FULL_GR | Standalone full-D globally attending Gaussian-reference FM | Strong full-model path control |
| A360 | Gaussian-analysis-only generator, widened to whole-model P range | A real whole-budget rival, not the old 90-second control |

Use the existing complete exact-root `GlobalInnovationFlow`, not its former
coarse Heun sampler. LFM/LFM_GR share that exact root, so their comparison cannot
credit J with removing 32 coarse-FM calls that only the comparator pays. FULL_FM
and FULL_GR have no separate root/analysis charge but receive the whole budget.
The same-weight stochastic decoder copy is an additional numerical control;
it copies the candidate law and inherits its full costs, and must tie exactly.

A/RQS/the cached module must use the same audited scalar arithmetic policy and
whole-pipeline checks. Do not benchmark an unchecked candidate against a checked
baseline. Permit the existing dense RQS optimization as an experimental
implementation only after its independent numerical/gradient gate; do not
silently change the active repository default. Preserve checked eager timings
for all arms. Compiled timings are a separately labeled complete table, with
first-use costs, and are eligible for an efficiency claim only when every
compared implementation passes its frozen equivalence gates. Never select a
favorable head or quality metric based on these timing results.

## Training and cost allocation

Model-specific construction, device movement, preprocessing/caching, optimizer
creation and fitting share a 360-second synchronized device-wall budget. The
common source/data verification and data I/O bill is reported and charged equally
in addition; it is not silently omitted. Absolute charged phase endpoints for
staged arms are 90,180,270,360 seconds. Shared-stage reuse is charged by its
measured standalone cost, with copying/movement additionally charged. Empty
phases/zero optimizer updates are failures.

S/M/RQS/LFM/LFM_GR: Gaussian-analysis prefix to 90; exact-root likelihood prefix
to 180; residual fit to 360. The 270 boundary changes learning rate only.
J: same two prefixes; residual fit to 270; joint complete NLL to 360.
FULL_FM/FULL_GR/A360: their sole fitting procedure gets the complete 360-second
budget; no gratuitous 90+90 allocation is removed from those models.

Batch size32, Adam, learning rate1e-3 before charged time270 and1e-4 thereafter,
no weight decay, no EMA or tuning. Whole-model NLL is divided by3072 for training
scaling; conditional residual loss by2880. Report unnormalized per-image NLL as
well. Data-index streams use NumPy PCG64 with the declared seed and fixed order
prefixes; independent interpolation/source streams use declared disjoint seeds.
Different time-limited update counts/exposures are recorded, not hidden.

At the J joint boundary: clear the persistent analysis freeze flag, re-enable
analysis and decoder gradients, invalidate cached A(fit) values and frozen U
frames, re-encode each minibatch, and recreate the optimizer. Hold root weights
fixed, but retain gradients through root inputs back to A; do not wrap the root
in no_grad. This makes M-versus-J an analysis-adaptation comparison, not extra
root training. A stale fit-code
cache would defeat the representation experiment. Report each stage's update
counts, examples, order hashes, complete losses, source hashes, memory and wall
costs. Select the endpoint checkpoint only, never a best repair checkpoint.

Check deadlines before each update. One crossing update is preserved and fully
charged. Exceeding the total model-specific budget by more than5% is a budget
failure, not an accepted equal-budget result. Any smaller overrun still enters
the cost table. Uninterruptible/SIGKILL failure remains a failure; no partial
model is silently promoted. An algorithmic cost advantage cannot rest on a
candidate that exceeded a control's measured fitting budget.

## Numerical and cache preflight, before native fitting

Run the supplied CPU audit and additional target-GPU checks. CPU PASS is not GPU
PASS. Frozen tests cover float32/float64 scalar values, inverse, normalization,
zero-slope limits, identity and real-line tails; gradient checks away from knots
and finite one-sided behavior at knots; matrix orthogonality/inverse/logdet;
full tiny-model dense Jacobian; current/future-mask invariance; inference cache
invalidation on training and checkpoint loading; exact-copy identity.

On GPU, compare scalar float32 outputs to float64 reference with absolute1e-4,
relative1e-4; compare source/parameter gradients in both directions with
absolute1e-4 + relative1e-3. Check tails at +/-4, +/-20 and +/-10000 and adjacent
representable values at all fixed knots. Use no failed-discriminant clipping.
Before accepting a trained whole model, retain the existing full-composition
max absolute source/code error1e-3 and logdet cancellation1e-2 on eight fixed
untruncated Gaussian sources, plus finite-gradient/loss/output requirements.
Also audit a64-row source bank. Count sigmoid boundary outputs explicitly.

Prepared inference U frames must remain identical under the same weights,
be invalidated on checkpoint changes, and ignore batch composition. Prefix
cache tests alter current/future residuals while retaining actual prior blocks;
heads/eigenvalues must be bitwise unchanged. All GPU/compiler/numerical failures
are saved before any native quality or warmed-latency claim. No tolerance or
compiler fallback is changed after observing a failure.

## Frozen generated banks and diagnostic outputs

Generate2000 images per seed/arm/NFE from a complete saved Gaussian source bank,
seed training_seed+10000; use identical actual bytes and source-coordinate order
in every arm. One fixed independent pair is assigned to each of the1000 repair
images for the canonical ES. Retain raw logits, unit pixels, source arrays,
per-image scores, descriptor arrays, and input/source/output hashes.

All FMs are reported at4,8,16,32,64 actual velocity calls, two calls per Heun
step. No outcome-selected NFE or root budget. Report every prescribed point,
including adverse results. No approximate FM likelihood is reported as exact.

Additional diagnostic banks, no extra fitting or model selection:
(A) the shared90-second A inverse of the same Gaussian;
(B) the learned exact root plus Gaussian residual, through that same A;
(C) M with all mixing eigenvalues set to zero during generation, other weights
unchanged. This intervention changes downstream prefixes and is not a retrained
control. A versus B isolates adding the root at fixed A; B versus S/M identifies
residual effects. S versus M is the primary trained mixer comparison; M versus
J tests analysis/decoder adaptation while the root weights remain identical. Do not compare isolated root NLL terms across changed
A charts as if their target entropies were unchanged. Complete image NLL remains
comparable among exact normalized models.

## Metrics, advancement gates and interpretation

Freeze the Inception implementation/weight/preprocessing/hash closure from the
existing22ac114 evaluator before reading any scores. Primary perceptual statistic:
unbiased full-bank KID, degree-three polynomial kernel,2048-dimensional features.
Negative KID estimates remain valid and are not clipped. Also report canonical
normalized-pixel ES, all PRDC components with nearest_k=5, small-sample FID,
opposite-quadrant covariance, both gradient energies and the previously frozen
products/descriptors. FID is descriptive, not a replacement winning metric.
No reference subset, score or dimension is selected after viewing results.

Stage-I mechanism gate, required at EACH of the three seeds:
1. M's repair conditional residual NLL is below S's under the shared fixed A.
2. With kappa denoting the frozen opposite-quadrant covariance, require
   |kappa_M-kappa_real| <= 0.5*|kappa_S-kappa_real|. No denominator or ratio is
   used. A zero S error therefore demands zero M error; it is not a license to
   redefine this endpoint. Also require J's covariance error <= M's.
3. J has lower full-image NLL than M and lower KID than both S and RQS.
4. J's horizontal and vertical mean gradient energies are each within10% of
   the repair mean; ES_J <= ES_S+0.0005 and ES_J <= ES_RQS+0.0005.

These are proposed engineering advancement margins, not population confidence
claims or calibrated significance levels. Their strict conjunction may fail;
that failure is useful. Better dependence but worse KID is not a quality repair.
Failure of J while M passes distinguishes adaptation failure from mixer failure.

Only if Stage I passes, run all five Stage-II configurations without changing
any model, budget, data, metric or threshold. The practical pilot gate at EACH
seed requires J's KID below every trained rival and every prescribed FM NFE;
ES no more than0.0005 above each rival; PRDC precision and coverage no more than
0.02 below each rival; and both gradient-energy guards above. Use a predeclared matched-quality frontier, not the fastest low-quality FM.
An FM family/NFE point is eligible only if, at ALL three seeds, its KID is at
most KID_J+0.001, its ES is at most ES_J+0.0005, and its PRDC precision and
coverage are each at least J's value minus0.02. These raw-metric tolerances are
fixed here, before results; all ineligible and eligible points remain reported.
Do not select a different eligibility rule or metric after evaluation.

Against the fastest eligible FM point for each measured batch/seed, J must be
at least10% faster at batch64 and no slower at batch1. Peak allocated AND
reserved inference memory must not exceed that corresponding FM point's peaks.
If the eligible set is empty, report a quality result only: matched-quality
cost superiority has not been measured. Otherwise failing any cost guard means
no combined quality-cost pass. This is a finite, registered frontier comparison,
not strict dominance over every low-NFE approximation or the exact-copy model.

Timing uses matched hardware, fixed interleaved order seed77299,20 warmups and
50 measured repetitions per arm/NFE/batch1 or64; record median,p95, every sample,
CUDA-event and synchronized wall time. All other models are offloaded. Include
root, all conditioners/scalar/matrix transforms, A inverse, outer sigmoid,
source generation where included equally, validation and assembly. Report fixed-
source and Gaussian-RNG-inclusive times separately; RNG-inclusive is the primary
end-to-end statistic. Report frame-cache construction, compile/first-use cost,
host memory, fitting cache, peaks and resident parameters separately. Never infer
whole-model speed from a scalar kernel benchmark. No amortized break-even claim
without charging setup and naming the generated-image count.

Three seeds are a falsification pilot, not strong evidence of population
superiority. Even three concordant paired signs give a one-sided sign-test
probability of1/8 under exchangeability, not<.05. Reused repair additionally
precludes a fresh-confirmation interpretation. Report each seed, paired
contrasts and conditional bank-resampling uncertainty separately; do not treat
2000 generated samples or repeated timing calls as independent training seeds.

## Separate equal-dimensional representation test

After generator checkpoints are frozen, permit fitting labels to train the
same evaluation-only ridge classifier for each model with an A chart. No
classifier or evaluation feedback reaches generator training. Freeze TWO views:
C in192 dimensions, and the complete (C,R) chart in3072 dimensions. Do not choose
the favorable view. Standardize each coordinate by fit-only mean and standard
deviation (floor1e-6); use an unpenalized intercept, one-hot targets and
mean-squared ridge objective with lambda1. Score accuracy on the reused repair
set. Use the same fitting/repair IDs and labels for every representation.

Compare J to all shared-A latent controls and the whole-budget A360 control,
with the identical readout and dimension. A representation advancement requires
at least2 percentage points above the best matched-dimensional control at EACH
seed in BOTH predeclared views. A one-view gain is reported only for that view.
Charge feature extraction, model fitting and probe costs; report feature-encoding
latency and memory separately. This tests a fixed linear readout, not universal
semantic superiority. Standalone FM hidden activations have no automatically
equivalent A representation and are not silently assigned a favorable layer.

## Preservation and claims

Save every failure, endpoint checkpoint, source closure, config, stage ledger,
parameter census, raw source/output/feature banks, all score arrays and complete
timing tables, with hashes and a terminal status. The previous native energy/
cost failure and the accelerated-convex-optimizer failure remain in the record.
Pending canonical ES recomputation, perceptual evaluation45571074 and dense
RQS benchmark45570945 are not completed by this proposal. Even a full pilot pass
would only justify a separately frozen fresh-data/greater-budget study, not
image/video or SOTA superiority, and never strict superiority to the exact copy.
