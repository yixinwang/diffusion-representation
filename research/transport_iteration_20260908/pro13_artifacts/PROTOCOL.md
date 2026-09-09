# Prospective PSC protocol — not executed or submitted

## Decision and scope

Advance the **compiled packed-sign discovery plus exact tent-copula sampler** to a synthetic, observation-only comparison. Do not reopen the old tilted-Heun frontier, use native arrays, modify the active branch, or claim that this declared-law replacement has preserved the cosine distribution. The graph/kernel question has local evidence; global-FM and end-to-end matched-quality claims remain untested.

The experiment has two explicitly distinct endpoints: (1) actual recovery and ideal-law KL for the tractable pair models; (2) common masked-observable quality and total computational cost against learned global FM. Endpoint (2) is NOT a certified matched-full-joint-KL frontier unless a valid likelihood certificate is separately implemented for the actual finite-step FM sampler.

## Data, information, and freezing

Use three new public seed identifiers 1310001, 1310002, 1310003, separate from this package's 1309xxx development seeds. For each seed generate a positive-class law and its centered-zero-mean stress law. Each cell has 4000 fit, 1000 repair, and 2000 final arrays, all D=3072. Roots are unknown 8-bin histograms, C_1 density >=.5; four unknown matchings and group-shared theta functions obey the stated class. Positive curves must be nonlinear and satisfy the .35 mean-signal lower bound. Stress curves are centered using their actual root law. Hide all fixture probabilities, matchings, theta functions, phases, Gaussian generating sources, and evaluation summaries from learners.

Every learner receives the same observed fit arrays and public C_1 index, roots, groups, psi/knots, histogram grids, kappa, L, positive-class a, and class equations/inverse. Training-data access logs and hashes are mandatory. The exact 2000/2000 structural/regression split is frozen. No oracle matching repair. The 1000 repair arrays never enter the theorem estimator. Failure to recover a perfect threshold matching triggers the declared product fallback. Any theorem-changing refit or smoothing is labeled a different empirical estimator.

Lock data hashes, code, model configurations, thresholds, seeds, masks, optimizer/solver options, all fitted states and all final sample-source banks before opening final outcomes. Store checkpoints and raw predictions before scoring. Register the whole protocol before generating final arrays. The local synthetic cases remain development evidence, never fresh confirmation.

## Arms

| Arm | Fitting and inference |
|---|---|
| Candidate | Packed signs, tau=159/1000, degree-one-or-product graph rule; last-2000 pooled 32-bin moment regression clipped to .45; all-4000 Laplace roots; exact quadratic inverse. |
| Dense structural ablation | Same candidate estimator except independently computed real-psi Gram with threshold .175; permit symmetry-aware SYRK and GPU Gram. Same theta procedure and inverse. |
| Independent analytic-pair control | Independently implemented graph and root learning from the same arrays; public fast packed discovery is allowed, not withheld. Fit each accepted group's 32 coefficients by pooled conditional maximum likelihood, box constrained to [-.45,.45], on the last 2000 arrays. Use a registered safeguarded Newton/bisection solver with 64-step cap and recorded KKT residual; report every unconverged cell. Use the same public exact inverse. No true-graph input. |
| Exact candidate copy | Deserialize the candidate's actual fitted state, use the same full Gaussian inputs, and copy the exact map. Assert sample, log-density, and masked-prediction parity. Charge the inherited candidate fit cost, plus actual copy/serialization cost; do not assign it fictional zero learning cost. It is a required tie/control, not an independently fitted arm. |
| Learned global FM | A globally connected field with four width-1024 residual SiLU blocks; every output can depend on every residual input and all root context. No matching is provided. Input features include state, observation mask, observed values, public psi(Phi(state)), root context, the same 32-bin C_1 one-hot encoding, and 64-dimensional fixed time Fourier features. The common learned root generator is available with its training cost charged. |

Product and constant-context fits are reported as restricted diagnostics, not the strongest baseline. The analytic control may adopt the public bit kernel; any gain over a deliberately dense implementation must be called an implementation/discovery-cost gain, not an advantage over all analytic decoders.

The FM field works on Gaussianized residual coordinates. Use independent standard Gaussian training noise, t~Uniform(0,1), straight interpolation (1-t)z+t*x, and regression target x-z. Never supply the fixture's generating Gaussian noise or true conditional vector field. Train on a fixed 50:50 mixture of all-residual-missing masks and independent half-coordinate masks, with roots observed. Only hidden-coordinate velocity loss is used. This gives a properly trained conditional-sampling baseline rather than incorrectly clamping an unconditional ODE at inference. It does not assert that its separately trained mask-conditionals form an exactly consistent joint model.

Use AdamW, learning rate 3e-4, beta=(.9,.999), weight decay 1e-4, batch 128, gradient norm cap 1. Record actual parameter count. Primary FM training is FP32 with disclosed TF32 settings. A faster precision policy must be frozen before final scoring and pass the same repair/numerical gates; do not time only slow FP64 FM against fast FP32 candidate. Public information/analytic features are not withheld to handicap FM. This arm receives more representational capacity than the compact pair estimator, not an artificial parameter match.

## Equal compute and solver selection

Allocate the same exclusive hardware bundle to each arm: four CPU cores and the same single GPU. Use two common *total fit-plus-repair allowances*, 600 and 6000 node-seconds per cell. The larger allowance is a continuation of the same FM training trajectory, not a new hyperparameter search. All arms may use the entire allowance; unspent candidate time is reported, not filled with pointless work. The allowances are experimental compute caps, not predicted run durations.

Charge data transforms, root learning, feature construction, graph discovery, compilation/JIT, transfers, optimizer updates, checkpointing, repair generation and selection. Record node elapsed time, CPU core-seconds, GPU device-seconds, memory and model/cache size separately; do not equate a popcount word with a FLOP. Use a fixed 80:20 training/repair allocation within each cap and four evenly spaced training checkpoints. Preserve short/truncated runs and actual update counts. A cell with fewer than 1000 FM updates is an insufficient-training-budget observation, not evidence of adequately trained FM inferiority. No extra uncharged tuning when a budget is exhausted.

FM solver choices are fixed Heun budgets {4,8,16,32,64} actual field evaluations, not ambiguously labeled steps (each complete Heun step costs two evaluations). Evaluate the registered checkpoint/solver grid only on repair data, using identical frozen source/mask banks. Select the lowest total-cost configuration meeting every repair quality gate; if none meets the gates, declare that arm/cap ineligible at the target quality and retain its full curve. Any unfinished repair selection is incomplete, not silently successful. Faster product fallback is never compared as a matched-quality success.

For positive-law engineering gates, require estimated ideal pair-model joint KL <=1 nat/array, masked nonlinear-coordinate excess MSE <=.002, and pooled group/context nonlinear-interaction RMSE <=.03. These are engineering targets, not consequences guaranteed for each fit by the expected-risk theorem. For FM, use the common observable gates; do not pretend its joint KL is available from a heuristic CNF likelihood or a few moments. Preserve marginal, nonedge, and full-array sample-score checks as auxiliary falsifiers. Stress laws are expected graph failures and are reported without moving thresholds or fitting a rescue algorithm.

## Masked-observable scoring

Reveal roots and independently mask half of residual coordinates with a source-independent mask seed. Do NOT choose masks from true matching. Use h(x)=psi(x)/A and the observable event psi(x)>=0. The true Bayes conditional mean/probability is used only by the final scorer, after all predictions are locked; it is never training information.

The candidate/analytic models have exact mask-conditional predictions when C_1 is observed. For all-arm generated-output scoring, use two independent banks of 16 conditional draws (32 draws total) from full-dimensional Gaussian sources. Score the product of the two prediction-mean errors: it is unbiased for squared error of the model's conditional mean, unlike the naive squared error of a Monte Carlo sample mean. Alternatively report the corresponding unbiased U-statistic proper score. Keep the exact analytic prediction score as a separate diagnostic. Never claim the 1e-4 exact-prediction bound for a noisy Monte Carlo average without its variance term.

Compute paired uncertainty using *arrays* as independent units, and keep each law/seed separate. Do not bootstrap 1440 shared-context pairs as 1440 independent contexts. Pool pairs only where the conditional-independence structure justifies it analytically. Report graph recovery, zero/multiple-degree vertices, empty bins, coefficient clipping, conditional/root KL pieces, and constant-context/product scores. Final generated samples, raw per-array scores, masks, and actual Gaussian source hashes are retained.

## Timing and decision rule

Benchmark the full Gaussian-to-observed pipeline, not scalar inversion alone. Include all D Gaussian-source handling, learned roots, feature evaluation, graph-dependent layout, context lookup, inverse, output transforms, and synchronization/transfers. Also time masked prediction including any conditional integration or Monte Carlo required by the arm.

Use batches {1,64,1024}, warm and cold measurements separately, 30 interleaved warm repetitions, and fixed full-Gaussian source banks. Compare optimized packed kernels with full GEMM, SYRK, and GPU implementations when available on the same allocation. Record compiler flags, CPU/GPU identity, BLAS/torch versions, thread settings, and every failed finite/roundtrip/precision check. Cold compilation is not hidden inside a different arm's warmup.

Report end-to-end cost at sample demands S in {1000,100000,1000000}:

    C(S) = root + discovery + regression/neural training + compilation
           + repair/selection + S * full-pipeline sampling cost.

Count shared stages once in execution and attribute their actual cost to each standalone method comparison. The exact-copy control must tie the candidate's underlying generative law. A positive result may establish faster observed-only discovery and faster sampling at the registered observable quality on this class; it cannot establish universal superiority, full-joint-KL domination of FM without a valid certificate, or native image/video quality.

## Preservation and stop conditions

Keep Python-popcount timing losses, source hashes, the compiled-wrapper initial tuple/list assertion failure, all fallbacks, uncentered-sine warnings, unconverged analytic MLEs, FM budget failures, and changed-law quality losses. Do not rewrite the old Pro11/Pro12 results. Numerical failure, state/source mismatch, native-data access, insufficient graph degree checks, or an exact-copy mismatch stops that cell before final scoring. Any subsequent repair requires a new source revision and an explicit amendment, not hidden replacement of failed output.
