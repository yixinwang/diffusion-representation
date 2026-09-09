# Cached innovation stage1: frozen engineering protocol

Preparation only. Commit this protocol and complete runner closure before execution. This is a new eager-only, full exact-density development study, not a replacement for the prior shared coarse-FM study. No compilation, data-derived architecture search, checkpoint selection, training on repair arrays, discovery arrays or official test access is permitted. All three seeds77201/77202/77203 must be reported, including failed seeds. The reused canonical repair split prevents confirmatory claims.

## Data, source, models

Use the existing strict `load_observed_flow_data` canonical4,000-fit/1,000-repair RGB32 split; fixture bypass is unavailable. Preserve record IDs, canonical ledger and chart Jacobians. Source guard requires exact full HEAD and byte equality for every qalt source module/test, the manifest, both new runner/protocol/launcher and perceptual appendix sources, and shared utility sources before any data read. Snapshot the closure. External evaluator source and weights must match the existing pinned SHA256s before data access. No network fallback.

Each seed uses one shared analysis A and exact coarse root from `CachedInnovationFlow`: RGB32, levels2, pre2, analysis coarse6/detail4, exact root4, width32, bins8, attention heads4; residual packed45x8x8, coarse3x8x8, total3,072 independent Gaussian coordinates. A train loss is its exact Gaussian code NLL including both analysis determinants divided by3,072; root loss is exact root NLL divided by192. Adam learning rate0.001, batch32, no clipping, weight decay, scheduler or optimizer-based retries. Each stage samples fitting indices uniformly with replacement. Analysis/root/MJ decoder index seeds are training seed+10/+20/+30. All trainable participating parameters must have finite gradients (zero gradients allowed).

Arms:

* M: four observed-prefix blocks, rank16 SPD mixing, integrated-slope scalar innovations, shared A/root frozen; decoder180-second envelope.
* S: identical accessible observed prefixes/conditioner type, no mixing and no unused mixer parameters. Search integer width1..512 using parameter-count algebra only, selecting the smallest width whose complete active model parameter count lies in [M,1.05M]; verify by actual construction. Fit decoder180 seconds. This matches whole model counts, not just decoder counts. Failure to find a width fails the registration rather than relaxing the band.
* J: exact M first90-second decoder checkpoint, optimizer state and fitting RNG state; then up to90 seconds joint A+decoder with root parameters fixed. Load the decoder Adam moments exactly, add analysis parameters with fresh Adam state and learning rate0.001. Root input gradients remain live. Recompute A codes and prefixes on each joint step; fixed-A caches never enter its objective.
* RQS: original four-layer global conditional RQS residual decoder, width32/bins8/heads4, same A/exact root, decoder180 seconds. Report all parameters even if it is stronger/larger than M. No FM or Heun appears in any arm.

The main model initialization uses the training seed. New S and RQS decoder initialization each resets to seed+40 before construction; different architectures need not have equal initial functions, and all initializer/source choices are fixed without repair access. Models are trained in the declared tree order: shared A/root, M/J common prefix, M tail, J tail, S, RQS. This is not randomized thermal order; record hardware and timing, and do not claim a definitive optimized speed ranking.

A-only and exact-root/Gaussian-residual diagnostics use the saved shared analysis/root and all Gaussian coordinates. Charge A-only its shared A stage, root-only both shared stages. Neither is an equally trained360-second competitor. Source packing is identical to the primary arms.

## Strict charged envelopes and failure semantics

Each standalone primary arm has analysis90, root90, decoder180-second wall envelopes. Common data preparation, logit conversion/saving, model construction, device movement, optimizers, shape-only count search, caches and checkpoint/fork preparation are charged in the applicable envelope. Shared prerequisites are charged fully to every standalone arm, not divided among arms. Common data load/logit preparation is charged fully to each seed's analysis. Source verification, preflight tests and environment checks are reported execution prerequisites outside fitting. Evaluation is outside fitting.

Check the deadline before each update. One in-flight update and mandatory preservation may overrun; report actual stage and standalone elapsed time, deadline overrun and longest update. Do not call an overrun exactly360 seconds. A zero-update stage fails; do not extend it. M/J reuse their actual common prefix and each pay its full elapsed time including fork/optimizer state serialization/copying; remaining decoder time is180 minus this charged prefix. The shared exact root/A terminal copy is included in root cost. All kernels are eager. Separate kernel engineering measurements provide no uncharged compile/cache benefit here.

The optimizer performs three aggregated finite host decisions: loss before backward, all gradients before step, all parameters after step. Existing model-internal validation remains and may cause additional host barriers; this is not a claim of three total synchronizations. A failed gradient prevents the step. A failed updated parameter retains the failed model/optimizer rather than rolling back and continuing. Stage-progress JSON, terminal/fork checkpoints with optimizer/RNG, failure checkpoints, phase and traceback persist. Checkpointing is charged; previous stage checkpoints are retained. A post-update failure need not have a checkpoint of the immediately preceding update, and must not be represented as if that state were preserved. Training memory reports include fixed shared cached codes retained during the shared-run branches; they are not optimized standalone memory footprints.

## Frozen evaluation and gates

Freeze every seed's models before any repair likelihood/metric or feature extraction. Each arm gets the same saved2,000x3,072 Gaussian bank (seed+300), arranged as one independent pair per repair array. Save all generated float32 logits before sigmoid and all float64 pixel arrays, including any failing values already produced. Reject nonfinite logits before sigmoid. No clipping, replacement draws, sample selection or metric selection. Preserve Gaussian bank, repair identity hashes, per-array descriptor/energy values, features and likelihood components.

Before primary generation, eight independent fixed sources (seed+99) check complete source inversion<=1e-3 and LD cancellation<=1e-2 per array in float32. Save source, logits, recovered source and both determinants before rejecting a failure. The exact-copy control evaluates the same model on the same source with no additional parameters or noise and requires identical outputs. Fabricated preflight tests independently check nonidentity dense Jacobians and gradients; sampled numerical gates do not prove universal numerical invertibility or tail correctness.

Per-array complete repair NLL is root NLL + residual NLL - analysis forward logdet - outer logit-chart logdet. Save all five columns (root,residual,negative analysis LD,negative outer LD,total), nats/array. Report complete mean/D and residual mean/2,880. J's residual coordinates differ from M's; compare their complete likelihood, not their residual NLL in isolation.

Primary quality metric is the existing `perceptual_appendix/evaluator.py` and full-bank `metrics.polynomial_kid`. Its pinned FID-specific Inception source SHA256 is c6183fff54dd240fe66d53d207f4bd28c06fde98c21b5525f10ca0cc5cef7780, weights SHA2566726825d0af5f729cebd5821db510b11b1cfad8faad88a03f1befd49fb9129b2 (95,628,359bytes), block3/2048features. Common float32 cast, bilinear299 align_corners=False, normalization2x-1, batch32, eval/inference mode, no TF32/autocast; metrics in float64. Use all2,000 generated and1,000 repair features, full unbiased degree-three polynomial KID with no subsets; negative estimates remain unmodified. External features enter evaluation only. Preserve their existing local-only verified source/weights behavior. Feature extraction is outside the fit budget and does not guide any training.

Every seed must independently satisfy all engineering gates:

1. M residual NLL < S residual NLL and M opposite-quadrant covariance absolute error <= half S error. Compute covariance as mean(ab)-mean(a)mean(b) for the existing grayscale top-left/bottom-right quadrant means. Compare each generated bank against the repair bank. If S error iszero, M must also bezero; no positive margin is then established.
2. J complete NLL < M; J primary KID < S and < RQS.
3. Both J mean horizontal and vertical grayscale squared-gradient energies within10% of repair means, using the existing descriptor definition. A zero repair mean requires exactzero.
4. J normalized paired pixel energy <= each of S/M/RQS plus0.0005, separately. Score per repair image is half the sum of two distances to that image minus distance between generated draws, with Euclidean distance divided bysqrt3,072.

Report all six arms and all metrics. No seed average rescues failure. These are development engineering orderings, not significance or population-dominance claims. Pixel covariance can arise through A, coarse root and preceding blocks even in S; this diagnostic does not prove conditional-TC separation or necessity of mixing. No superiority over unrestricted latent diffusion/FM, learned semantic representation, realistic videos, or cost-quality Pareto frontier is implied.

## Execution interface

After exact source freeze, runner accepts only expected full commit, fresh output directory, and pinned local Inception source/weights paths. CUDA is mandatory; no CPU fallback or fixture mode exists. It records GPU/software/host and final payload hashes, marks completed only after every fit and evaluation succeeds, and preserves failures. There is no actual execution authorized by this preparation artifact.

## Preservation and secondary measurements

Full-size memmaps are not complete banks until `generation_progress.json` reports completed_rows=planned_rows=2,000 and status completed. Before each chunk and after flushed logits/pixels, persist its phase and verified complete-row prefix. Failure state records seed, arm, chunk start and complete rows. Logits saved pending validation can include failing values beyond the valid generated prefix; zero-initialized unwritten memmap rows are never samples. Preserve these files on failure without relabeling them complete.

Report PRDC precision, recall, density, coverage and diagnostic tie/duplicate counts from the same fixed features and existing `metrics.prdc` implementation (nearest_k5), in addition to primary KID. These are descriptive secondary outputs and do not alter any engineering gate. Count actual float32 sigmoid endpoints separately for zero and one; no clipping or resampling. Record actual PyTorch thread count4 and launcher thread environment, including NUMEXPR_NUM_THREADS1.

Model-specific output preparation and RNG initialization are charged at the start of the analysis envelope. Environment-report serialization is an execution prerequisite outside fitting. The complete fixed-A cache (every coarse, residual and analysis-LD value) must pass a global finite check before root fitting; failed cache tensors are retained. The launcher requests90minutes, with a warning120seconds before termination and no automatic extension.
