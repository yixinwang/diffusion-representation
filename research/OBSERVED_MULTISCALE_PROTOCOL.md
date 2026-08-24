# Observed-data Multiscale QALT protocol

This protocol is frozen after procedural Stage-A confirmation and before fitting any observed image or video model. It preserves the full objective through a representation-to-endpoint staircase rather than treating a small conditional-density result as a generation result.

## Claims and evidence ladder

Stage B1 asks which blocks of the fixed equal-dimensional multiscale representation admit a certifiably accurate one-shot conditional fiber on real images and videos, and which require iterative conditional transport. Stage B2 asks whether the resulting adaptive route plus a neural active sampler improves complete generation endpoints. B1 cannot establish unconditional generation quality; a failed B1 stops B2 because the proposed mechanism lacks observed-data support.

## Data and leakage boundary

### Images

Use the canonical CIFAR-10 Python archive at `/ocean/datasets/community/cifar/cifar-10/cifar-10-batches-py`. The 50,000 official training images are deterministically partitioned by a committed index manifest into 45,000 fitting and 5,000 validation images, stratified by class. The 10,000 official test images remain unread until a separate confirmation freeze. Class labels are supplied either to every method or to none; the first unconditional study supplies them to none.

### Videos

Use the public `sayakpaul/ucf101-subset` archive at immutable Hugging Face revision `b9984b8d2a95e4a1879e1b071e9433858d0bc24a`, derived from UCF101. Preserve its train/validation/test directories. Parse UCF video group identifiers and reject any cross-split group overlap, because clips from one original group can share background and viewpoint. Decode one deterministic 16-frame, 64x64 center clip per source video using a committed temporal rule. No test clip is decoded before a separate confirmation freeze.

Record archive hashes, relative decoded source paths, group identifiers, frame indices, software versions, and every rejected clip. The immutable archive has SHA-256 `e9fcc76af48d320be88c5265f2e0576ecd615956976f6ce4742fdf2b042b71eb`, size 171,386,880 bytes, and is an uncompressed POSIX tar despite its `.tar.gz` suffix. It contains 300/30/75 videos and 195/25/30 `(class,gNN)` groups in train/validation/test with zero cross-split group overlap. Dataset download, metadata validation, and decoding are data preparation, not evidence.

## Shared representation and information

Apply the same fixed one-level separable orthonormal Haar transform to every method. Images retain all 3x32x32 coefficients; videos retain all 3x16x64x64 coefficients. The all-low-pass band is the coarse representation. Every conditional fiber receives the same coarse coefficients, local parent alignment, learned location features, and optional class condition. Every method is fitted on the same examples and evaluated on the same coefficients. No truth mixture label exists for observed data. Use paired uniform dequantization and continuous conditional log densities; the orthonormal Haar chart has unit absolute Jacobian. The paired dequantization draw is keyed by source record and supplied identically to all methods.

The equal-dimensional claim refers to the complete invertible coefficient representation, not only the coarse state. Reports must give both full representation dimension and iterative-state dimension.

## Stage B1: observed conditional-fiber bridge

Development uses fitting and validation data only. The image-first executable configuration, estimand, simultaneous empirical Bernstein bound, and selection rule are frozen in `qalt/theory/OBSERVED_B1_ROUTING_PROTOCOL.md`. Candidate closure families in the broader ladder are:

- parent-conditioned Gaussian scale mixture with 2 or 4 components and a shared small convolutional location/gating head;
- conditional radial generalized Gaussian on 9-dimensional image or 21-dimensional video Haar-detail blocks;
- bounded few-color triangular conditional spline;
- fixed diagonal Gaussian decoder with the same coarse representation and learned location head;
- heteroscedastic diagonal Gaussian decoder with the same parent and location features;
- flexible same-information controls: an 8-component mixture and a conditional coupling flow or masked-convolution model with the same inputs;
- coarse-only unit Gaussian detail decoder;
- unconditional per-band empirical mixture, which tests whether parent conditioning matters.

These density families are closure estimators, not the claimed novelty. The proposed mechanism is certified adaptive transport-depth routing: form a frozen finite family of routed-block candidates, estimate each candidate's paired conditional log-score regret against the strongest same-information iterative control on held-out sources, construct simultaneous upper confidence bounds at the image or `(class,gNN)` video-group level, and choose the least predicted repeated-compute route whose upper bound is within a preregistered tolerance.

All heads have an explicit parameter count. If exact matching is impossible, report validation NLL versus parameter count, a capacity-matched width, a compute-matched model, and an overcapacity strong baseline. Report both fixed-example/update budgets and convergence diagnostics. Minibatches, augmentations, and dequantization draws are keyed by record and shared. Selection minimizes validation conditional NLL under the registered routing certificate; no FID/FVD or test metric selects a head. Diagnose radial probability-integral-transform calibration, angular residuals, cross-block energy dependence, and temporal phase dependence.

Primary development gates, separately for images and videos:

1. Multiscale QALT beats the fixed diagonal and coarse-only controls by a paired bootstrap 95% lower endpoint above `0.01` nat per detail coefficient.
2. Parent conditioning beats the unconditional empirical mixture by a lower endpoint above zero.
3. QALT is within `0.01` nat per detail coefficient of the flexible same-information control by two one-sided equivalence tests.
4. Deterministic Haar round-trip error is below `1e-6` in decoded float precision.
5. There is no source/group overlap and no test access.
6. One-shot conditional resampling is finite and produces valid pixel ranges after the declared inverse transform; qualitative samples are diagnostic and cannot determine selection.

Use development seeds `1100..1104` for initialization and minibatch ordering. If either modality fails, preserve the result and repair only through a newly registered mechanism, not a threshold change.

After an eligible configuration is frozen, B1 confirmation uses untouched training seeds `1200..1229` and the sealed official test split exactly once. Images and video groups, not coefficients, frames, or seeds, are the independent resampling units; seeds quantify algorithmic variability. Holm correction covers all directional and equivalence components across modalities. Passing confirms conditional representation closure and the registered relative routing certificate, not unconditional generation.

## Stage B2: neural endpoint generation

Only after B1 confirmation, freeze complete generative training. The first feasible observed endpoint settings are unconditional CIFAR-10 32x32 and unconditional UCF101-subset 16x64x64. If the video subset is too small for defensible unconditional generation, label it a pilot and do not promote a real-video superiority claim.

Mandatory methods receive identical training examples, conditioning, a common full `N`-dimensional Gaussian base vector, dequantization, and augmentation. Each method may transform or ignore coordinates, but every ignored coordinate is reported:

- full pixel/voxel flow matching or diffusion using the common backbone family;
- full equal-dimensional Haar-latent flow/diffusion;
- conventional compressed VAE latent flow/diffusion with an optimized public or jointly trained stochastic tokenizer/decoder allowed the same fiber-noise budget;
- Multiscale QALT: the common coarse sampler plus the frozen B1 fiber;
- exact same-information split baseline, which must tie QALT by construction;
- QALT with diagonal fiber;
- coarse-only lossy decoder;
- few-step/distilled full-latent control when a stable implementation is available.

The coarse sampler architecture, optimizer, number of training examples, and sampler schedule are shared wherever tensor shape permits. Full methods receive parameter-matched, compute-matched, and strong overcapacity variants. Decoder, chart, and fiber time are included. All methods use common initial noises and sample counts for paired metrics. Reports separate the full representation dimension from the repeated sampler-state dimension; no unequal total dimensions are called equal-dimensional.

Primary endpoint metrics are FID and KID for images; FVD plus frame FID and temporal consistency for videos; precision/recall; held-out likelihood or a common proper conditional score where tractable; wall-clock latency, peak accelerator memory, FLOPs/token updates, parameter count, and sampler evaluations. Evaluator identifiers and preprocessing are frozen. Report confidence intervals over independent training runs and evaluator resamples. Never compare numbers copied from papers under different preprocessing.

Development seeds are `1300..1304`; prospective confirmation seeds are `1400..1429`. Official test endpoints remain sealed until architecture and inference are frozen. Strict superiority over unrestricted latent diffusion is not a valid claim because the exact split control ties; any strict endpoint statement names its baseline recipe and compute budget.

## Stop rules

- B1 failure stops endpoint scaling of this fiber.
- If the optimized same-information control materially beats QALT, expand the fiber only through a new registration and restart development.
- If full or latent baselines are undertrained, label the run diagnostic and do not infer superiority.
- Treat UCF101-subset as a conditional-density pilot only; acquire a larger group-disjoint video dataset before any broad unconditional-video or FVD claim.
- A production efficiency claim requires accelerator latency and allocator memory; NumPy kernel timing is not sufficient.

## Source boundary

CIFAR-10 originates from Krizhevsky's *Learning Multiple Layers of Features from Tiny Images*. UCF101 originates from Soomro, Zamir, and Shah, arXiv:1212.0402; its official documentation warns that clips from one group must not cross train/test. The subset revision and all preprocessing are recorded as implementation provenance. Latent Video Diffusion, Wavelet Diffusion, WaveletFlow/MRCNF, and few-step diffusion remain required prior-art and baseline families, not sources of directly comparable scores.
