# Multiscale QALT: frozen theory-first protocol

## Motivation and claim boundary

The learned dense residual-chart estimators are retired. The next mechanism fixes an invertible, local, volume-normalized spatiotemporal wavelet chart and asks whether only its coarse coefficients require iterative transport. This is realistic enough to expose the token and temporal scaling used by image/video latent diffusion, while avoiding an unpriced dense chart.

No unrestricted quality-dominance claim is possible: a same-information latent baseline can implement the identical exact conditional fiber and tie Multiscale QALT. The registered target is instead a strict Pareto result against (i) a full-token latent diffusion/flow that numerically transports every coefficient, (ii) a lossy coarse-only tokenizer, and (iii) a fixed-chart diagonal VAE decoder. The exact same-information split solver is mandatory and must tie quality.

## Algorithm

Let `W` be an orthonormal 2D Haar chart for images or separable causal-time/2D-space Haar chart for videos. Write `W x = (a, h)`, where `a` is the coarsest field and `h=(h_1,...,h_L)` are detail fields ordered coarse to fine.

1. Learn a flow-matching or diffusion sampler only for `a`.
2. Fit local conditional fiber heads `p(h_l | parent_l(a,h_{<l}), c)`. The first assay uses a zero-mean Gaussian scale mixture with train-fitted mixing probabilities, scales, and local parent features. This family is nonlinear and non-Gaussian after marginalizing the component.
3. Generate `a` iteratively; sample every detail coefficient once from the common base uniforms/Gaussians through the conditional inverse CDF; return `W^T(a,h)`.

The chart is fixed before data inspection. All methods receive the same condition `c`, train samples, total coefficient dimension, base random variables, and admissible local parent features.

## Theoretical study

For a declared multiscale law

`p(x|c) = p_a(a|c) product_l product_i p_l(h_{l,i}|parent_{l,i},c)`,

with exact active sampler and realizable fiber heads, prove:

1. **endpoint parity:** Multiscale QALT equals the full-dimensional target law by change of variables and conditional composition;
2. **density parity:** orthonormal `W` has zero log-Jacobian, so active plus fiber likelihood is the exact image/video likelihood;
3. **iterative complexity:** with `K` denoising/flow steps, full-token work is `Omega(K N)` token updates, while QALT is `O(K n_a + N)` local updates; it is strictly smaller when `K>1` and `n_a<N`, subject to measured architecture constants;
4. **finite-step separation:** for an Euler baseline that transports an analytically solvable fiber, derive a positive accumulated KL/Wasserstein error while the exact fiber has zero solver error;
5. **restricted VAE gap:** at equal coarse-code dimension and a fixed diagonal Gaussian decoder, the expected conditional KL equals the omitted mixture/dependence information and is strictly positive unless that decoder family contains the true fiber;
6. **countertheorem:** an optimized same-information baseline using the same exact fiber ties both quality and asymptotic work. This prevents a universal superiority statement.

The proof must include chart cost, fiber cost, decoder cost, memory, and sampler evaluations. A dense-transform implementation cannot support the efficiency claim; the executable uses local lifting/convolution.

## Stage A: preregistered nonlinear/non-Gaussian assay

Use procedurally generated 32x32 images and 16x32x32 videos with nonlinear coarse dynamics and parent-dependent two-component detail mixtures. The population law is known but truth parameters are unavailable to fitted methods. Development seeds are `700..704`; any confirmation uses untouched seeds `800..829` after a separate freeze commit.

Mandatory methods:

- oracle full law (evaluation only);
- full-token flow/diffusion with the same network family and solver budget;
- coarse-only lossy latent model;
- Multiscale QALT;
- exact same-information split baseline;
- misspecified diagonal Gaussian fiber;
- finite-step full-token Euler fiber control.

Primary gates before confirmation:

- exact split and QALT quality equivalent within `0.02` nat/dimension;
- QALT excess NLL below `0.01` nat/dimension;
- diagonal and coarse-only lower 95% NLL-gap endpoints above zero;
- measured median latency and peak-memory upper confidence bounds below the full-token baseline at matched active sampler and batch size;
- empirical update-count ratio agrees with the declared complexity accounting within 10%;
- no validation, test, or truth parameter enters fitting or method selection.

Failure of any gate stops confirmation and is preserved.

## Stage B: observed images and videos

Only after Stage A confirmation, freeze an observed-data study. Start with CIFAR-10 or ImageNet-32 for images and UCF-101 16-frame 64x64 clips for video, using official train splits for fitting, a train-derived validation split for all choices, and the official test split once for final metrics. If compute makes a strong baseline infeasible, the study is labeled a pilot and cannot support the realistic superiority claim.

Report likelihood or compression-calibrated rate where available, FID/KID for images, FVD plus frame/temporal diagnostics for video, precision/recall, latency, peak memory, FLOPs, parameter count, and sampler evaluations. Repeat data splits and training seeds. Use identical preprocessing, conditioning, augmentation, base prior, evaluator checkpoints, sample count, and test examples. Include distilled/few-step latent baselines because many modern latent models no longer use long samplers.

## Baseline provenance

The study is motivated by primary sources, not by their reported numbers: latent video diffusion explicitly uses an image autoencoder and latent-space temporal model; projected latent video diffusion targets memory/compute; wavelet diffusion and WaveletFlow establish that multiscale fixed charts are prior art. Relevant sources are recorded in `research/IMAGE_VIDEO_BASELINES.md`. Multiscale transforms and analytic fibers are not claimed as individually novel; the research question is the registered Pareto combination and its honest boundary.
