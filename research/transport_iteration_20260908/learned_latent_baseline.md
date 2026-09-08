# Strong learned-analysis latent flow-matching baseline

Date: 2026-09-08. This implemented baseline supports an independently trained comparison with learned invertible analysis and a joint stochastic residual model. Numerical checks supply no quality, efficiency, or semantic-representation claim; no training experiment has run yet.

## Existing APIs and the structural issue

`MultiscaleSplineFlow.encode(x)` returns a full-dimensional flat code and exact log determinant. Code order is deepest coarse followed by detail blocks from deepest to finest. `block_shapes` specifies the layout; `decode(z)` inverts it. However, the first retained coarse block depends only on the fixed deepest Haar lowpass. Learned detail coupling stacks cannot move information from a discarded Haar detail into that retained code.

`FullTensorFlowMatching` provides an unconditional convolutional velocity, dimension-normalized `training_loss`, and `sample_from_gaussian(z, steps)`. Its `levels` argument only imposes image-size divisibility. It can supply the learned coarse-code prior by choosing the coarse spatial size and `levels=1`.

`HierarchicalFlowMatching` learns a coarse prior and conditional detail fields in fixed Haar coordinates. At generation it correctly uses generated parents, but it is a fixed-analysis baseline with separate detail-level fields. It should remain as that comparison. It does not itself implement a learned invertible analysis or a joint residual field spanning all scales.

The existing `CopiedStochasticLatentDecoder` shares parameters and computation with its spline flow. It is an exact equality control. It is not an independently trained competitor and must remain separate from the proposed baseline.

## Construction

Use a learned unconditional `CouplingStack` of at least two alternating layers before Haar decomposition, followed by a `MultiscaleSplineFlow`. Their composition is the analysis map E. The pre-Haar network makes the retained coarse coordinates depend on learned image-space mixing, so the code is no longer restricted to the original fixed Haar lowpass. This gives a learned invertible analysis, without asserting semantic compression.

Initially use two Haar levels. With C image channels, image side S, coarse side h=S/4, and D=C*S*S, retain d=C*h*h coordinates. The two residual blocks have shapes (3C,h,h) and (3C,2h,2h). Pixel-unshuffle the finer block by a factor of two, then concatenate channels, giving one residual tensor of shape (15C,h,h). This is a lossless permutation with D-d coordinates. Unpacking performs the exact inverse. More levels can use factors 2^i for detail level i.

Fit an unconditional flow-matching prior for the retained coarse code. Fit one conditional flow-matching field for the entire packed residual tensor, conditioned on that coarse code. The residual field contains global self-attention on coarse-grid tokens, with state, coarse context, time, and positional features in its input. This makes distant sites and all residual scales available within a single velocity evaluation. It does not replace the residual law by independent sites, bands, or Gaussian output noise.

For 8-by-8 grayscale arrays: D=64, d=4, and residual shape (15,2,2). For 32-by-32 RGB arrays: D=3072, d=192, and residual shape (45,8,8). The attention grid has four or 64 tokens respectively, avoiding full-resolution attention. Two pre-Haar coupling layers and the existing multiscale stacks are the initial analysis. Width and layer budgets must be selected through the shared validation procedure.

## Two-stage fitting and generation

1. Independently initialize the entire analysis. Fit it by its own full observation likelihood against the same D-dimensional standard Gaussian base. Charge every analysis update, initialization, validation run, and encoding pass to this baseline.
2. Freeze all analysis parameters. Fit the coarse prior and the conditional joint residual field using only encoded training observations. Real coarse codes are valid conditioning during conditional training. The loss is total coarse-plus-residual squared velocity error divided by batch times D, with each coordinate weighted equally.
3. At generation accept only one supplied Gaussian matrix with D coordinates. Its first d coordinates drive the coarse prior; all remaining coordinates drive the packed residual field. Generate the coarse code first. Use that generated code as the residual context. Unpack the residual output, concatenate all code blocks, invert the analysis, and return the full observation.

No training or evaluation image can be passed as generation context. Cached encodings are used only for training. No additional random draw is introduced by the decoder or attention. No test array is used to choose model configuration, training allocation, checkpoint, or solver steps.

The analysis is trained separately from the proposed method, with independent initialization and its own charged optimization. Sharing a trained proposed flow would produce a different comparison and would require explicit accounting. The exact-copy equality control is retained only as a separate identity check.

## Implemented methods

The `LearnedLatentFlowMatching` wrapper exposes `train_analysis_loss(x)`, `freeze_analysis()`, `training_loss(x, generator=None)`, and `sample_from_gaussian(z, steps=16)`. The second-stage methods require a frozen analysis. The frozen state should survive model saving and loading, and stage-two backpropagation must not reach the analysis or input chart fitting.

Use the existing coupling, spline, flow-matching, and Heun primitives. Add only the pre-analysis composition, residual pack/unpack, globally attending conditional velocity, and stage controls. Keep the existing baseline modules unchanged. Tests must independently verify packing Jacobians, all source coordinates, global dependence, freeze behavior, and generation without observed contexts.

## Cost and evaluation obligations

Report end-to-end analysis training, latent/residual training, encoding, inference, parameter count, memory, source count, and solver field evaluations. A fixed total budget must include both stages. Training-budget splits and solver resolutions require a modest, matched validation allowance; a poorly trained first stage cannot be excused by omitting its cost, and the baseline should not be handicapped by an arbitrary zero-length second stage.

The residual decoder is deliberately expressive and may be expensive because it integrates D-d coordinates. This is a strong stochastic decoder comparison. Its speed may differ from that of a one-pass VAE decoder. Compare quality-cost curves under matched total budgets. Do not infer a speed advantage from equal iteration counts, or infer quality from flow-matching training losses.

Existing fixed-step Heun sampling has a valid generated probability law, but the implemented solver has no proven exact invertibility or available exact likelihood. Do not report a spline Jacobian as the density of the composed FM baseline. The continuous-time ideal and the finite solver must be distinguished.

The 8-pixel construction is a development check. The 32-pixel construction is a finite-capacity image baseline. Neither reproduces a modern pretrained latent image/video system. No theoretical result presently establishes improved semantic representations, image/video quality, or runtime for this baseline. A successful comparison against it would not by itself establish superiority over state-of-the-art latent diffusion or latent flow matching.

## Implemented companion and checks

The separate implementation is `qalt/src/qalt/learned_latent_flow_matching.py`, with independent tests in `qalt/tests/test_learned_latent_flow_matching.py`. Its two-stage methods follow the proposed calling sequence. Packing is checked as an exact permutation with a dense Jacobian. Tests verify that a fine-detail perturbation with unchanged original Haar coarse can alter the retained learned code, that a nonzero output head exposes distant state and context gradients through global attention, and that frozen analysis receives no stage-two gradients. They also verify generation from its own coarse output, complete source recovery for zero fields, and saved freeze state. These are numerical and structural tests. No training experiment was run.

The planned real-image input convention is float64 with a shared external logit transform. Under that convention construct the wrapper with `unit_interval=False`, and apply the shared inverse transform after sampling. Do not apply both the external logit and the wrapper's optional internal logit. The FM composition exposes no `log_prob` method: the analysis-only likelihood remains unchanged when FM fields change the generated law and cannot score that final law.
