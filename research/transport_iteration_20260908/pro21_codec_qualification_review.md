# Review21: bounded real-image codec qualification

Reviewed September 9, 2026. [Conversation](https://chatgpt.com/c/6aa1bbf5-dc48-83ea-b5d8-3911ceb8b8ea) completed through the ordinary available conversation, displaying Extra High. Pro was disabled in the preceding visible model menu; this is not verified Pro execution. This note is an independent summary, not a verbatim transcript.

The review reported read-only inspection of three files at `84c0cad383be64d6632a5fa61e5d1bc4b605ca5f`. Root independently matched its reported Git blob IDs:

| File | Git blob |
|---|---|
| `qalt/src/qalt/rgb_codec_flow_matching.py` | `bcac5eb981914ba3a98a9cc651ec852d18cd14a5` |
| `qalt/experiments/rgb_codec_readiness/PROTOCOL.md` | `0642e89d9a959164ba9fc4e78efbf713b99bf7f2` |
| `qalt/experiments/innovation_response_pilot_v2/PROTOCOL.md` | `ce12e1b4a17237f3a297b90206556cbbc7c40685` |

## Accepted scope and safeguards

A fresh 600-second codec fitting screen on the fixed 4,000 FIT images can test training-set reconstruction and usable FIT-only normalization/cache. It cannot establish held-out reconstruction, unconditional generation, useful semantic representation, latent-generator competence, or superiority to latent diffusion/FM. Per-channel standard deviation across spatial sites does not establish 1,024 independent useful coordinates.

Root retains a transparent wall-clock policy: no update starts at or after 600 seconds; a final update and required serialization may overrun, and actual elapsed time and overrun are charged and reported. This is not a strict completed-within-600-seconds certificate. Checkpoint I/O advances the wall-based learning-rate schedule. All fitting targets, ordered IDs, hashes, draw records, and frozen state are retained to bind the subsequent normalization/cache and reconstruction evaluation to the same FIT observations. The loader physically constructs repair arrays, but they never enter model computation, statistics, selection or evaluation in this screen.

Global PSNR uses global pixel MSE; SSIM uses the declared valid 11-by-11 Gaussian, sigma 1.5, population moments. All weights freeze before reconstruction scoring. Failures are preserved without changing gates within this iteration.

## Prospective full-generation comparison, not yet launched

The review proposed adding the implemented pixel FM and codec FM to the existing seven native systems. All solver points should be reported: Heun steps 4, 8, 16 and 32 imply 8, 16, 32 and 64 field evaluations. Native full-flow calls must be compared through measured full sampling time and memory, not relabeled as an equivalent single FM evaluation. Codec sampling includes its decoder.

It suggested three fresh baseline seeds, 600 seconds codec fitting and up to 1,800 seconds field fitting, with a saved cumulative-cost-matched latent checkpoint as well as its full endpoint. These remain proposals requiring a complete prospective protocol and resource assessment. The existing native experiment has much shorter fitting budgets; merely appending longer baseline endpoints would not establish a matched-compute advantage. Every standalone cost, all budget endpoints, rejected development work and baseline adequacy must remain visible. An undertrained baseline leaves the comparison unresolved.

Native source dimension is 192 coarse plus 2,880 residual equals 3,072, not 192. Pixel FM has 3,072 source dimensions and codec FM 1,024. Unequal-dimensional Gaussian banks can have the same distributional convention but are not identical sources or equal-dimensional representations. No VAE comparison has been run. The small convolutional FM architecture is a specified baseline, not representative evidence about all latent diffusion or modern multiscale FM.

Review20's invalid permutation falsifier is not used in this qualification. The overall research goal remains unmet.
