# Perceptual evaluator readiness — 2026-09-08

Status: dependency and metadata inspection only. No package installation, weight download, image decoding, feature extraction, job submission, model training, or user-environment changes were performed. This appendix is prospective and cannot supply generation-quality evidence yet.

## Existing PSC environment

Read-only inspection used `ssh -o BatchMode=yes bridges2-codex` and `/ocean/projects/mth250006p/ywang26/pytorch/bin/python` with `importlib.metadata` / `find_spec`, followed by a `python -B` import smoke check. The metadata check completed; the import check returned no retained output before its process session expired, so installed metadata does not establish successful imports or runtime compatibility.

| Component | Metadata version | Location/status |
|---|---|---|
| Python | 3.10.9, GCC 11.2.0 | `/ocean/projects/mth250006p/ywang26/pytorch/bin/python` |
| torch | 2.10.0 | `/ocean/projects/mth250006p/ywang26/pytorch/lib/python3.10/site-packages/torch/__init__.py` |
| torchvision | 0.25.0 | same site-packages root, `torchvision/__init__.py` |
| Pillow | 12.1.0 | same site-packages root, `PIL/__init__.py` |
| pytorch-fid | absent | neither distribution metadata nor import spec |
| torch-fidelity | absent | neither distribution metadata nor import spec |

Resolved Torch cache root: `/jet/home/ywang26/.cache/torch`. Both `hub/checkpoints` and `checkpoints` under that root are absent, so no weight filenames were found there. This is a narrow cache check, not evidence that no evaluator weights exist anywhere on PSC. No broad home/project search and no weight contents were inspected.

Exact repeatable import check (does not instantiate/download a model):

```sh
ssh -o BatchMode=yes bridges2-codex '/ocean/projects/mth250006p/ywang26/pytorch/bin/python -B -c "import torch,torchvision,PIL; print(torch.__version__,torch.version.cuda,torchvision.__version__,PIL.__version__)"'
```

## Candidate source and weight pins

Prefer the FID-specific Inception implementation from [mseitzer/pytorch-fid](https://github.com/mseitzer/pytorch-fid), commit `b9c18118d082cbd263c1b8963fc4221dc1cbb659`. Its [Inception source](https://github.com/mseitzer/pytorch-fid/blob/b9c18118d082cbd263c1b8963fc4221dc1cbb659/src/pytorch_fid/inception.py) SHA256 is `c6183fff54dd240fe66d53d207f4bd28c06fde98c21b5525f10ca0cc5cef7780`. Use its default FID-specific network, final 2048-dimensional pool features, 299×299 bilinear resizing with `align_corners=False`, and its own input normalization. Ordinary torchvision classifier weights/features are a different evaluator and must never be silently labeled standard FID. The implementation is Apache-2.0; its LICENSE file SHA256 is `c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4`. Library/interpolation differences still prevent an assertion of bit-identical TensorFlow FID. [Primary documentation](https://github.com/mseitzer/pytorch-fid#readme).

Source specifies [release asset pt_inception-2015-12-05-6726825d.pth](https://github.com/mseitzer/pytorch-fid/releases/download/fid_weights/pt_inception-2015-12-05-6726825d.pth). GitHub release API reports 95,628,359 bytes and `digest: null`. **The full model SHA256 remains unknown: no weights were downloaded.** The eight-character filename suffix is insufficient as a full cryptographic pin. Before evaluation, acquire the weights into an isolated evaluator cache under the existing experiment authorization, record the complete downloaded hash and source URL, then freeze that hash in the protocol. The inspected project license establishes code licensing; this audit does not independently establish a separate weight-specific license or original training-data rights. Preserve upstream notices; no weight redistribution is proposed.

For metric formulas, primary candidates are [torch-fidelity KID](https://github.com/toshas/torch-fidelity/blob/5e211a950a7b45206bd4976813ffd6aed6cf4ccc/torch_fidelity/metric_kid.py), commit `5e211a950a7b45206bd4976813ffd6aed6cf4ccc`, and [PRDC](https://github.com/clovaai/generative-evaluation-prdc/tree/e320c1d2811d33081361a08f595b43830b78641c), commit `e320c1d2811d33081361a08f595b43830b78641c`. KID can operate on the same saved Inception feature arrays using the unbiased degree-three polynomial MMD, kernel `(x·y/2048+1)^3`; negative estimates are valid. The cited KID source attributes adapted routines to BSD-3-Clause MMD-GAN. If any implementation is copied, preserve its notices and audit the selected package license. Pin actual source hashes before execution.

## Minimal isolated evaluation plan

1. Freeze the appendix protocol before feature extraction: all completed prescribed arms and NFE settings, exact preserved bank hashes, equal sample counts, the previously authorized repair-reference IDs, common preprocessing, feature source/weight hash, metric parameters, and all reported outputs. Do not choose checkpoints, remove weak arms, tune models, train feature extractors, or feed these scores back into selection. External ImageNet-derived weights supply evaluation information only; no method receives that information in training, architecture selection, gradients, or sampling. Report this distinction explicitly.
2. Use a separate project-local evaluator directory/environment and cache, leaving the existing PyTorch environment unchanged. A lightweight isolated environment inheriting the existing packages is possible only after dependency/import compatibility checks; install pinned evaluator code with no dependency upgrades. Set an explicit isolated `TORCH_HOME`, load the already hash-verified local weights, and fail rather than download implicitly during the evaluation job. Freeze a dependency manifest and test only fabricated tensors first. No installation command was executed here.
3. Read only preserved generated banks and the already approved development reference through its strict allowlist/ID guard. Never official test or excluded discovery. Preserve floats through a single declared common image conversion; do not independently choose PNG quantization, clipping, resizing, or antialias settings per arm. Freeze exact conversion after inspecting the bank schema, before evaluating contents. Nonfinite/out-of-range inputs should fail or receive a predeclared, fully reported shared rule. Extract once in `eval()` plus inference mode, gradients disabled, fixed precision/batch size, no augmentation. Save feature and input hashes, not newly fitted feature models.
4. Primary appendix statistic: unbiased full-bank KID, with exactly matched counts and no per-arm subsampling advantage. Secondary descriptive FID uses the same 2048-dimensional features; small banks have singular empirical covariances and substantial finite-sample bias, so label it small-sample development FID, not benchmark FID. Do not lower feature dimensionality to obtain a favorable number. Predeclare PRDC coverage with `nearest_k=5` (requiring enough distinct reference observations), identical feature scaling, Euclidean distance, and self-distance exclusion; report ties/duplicate counts. Coverage is a feature-space neighborhood diagnostic, not proof of support coverage.
5. If uncertainty is reported, distinguish paired bank/reference resampling uncertainty from independent model-training uncertainty. Repeated KID subsets are not independent training replications, and their standard deviation is not a significance test for algorithm superiority. With one training seed, the appendix remains descriptive. Report all metrics, including adverse results, with bank sizes and separately measured evaluation costs; evaluator time is not generator inference time. Do not claim standard video quality from framewise image Inception metrics.

The current task leaves package acquisition, full weight hashing, bank-schema review, protocol freeze, fabricated input tests, and the evaluation job outstanding. No quality result exists from this readiness check.
