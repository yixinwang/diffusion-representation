# Reproduction audit before LIFT-FM development

Date: 2026-08-26  
Repository base: `e00a034594e39f110c7f148502d5106926c46097`

## QALT strict oracle

Reconstructed and executed the registered oracle runner from the base commit. All gates passed.

- Euler fiber KL gaps for K = 4, 10, 20, 50: `0.0361648201`, `0.00642797995`, `0.00166718724`, `0.000272823036`.
- Correct pooled-vs-coordinatewise gain at n = 512: `0.0106015413`, paired 95% interval `[0.00999721098, 0.0112268850]`.
- Misspecified pooling contrast at n = 512: `-1.32849168`, interval `[-1.32907442, -1.32790394]`.
- Exact exponential, structural-split, and same-information pooled controls tie QALT.
- Nonlinear triangular decoder inverse error: `8.88e-16`.

Interpretation: the algebra is reproduced, but the strict quality gap is only against the registered inexact or restricted comparator. It vanishes when the full baseline copies the exact fiber or pooling rule.

## QALT multiscale Stage A

Executed the 30-seed confirmation (seeds 800--829) for synthetic nonlinear image/video tensors. Non-timing metrics reproduce the committed summary; local CPU timing differs, as expected.

- Image: QALT-minus-oracle NLL `0.000121314`; diagonal-minus-QALT `0.101633`; coarse-only-minus-QALT `0.105406`; Euler-minus-QALT `0.000146536`; memory ratio `0.75`.
- Video: QALT-minus-oracle NLL `0.0000804055`; diagonal-minus-QALT `0.106124`; coarse-only-minus-QALT `0.114061`; Euler-minus-QALT `0.0000965651`; memory ratio `0.625`.
- All 16 registered Holm-adjusted gates passed.

Interpretation: this is controlled generated data with a fixed Haar chart and known conditional-mixture structure. It does not establish realistic image/video endpoint generation.

## Observed CIFAR B1 evidence

The base commit contains a preserved five-seed development result, but the local environment does not contain the CIFAR archive, so it was not independently rerun. The official test batch was not used. The preserved validation evidence reports:

- four-component conditional marginal vs one Gaussian: `0.125954` nat/detail coefficient;
- coarse conditioning gain: `0.0043905`;
- scalar route selected depth nine in every seed;
- omitted within-band cross-color dependence caused roughly `0.08`, `0.071`, and `0.023` nat/detail-channel regret.

The registered next repair is the joint RGB block fiber already present in the base source and unit-tested there. No full diffusion, VAE, FID, or endpoint claim follows from B1.

## FIQ-FM audit

FIQ-FM is a fixed-aligned-chart special case with cross-fitted residual grouping and a one-shot block-Cholesky fiber. Its README explicitly marks historical digits/synthetic tables as unconfirmed because raw artifacts are missing or schemas no longer match. It therefore supplies implementation context, not reproduced empirical evidence.
