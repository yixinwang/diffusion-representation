# HSRF failure ledger

## R1: full output map

- Reproduced exactly from source/configuration/data hashes.
- Global diagnostic error fell from roughly 0.33 to roughly 0.026.
- Exact likelihood worsened versus the zero-global control and improved over the local control by only 0.00139 nat/dimension, below the frozen 0.02 margin.
- First failed layer: fitting an unrestricted `m x q` output map paid the finite `q m/(n-m-1)` estimation term.

## R2: reduced output rank

- Only the output-map factorization changed.
- Validation selected rank four.
- Zero-minus-candidate NLL improved to 0.00299 nat/dimension.
- Local-minus-candidate remained 0.00141, below the frozen 0.002 margin.
- One seed failed the 90% global-error-reduction gate.
- Positive-rate PPCA remained better in NLL.
- First failed layer moved from finite output variance to omitted distributional structure.

## R3: conditional log scale

- Only conditional variance modeling changed.
- Mean-only minus affine NLL was approximately `-2.15e-6`.
- A held-out rank/cap sweep found no positive scale headroom.
- A separate synthetic conditional-variance teacher yielded a positive gain, so the implementation itself was not the failure.
- The affine family was retired.

## Genuine development diagnostic

- Used fresh CIFAR-10 train/validation/fiber/development arrays and source-group-separated UCF clips.
- Confirmation files were not extracted.
- HSRF reduced stored model bytes, but failed joint exact-NLL, feature-geometry, precision/recall, training-time, latency, and peak-RSS gates.
- Video HSRF recall was zero.
- Positive-rate PPCA was the strongest joint quality-efficiency control on CIFAR and had much better feature/recall metrics on video.

## Full-control convergence audit

- Extending optimization improved the charted full controls, confirming that some original full-flow horizons were inadequate.
- This did not reverse the HSRF decision because HSRF still lost to exact diagonal and positive-rate controls and failed systems margins.

## Stop boundary

The first failed layer is the joint local/global conditional distribution family and stage alignment. Adding more interaction rank, conditional scales, or local stages would erase the low-cost advantage and repeat the MCQF/RMPF storage, memory, and latency boundary. Confirmation remains sealed. Reopening requires a new exact transport family and a new preregistered finite advantage; post-hoc rank, feature, or threshold changes are prohibited.