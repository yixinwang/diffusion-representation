# RMPF-R10: fused streaming multirate lifting flow

This append-only milestone changes one layer relative to RMPF-R9: the repeated multiresolution trunk is replaced by a two-level exact fused streaming lifting flow while the exact R7 coupled global-copula endpoint, all observed dimensions, Gaussian source, data roles, seeds, controls, and sealed confirmation are preserved.

## Scientific decision

`R10_KNOWN_TRUTH_PASS_SYSTEMS_PASS_OPENED_QUALITY_FAIL_C3_FAIL_C4_ACTIVE_CONFIRMATION_SEALED`

R10 passes deterministic inverse/Jacobian checks, the five-seed hidden-copula known-truth gate, and all frozen full-flow and positive-rate-latent systems gates. This legally opened development quality. On five CIFAR/UCF development seeds, however, R7 has no attributable proper-energy or global-dependence gain and the complete candidate loses the full-flow and positive-rate latent-flow quality frontiers. Scientific promotion is false.

The one-layer C3 child replaces only diagonal RGB residual scaling by exact 3x3 Cholesky whitening. Its five-seed Gaussian known truth passes: predicted NLL gain 0.355491 nat/residual dimension; observed 0.357075 [0.353931,0.360218]. In the real seed-9400 smoke it improves NLL by 0.687029 on images and 0.663931 on video, but worsens proper energy by 0.011529 and 0.000681, leaves R7 unattributed, and exceeds both frozen fit budgets. The result is reproduced exactly.

One child remains frozen but unexecuted: an exact radial shared-scale transport on the color-whitened residual. Confirmation is absent and unopened.

## Key numbers

- Parent R9 head preserved as ancestor: `0418b2576863133133ebda1f8895538913f916fd`.
- Local final branch: `pro/rmpf-r10-fused-multirate-lifting`.
- Local final commit: `3baae18226f1b19e9b9aa6392439ddad42595126`.
- Hidden-copula coordinatewise-control minus R10 NLL: 0.0107923 [0.0107017,0.0108829].
- R10 systems, image: fit 1.2691 s, batch 0.05873 s, single 0.00200 s, 22,185 stored bytes, 42,205,184 batch FLOPs.
- R10 systems, video: fit 1.2885 s, batch 0.02659 s, single 0.00362 s, 102,245 stored bytes, 22,783,744 batch FLOPs.
- Image energy: R10 20.27708, full flow 19.23686, positive-rate latent flow 19.18998.
- Video energy: R10 48.97878, full flow 48.63753, positive-rate latent flow 47.49402.
- R7 attribution: image local-minus-candidate energy -0.00000613 [-0.00001957,0.00000730]; video exactly 0 [0,0].
- Confirmation opened: false.

The unrestricted equivalence control ties exactly; no universal dominance claim is made.

## Contents

- `THEORY_AND_FAILURE.md`: exact inverse, log-Jacobian, complexity, finite mechanism, and failure boundary.
- `reproduce_known_truth.py`: NumPy-only exact hidden-copula reproduction for seeds 9600-9604.
- `MACHINE_VERDICT.json`: compact machine-readable decision and immutable hashes.
