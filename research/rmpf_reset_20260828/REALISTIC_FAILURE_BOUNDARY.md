# R4 structured coarse dynamics: quality recovery without attributable or measured-efficiency success

Date: 2026-08-28. The CIFAR train/validation roles were opened; confirmation pixels and UCF confirmation clips remained sealed.

## Frozen outcome

The full-rank time-modulated FWHT coarse model passed its finite affine known-truth check at `3.56e-25` MSE. On the frozen CIFAR validation identities:

- R2 hierarchical no-global energy: `19.52038013`.
- Selected structured no-global, rank 192: `19.42433276`.
- Coarse recovery: `0.09604737`, exceeding the frozen `0.00786` margin.
- Strongest parameter/FLOP-matched RFF width 192: `19.52988726`.
- Structural gain: `0.10555450`, exceeding the frozen `0.002` margin.
- Structured plus unchanged stable global A: `19.42429194`.
- Attributable A gain: `0.00004082`, below the frozen `0.001` margin.
- Retained full-pixel RFF: `19.48892295`.
- Retained positive-rate PPCA latent flow: `19.50581707`.

Thus the time-state coarse basis gives a large opened-validation quality improvement, but the registered global-local component A does not cause it.

## Systems diagnostic, no retuning

Against the retained full-pixel model:

| method | batch latency ratio | single latency ratio | peak ratio | workspace ratio | stored-byte ratio | FLOP ratio |
|---|---:|---:|---:|---:|---:|---:|
| structured no-global | 3.9075 | 7.5792 | 1.5171 | 3.1088 | 0.8464 | 0.9668 |
| structured plus A | 6.0176 | 28.0102 | 1.6228 | 3.1088 | 1.4985 | 0.9999 |

The implementation retains all dimensions but computes FWHT/state and hierarchical parent context in unfused NumPy kernels. More importantly, even the analytic operation count leaves only 3.3% FLOP headroom without A and essentially zero with A. Kernel fusion cannot create the registered large latent-frontier gap from that operation budget.

## First failed layer and rival diagnosis

The first scientific failure is attribution: the recovered quality comes from the structured time-state coarse basis, not periodic global detail communication. This supports rival R1 (multiscale/structured inductive bias) rather than the proposed compute-allocation mechanism. The systems failure is downstream and independent: the full-rank coarse basis collapses the compute path back to full-state cost, and A erases the remaining storage advantage.

The prior fixes failed for distinct reasons:

1. stable parity communication solved the exact adversary but did not generalize;
2. actual Haar parents improved velocity MSE weakly but did not transfer to endpoint score;
3. cross-fitted bin gains were negative in both image and video;
4. increasing NFE amplified fitted-field bias;
5. structured coarse dynamics recovered quality only by using maximal rank, while A remained negligible and measured efficiency reversed.

Per the frozen R4 stop rule, this establishes the no-feasible-small-revision boundary for the RMPF family. No confirmation split is opened and no additional rank, threshold, NFE, local feature, or mixer revision is allowed in this lane.
