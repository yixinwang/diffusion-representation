# RMPF-R22 joint two-plane angular-copula boundary

This append-only milestone preserves RMPF-R21 at `9fa16d0fefa9cc1024f394b256c8cac144a60e6c` and changes one scientific layer only: two independent target-plane angular conditioners are replaced by a matched-budget joint two-plane Fourier shear conjugated around the unchanged shared six-bin periodic rational-quadratic map.

The exact normalized all-coordinate no-VAE flow, reversible multiresolution scaffold, source conditioner inputs, two states, rank, data, seeds, likelihood, controls, budgets, development roles, replication seeds, and untouched confirmation are unchanged.

## Exact map

After the unchanged first-harmonic phase offsets, let `x1,x2` be the residual target angles and let

`h_kappa(x) = kappa1 sin(x) + kappa2 cos(x)`.

R22 uses

`H_kappa(x1,x2) = (x1, wrap(x2 - h_kappa(x1)))`

and

`T22 = H_kappa^{-1} o (R x R) o H_kappa`.

The inverse is analytic. Both shears have determinant one, hence

`log |det DT22| = log R'(x1) + log R'(x2 - h_kappa(x1))`.

Zero shear is exactly padded R19; identity splines recover the ordinary parent. R22 and the padded control each use 18 learned degrees, 314 compact bytes, and the same frozen operation accounting.

## Executed result

Deterministic checks passed: maximum round-trip error `1.843e-14`, logdet cancellation `4.758e-13`, Cartesian finite-difference logdet error `2.199e-09`, exact parent fallback, finite density, and exact copied-mechanism equality.

Across seeds 10030--10034, R22 substantially improved exact NLL and the registered dependence diagnostic over padded R19:

- padded-R19 minus R22 NLL/dimension: `0.0642744 [0.0603107, 0.0682380]`;
- padded-R19 minus R22 dependence error: `0.948886 [0.845182, 1.052589]`.

The required proper-energy result failed:

- identity minus R22: `0.0073633 [0.0043397, 0.0103868]`, below the absolute 0.005 LCB gate;
- padded-R19 minus R22: `0.0039248 [0.0002326, 0.0076171]`, below the incremental 0.002 LCB gate;
- compute-matched generic nonlinear shear minus R22: `0.0001491 [-0.0004280, 0.0007262]`, a tie.

A nonpromotable source-only diagnostic then failed in both CIFAR states: incremental held-out logdet gains were negative, pixel proper-energy changes were effectively zero/adverse, and dependence error worsened. UCF remained ineligible with only 39 and 27 source-separated examples.

No real development, replication seed 9401--9404, or untouched confirmation outcome was opened.

## Boundary

The failed premise was that the smallest matched-budget rank-two cross-plane shear would expose at least 0.002 incremental observation-space proper-energy headroom over padded R19. It instead improved likelihood and one dependence statistic without a practical proper-quality gain. Symmetric or multibranch copulas can alias the one-direction circular score. Another shear coefficient, plane order, rank, harmonic, or likelihood selector would repeat this negative.

Local verified state: branch `pro/rmpf-r22-joint-two-plane-copula`, head `78228c4f214ece2676fe2c5f01cd110de4b210a1`, tree `796690f0b2d012b9f1d5c05f608afb20dc331b69`, tag `rmpf-r22-joint-crossplane-boundary-20260830`, 3 tests passed, independent verifier PASS with 152 checks.