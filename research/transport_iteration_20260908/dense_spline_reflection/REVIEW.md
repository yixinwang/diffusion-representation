# Dense inverse boundary audit and reflected scratch prototype

No production file or trained state was modified. The native failure has not yet been attributed to a specific tensor by this audit. `original_dense_spline.py` is the unchanged source snapshot; `boundary_failure_01.json` contains a **fabricated** failure, including complete raw parameters, generated with the recorded seed. Earlier scratch prototypes and their exact receipts are preserved in `attempt1_two_roots` and `attempt2_midpoint_selector`.

## Confirmed fabricated defect mechanism

Among1024 ordinary float32 scalar splines, inputs one nextafter step below selected interior right knots gave21 out-of-bin inverse roots, with maximum `1.0000001192092896`. For the retained first case, input `.8682846426963806` is less than its right knot `.8682847023010254`, but `(input-y_left)/height` rounds to1. The standard positive quadratic formula then returns theta one ULP above1. The dense kernel correctly rejects this result; the legacy scalar implementation does not enforce the same root-range gate, so merely obtaining finite legacy outputs is not a valid repair criterion.

There is no justification for clipping theta or treating `eta==1` as an exact input endpoint: in this case it is demonstrably an interior input. The input/parameter law and exact-real spline remain well-defined. The defect is in evaluating a near-endpoint inverse using a long distance whose subtraction/normalization loses the short distance.

## Reflected inverse, exact-real equivalence

For an RQS bin with positive width/height, secant delta and endpoint derivatives dl,dr, write its input fraction theta and complement q=1-theta. Reversing both axes replaces normalized output eta by its complement and swaps dl/dr. Thus solve the same quadratic for q using **direct** `(y_right-input)/height`, not `1-eta`, and reconstruct the source as `x_right-q*width`. Keep the accurate short distance q separately when computing `theta*q` and the derivative numerator.

Monotonicity locates theta=.5 at normalized ordinate

`eta_mid = (delta+dl)/(2*delta+dl+dr)`.

The current prototype selects the left formula if eta_left<=eta_mid, otherwise the reflected right formula. It computes only the selected root; there is no preliminary potentially invalid root. Positive rescaling by max(delta,dl,dr) keeps the selector's sums finite for finite inputs without changing the mathematical ratio. In the negative-b quadratic branch it uses `a=delta-b`, avoiding the separately cancellation-prone equivalent expression. This is an algebraic reformulation, not a relaxed validity gate. Both theta and q are checked finite and within[0,1]. The density derivative remains the same RQS derivative, evaluated using the retained short distance.

The formula is smooth in exact arithmetic across the branch threshold because both branches represent the same inverse. Its branch choice therefore does not introduce a different exact-real model. Floating implementations still require explicit numerical and gradient validation; this is not a proof of bitwise equivalence or global rounded invertibility.

## Fabricated checks and limits

The current checks pass the saved failure,3712 arguments across128 raw-parameter templates at all knot neighbors/exact knots plus tails, nondecreasing output order, exact tail behavior, both output and logdet gradients with respect to every input/raw parameter group, and a midpoint-branch round trip. Float64 autograd finite-difference gradcheck passes; the float32 versus float64 raw-recomputed comparison has maximum gradient difference about2.14e-6 and output difference about4.39e-7.

An independent Decimal80 inverse uses bisection of the forward rational function on the **realized float32 knots and derivatives interpreted as exact numbers**. The saved case's value error is8.55e-9 against this reference (output ULP1.19e-7), and its logdet error is6.05e-7. This is a high-precision diagnostic, not interval arithmetic. Recomputing all raw parameters in float64 changes softmax/knots slightly and is separately labeled; it is not an exact replay of float32 intermediates.

The exact raw-parameter/grid bank and current outputs are saved as `adversarial_boundary_bank.npz` and `adversarial_boundary_outputs.npz`. `results.json` records source hashes and all outcomes. Extreme finite raw derivatives1e30 still overflow the inverse discriminant and are rejected. This unresolved case is retained: the scratch repair is not advertised as a universal finite-float32 inverse. Positively rescaling the whole quadratic might address that distinct overflow, but has not been adopted or claimed here.

## Required native diagnosis

Before attributing the actual failure, preserve the failing stage/seed, accepted-update count, exact model/optimizer state, fitting-index sequence or failed batch IDs, source commit, device/dtype/TF32/runtime, and failing coarse/residual batch. A failed forward call occurs before an optimizer step, so its saved weights should be the last accepted state; confirm this from the trace. If indices were not explicitly saved, reconstruct exactly `updates+1` batches from the registered family RNG and verify the recorded order hash, including the failed attempt.

For each inverse layer, save the actual layer input, coarse context, mask, conditioner raw tensor, and all failure-category flags. At offending coordinates retain both knots, selected index, actual rounded widths/heights, dl/dr, delta, eta, a/b/c, discriminant, square root, selected denominator, theta, derivative numerator/denominator, mapped value and logdet. Mark whether the coordinate is inside the spline domain and whether the coupling mask ultimately changes it. Distinguish a root-range error from nonfinite raw/affine values, degenerate bins, discriminant/denominator overflow, accumulated-logdet failure or failures only in a masked coordinate. The outer dense flow computes and checks even masked entries; changing that policy would be a separate disclosed numerical implementation change.

GPU-realized scalar inputs and raw parameters are more diagnostic than CPU recomputation of the full conditioner, which may move a near-boundary value by an ULP. Two useful references are high-precision arithmetic on saved realized knots, and full float64 recomputation from raw parameters; label them separately. Do not infer the actual native cause from this fabricated reproduction alone.

## Comparator fairness if a repair is adopted

Preserve the failed native attempt unchanged. Authenticate and freeze the revised backend with specific regression cases, full-model source/inverse/logdet checks and gradient comparisons before another run. A backend change cannot be hidden as an unchanged replay, a larger tolerance, removal of the range guard, or a legacy fallback that accepts the same invalid root. The architecture and ideal density may be unchanged, but arithmetic, runtime and failure behavior have changed. Charge the revised arithmetic/setup under the same budgets and register the new attempt explicitly. Resuming only RQS with extra free fitting time, selecting a repair using repair-set quality, or discarding the original failed comparator would invalidate the proposed fair comparison. No quality or speed advantage over FM/latent models follows from a numerical backend failure.
