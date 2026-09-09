# Independent captured-scalar and full-model numerical audit

No fitting, optimizer update, repair-quality evaluation, or additional PSC job was performed in this audit. Inputs are authenticated saved captures; source/payload/checkpoint authentication is separately recorded by psc_audit in completed-authentication.json. Authoritative arithmetic report: independent_numerics_with_cpu_coordinate.json.

## Reproduced GPU mechanism, with node limitation

Both saved GPU replays on v007 contain the same layer2 encode-order locals SHA `fafbe63887efb9cbebad73fce43712b87f50c9221ea4e1fee26853039cbfe395`. Exactly one inside-bin `good_theta` predicate fails, coordinate [1,22,5,1]. Captured x=-0.08074430376291275 is strictly below its right y-knot -0.08074426651000977. Captured normalized eta=.9999999403953552 is in [0,1], but the old quadratic computation returns theta=1.0000001192092896 and maps beyond the right x-knot. This is actual numerical guard attribution in the reconstructed v007 replay, not a claim that original v024's unsaved internal tensors were identical.

Independent100-digit monotone bisection of the rational quadratic on the captured knots/slopes gives theta=.9999999514812142599345, inverse value=.0301584828908283942793, inverse logdet=-.0107644954798128878796. The ideal inverse remains inside its bin. The old captured mapped value error is1.56063e-7. Independently implementing reflected scalar arithmetic locally on those captured float32 knots gives inverse value .030158482491970062, absolute error3.98858e-10; logdet error1.20970e-8. The direct right distance preserves a complement4.85187854e-8 instead of evaluating a rounded near-one root. These computations are ordinary high precision, not certified intervals; the independent scalar execution is CPU arithmetic, whereas the saved full candidate GPU tensors provide GPU evidence.

The normalized rational quadratic is T(t)=yl+h*(delta*t²+dl*t*(1-t))/Q, Q=delta*(t²+(1-t)²)+(dl+dr)*t*(1-t). Its derivative is delta²*[dr*t²+2delta*t*(1-t)+dl*(1-t)²]/Q²>0. Bisection therefore identifies the unique inverse. Reflection swaps endpoint derivatives and solves the direct right ordinate distance for1-t; its equivalence follows by substituting1-t into1-(T-yl)/h. The compact actual_failed_gpu_scalar.json preserves raw parameters, captured scalar terms, provenance hashes, and both calculations; check_failed_scalar.py reruns the independent bisection without loading observations or a model.

The full CPU replay does **not** reproduce this old-kernel invalid theta. At the same index its x=-.08073683083057404 differs by~7.473e-6, lies on the other side of the nearby knot, and selects the adjacent bin. Its valid theta is1.0334682e-5. Thus neither index matching nor equal checkpoints implies equal intermediate CPU/GPU values. Earlier separately captured CPU evidence, if used, must retain its own execution mode/source provenance rather than being conflated with this full CPU capture.

## Independently recomputed complete3072 gates

| Saved scope | Source RT maximum | Source LD cancellation | Observed RT maximum | Observed LD cancellation |
|---|---:|---:|---:|---:|
| CPU | 6.15119934e-5 | .00146484375 | 1.09672546e-5 | .0009765625 |
| GPU | 6.52670860e-5 | .00146484375 | 1.38282776e-5 | .001953125 |

All are below unchanged .001 coordinate/.01 full-array LD thresholds and exactly match reported maxima. Saved sources have shape8x3072 and observed logits32x3x32x32. Every numerical-bank and joint-forward tensor is finite; independently reconstructed complete joint scalar losses are finite. Saved reloaded generator values and LD match bitwise. All152 expected parameter-gradient tensors, comprising104 analysis and48 residual tensors (517476 scalar elements), are present and finite; their names exactly equal the declared expected analysis-plus-residual union. All four retained input gradients (observed logits, coarse, residual, isolated frozen-root input) are finite and nonzero.

The frozen-root parameter flags and model state equality remain source-reviewed runtime assertions bound to authenticated status/source; gradient arrays alone cannot reconstruct requires_grad settings or prove pre/post state identity. This distinction is preserved in the machine report. No optimizer state was changed by the diagnostic.

## Preserved audit failure and source identity

The first independent checker incorrectly required regenerating the saved CPU Gaussian bank with the local Torch2.14/macOS build to be bitwise identical to the PSC Torch2.10 build. This assertion failed while every other complete-model check passed. Its source/stdout/stderr are preserved under independent_attempt1. Seed identity is not a cross-version RNG bitstream guarantee. The corrected checker retains the mismatch as a reported limitation and uses the actual saved banks, not replacement draws: CPU and GPU raw Gaussian bytes match exactly, SHA256 `678cb83aa6442b431128c0390043e0e1ba671706131d5e66bbf74851d6266a12`. Their serialized bank hashes also match each scope's recorded hash. No numerical threshold was changed, no draw was replaced, and no additional model fitting/evaluation was undertaken.

This audit supports bounded diagnostic qualification on the saved banks and scopes. It does not establish global floating invertibility, tail stability, exactness for all inputs, or generation superiority, and does not erase the original failed run or earlier adverse quality results.
