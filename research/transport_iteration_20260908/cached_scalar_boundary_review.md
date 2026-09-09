# Cached scalar boundary qualification

The integrated-linear derivative heights normalize exactly in real arithmetic: with K bins, fixed endpoint heights one and interior sum K−1, the trapezoidal integral over [-4,4] is eight. The analytic map is normalized, increasing and joins identity tails with derivative one. This does not imply an exact floating-point C1 or strict-interior guarantee.

The deterministic repro `cached_scalar_boundary_check.py` uses 4,096 rows of seven logits with scale 30, seed 992, and nextafter points immediately inside both ±4 boundaries. Its saved `cached_scalar_boundary_results.json` records actual crossings. The positive-endpoint float32 case has 126 outputs strictly beyond four, maximum overshoot 9.537e−7 and maximum roundtrip error 1.192e−6. Float64 has 126 strict crossings, maximum overshoot 1.777e−15 and roundtrip error 2.221e−15. Both negative-endpoint cases have zero observed overshoot/error. All transforms return valid under the declared arithmetic tolerances.

This is cumulative-knot rounding at the positive endpoint. A rounded output may enter the identity-tail inverse branch; the resulting discrepancy remains within the fixed numerical gate. The test does not clip or repair inputs/outputs, and records this limitation rather than declaring a strict floating-point theorem. The added targeted regression uses fixed maximum roundtrip tolerances 2e−5 for float32 and 5e−14 for float64; both pass. These are numerical tolerances, not approximation or learned-quality claims. Similar finite-precision knot qualifications apply to other spline implementations.

No production algorithm was changed. Only the new deterministic boundary regression was run; the existing full suite was not repeated. No data, fitting, GPU job or commit occurred.
