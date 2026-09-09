# Independent certificate-package review

The package review passes for the stated restricted mathematical and descriptive-cost claims. No integration, fitting, timing, PSC work, or original-file changes were performed.

All 18 manifest entries match their declared byte sizes and SHA256 hashes. Enumeration is exact after excluding the manifest itself. All 16 preserved counterparts available in `work/`, including the original wrapper under its `.py.txt` name, are byte-identical. The reviewed wrapper is intentionally different from the original wrapper. Its independently repeated classification output is exactly equal to `pro10_frontier_reviewed.json`; the new output is `work/pro10_frontier_independent_reclassification.json`.

The wrapper checks each report's successful scope, grid, cutoff, recorded library version, and original runner hash. It requires a finite imaginary ball containing zero. Its positive-lower, ordered-upper and finite-width inequalities reject nonfinite real bounds as well. Parsing retained enclosing strings, taking outward endpoints, multiplying the lower endpoint by 1-p_gamma-p_j, and adding 17280 p_j+17408 p_gamma to the upper endpoint are conservative under the earlier finite-family learning assumptions. The probability bounds and the separate exact-sampler risk expression match the reviewed derivation. They yield the displayed excluded/eligible/unresolved classifications without using rounded numbers to classify.

All 15 displayed numerical KL endpoints in the README are rounded outward from the retained conditional/expected enclosures. The displayed learning-failure allowance and exact-sampler bound are also outward upper bounds. The explanation correctly distinguishes the ball surrounding a computed upper endpoint from the wider enclosure of the unknown KL, and separates average-over-tilt claims from pointwise tilt claims.

I recomputed ratios of each arm's median primary latency to the same seed/batch optimized exact median from `psc_tilted_cost/primary_timings.json`. Raw min/max across the three seeds are:

| Batch | Stages | Minimum ratio | Maximum ratio |
|---|---:|---:|---:|
|1|4|0.97542055|0.98294715|
|1|8|1.70402352|1.71209899|
|1|16|3.05589257|3.07348573|
|1|32|5.74275691|5.78779927|
|1|64|11.02887475|11.16098941|
|64|4|0.92150189|0.93170224|
|64|8|2.00394556|2.04944112|
|64|16|3.98055947|4.14408881|
|64|32|8.08646172|8.34695785|
|64|64|16.24867903|16.76697094|

Every README range contains the corresponding raw ratios. The four-stage loss, traced-allocation limitation, fixed archived-fit scope, and copied-decoder tie are retained. No broad latent-model, representation, native-image, video, asymptotic, or training-efficiency claim appears.

Operational qualifications: the wrapper uses Python assertions for validation, so reproduction should use ordinary Python without `-O`; it checks the reports' recorded version, rather than independently pinning a newly installed runtime binary. Dependency trust remains as explicitly described in the README. These do not alter the verified package results. The preserved original runner's sibling dependency lookup also means the isolated directory must be present or python-flint installed in the invoking environment, as the README states.
