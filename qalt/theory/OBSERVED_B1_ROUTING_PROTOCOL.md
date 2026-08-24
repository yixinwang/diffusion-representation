# Frozen observed B1 routing protocol

Frozen at repository state after the routing theorem and data-integrity implementation, before any observed Haar coefficient, conditional fit, or validation score was computed. This is an image-first development protocol. UCF101-subset remains metadata-only until the image implementation and deterministic video decode are separately audited.

## Question and claim boundary

The study asks whether real CIFAR-10 Haar-detail channels can replace sequential autoregressive conditionals by parallel one-shot conditionals while retaining a source-level held-out forward-log-score certificate. It does not test an active/coarse generator, FID, a VAE, full diffusion, or endpoint generation. A passing study licenses a neural endpoint experiment; it does not establish endpoint superiority.

## Data and statistical units

Use only the five canonical CIFAR-10 training batches and the committed manifest specification. The deterministic 45,000-example fitting set trains every parameter. The disjoint 5,000-example validation set is the calibration sample. One image is one independent cluster; coefficients are averaged within image and are never treated as independent replicates. The official test batch remains guarded and unread.

For each image and seed, draw one paired uniform dequantization tensor from a counter-based key `(record_index, dequantization_seed)`, add it to the uint8 values, and divide by 256. Every method receives the identical tensor. Apply the existing one-level orthonormal 2D Haar transform. The coarse field has three channels and the detail field is ordered as nine `(band,color)` channels. All coefficients remain in the representation.

## Proper joint candidates

Every candidate uses the same topological detail order. A one-shot kernel may ignore previous details; it remains a proper kernel on the common DAG.

For each spatial site, the shared coarse feature map contains an intercept, the three coarse coefficients, their squares, their three pairwise products, and their hyperbolic tangents. A ridge location model with fixed penalty `1e-3` is fitted separately for each detail channel. Locations are clipped to the mathematically valid detail interval `[-1,1]` for every method.

- `O4`: nine parallel one-shot conditionals. Locations use only coarse features. Residuals use a four-component zero-centered Gaussian scale mixture within each of four coarse-energy strata.
- `I4`: full autoregressive comparator. Channel `v` adds data-parent detail channels `1,...,v-1` to the same coarse location features; its residual family and energy strata otherwise match `O4`.
- Routed candidates: for every subset of the nine channels, use `I4` on that subset and `O4` elsewhere. Evaluation uses held-out data parents. Generation samples all `O4` channels first and then the routed `I4` channels in topological order; a routed `I4` kernel may use earlier generated channels.
- `O1`: fixed conditional diagonal Gaussian with the identical `O4` location features.
- `O4-unconditional`: four-component mixture with the identical location features but no coarse-energy stratification.
- `O8`: eight-component same-information parallel control.
- `I8`: eight-component autoregressive overcapacity control.
- Exact copy controls call the same fitted density path under two labels and must tie bitwise.

Mixture weights are at least `1e-4`; scales are constrained to `[0.05,2]`. Expectation--maximization uses fixed initialization quantiles, at most 100 iterations, and relative tolerance `1e-7`. Fit the mixtures on a counter-based maximum of 1,000,000 fitting residuals per channel, shared across component-count comparisons. Report every parameter count and selected sample identity hash.

## Routing certificate and compute estimand

For validation image `i` and route `r`, let

`D_ir = {-log q_r(X_i)+log q_I4(X_i)}/2304`,

where 2304 is the number of detail coefficients. Models are frozen before these values are read. For each route, compute its sample mean and pairwise sample variance. Analytic coefficient/support, location, weight, and scale constraints provide a deterministic interval `[a_r,b_r]` for `D_ir`; no validation extrema define this interval.

Apply Maurer and Pontil's 2009 empirical Bernstein bound after mapping the interval to `[0,1]`. With `M=512` routes and familywise error `alpha=0.05`,

`U_r = mean(D_r) + sqrt{2 V_n(D_r) log(2M/alpha)/n} + 7 (b_r-a_r) log(2M/alpha)/{3(n-1)}`.

This is Corollary 5 obtained from their Theorem 4 by a finite-class union bound and affine rescaling. The route is certified when `U_r <= epsilon=0.01` nat per detail coefficient. Choose the certified route with the fewest autoregressive channels, breaking ties lexicographically. If no route beats the full comparator's depth, abstain to `I4`.

The analytic sequential-depth proxy is `number_of_I4_channels + indicator(any O4 channel)`: one for the all-parallel route and nine for the full autoregressive route. Report CPU/GPU latency separately; do not infer wall-clock speed from depth. A later latent-efficiency claim requires a matched latent route in the candidate set and measured cost bounds.

## Development gates and diagnostics

Across the 5,000 validation images:

1. `O4` beats `O1` by a paired image-bootstrap 95% lower endpoint above `0.01` nat per detail coefficient.
2. `O4` beats `O4-unconditional` by a lower endpoint above zero.
3. `O4` is within `0.01` nat per detail coefficient of `O8` by paired two-one-sided equivalence tests.
4. The selected route passes the simultaneous empirical Bernstein certificate and has depth below 9. If this gate fails, retain the abstention and use its decomposition to revise the conditional family or theorem; do not change `epsilon`.
5. The exact copy ties bitwise, all scores are finite, Haar round-trip error is below `1e-6`, input/dequantization hashes match across methods, and no official test record is deserialized.

Report per-channel regret, radial probability-integral-transform calibration, angular residual dependence, cross-channel residual-energy correlations, and coefficient heatmaps. Five development seeds `1100..1104` vary only fitting subsampling and dequantization. They quantify algorithmic variability, not five independent datasets. No confirmation seed or official test datum is accessed in B1 development.

## Stop and revision rule

Any failed gate is a preserved result. A large `O4-I4` gap implicates missing cross-detail dependence; failure against `O8` implicates inadequate marginal tail shape; failure of the analytic certificate with a small empirical gap implicates bound looseness rather than model quality. These three signatures trigger different preregistered revisions. No threshold, split, or candidate is changed after validation scores are observed.
