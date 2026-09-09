# Restricted Heun quality certificate

The previously missing compact-domain forward-KL integral now has a reviewed interval enclosure. All five fixed Heun grids were integrated with python-flint 0.8.0/Arb; a separate reviewer reran the unchanged 64-call calculation and reproduced every enclosing string and callback count. The analytic tail envelope is evaluated in ball arithmetic. The computations concern the ideal continuous laws of the fixed tilted-normal family and the fixed canonical Gaussian-reference field.

The declared trust includes the earlier global derivative, bijection and tail proof, the nested analytic callback contract, and the pinned numerical libraries. This is a computer-assisted numerical certificate, with an independently reviewed derivation. It has no proof-assistant verification. Floating generated arrays are discrete; these ideal-law KL values are not assertions of continuous-density KL for atomic floating outputs.

The original compact runner and all original saved reports are unchanged. The classification wrapper adds source, version and finite-imaginary checks after independent review; its original source/output and new reviewed output are all retained. These checks leave every target classification unchanged. The independent reviewer checked the full complex tilt ball, propagation of both analytic flags to logarithms, the discrete Jacobian, exact interval averaging, and the tail formula. The outer tilt distribution is uniform on [0.4,0.6] after correct root/index recovery. Its average is not a bound for every individual tilt. See the independent review and the adjacent forward-KL derivation for the proof dependencies.

## Reproduce

Install a platform-compatible python-flint 0.8.0 in an isolated environment. The dependency record describes the actual macOS arm64 wheel and native hashes; it is not a Linux dependency lock. Runtime has no network operations. With that environment's Python, run:

```
python pro10_arb_compact.py --calls 64 --average --seconds 45 --output NEW.json
python pro10_frontier_from_balls.py
```

The first command refuses to replace an existing output. The second only classifies preserved enclosing intervals, without repeating integration, fitting or timing. It parses enclosing strings with Arb, widens their endpoints conservatively, and compares to exact rational targets. The five primary reports bind the same source SHA256. Earlier fixed-tilt/first-average feasibility reports predate source-hash reporting and are explicitly separate; the original earlier source revision was not preserved. Their provenance is incomplete and they are not used by the classification.

## Certified eligibility in the fixed grid

The following interval endpoints are deliberately rounded outward for readability. Values are joint KL in nats for 2,880 residual coordinates. The common root is exact under correct recovery. All retained conditional two-sided interval widths, including the tail allowance, are below 2e-8.

| Heun mathematical stages | Conditional lower | Conditional upper | Expected-risk upper including learning failures |
|---|---:|---:|---:|
| 4 | 0.2237247027 | 0.2237247042 | 0.2237261891 |
| 8 | 0.0320629056 | 0.0320629070 | 0.0320643919 |
| 16 | 0.0021465015 | 0.0021465029 | 0.0021479878 |
| 32 | 0.0001377477 | 0.0001377491 | 0.0001392340 |
| 64 | 0.0000087094 | 0.0000087107 | 0.0000101957 |

Expected-risk lower bounds multiply the conditional lower bounds by 1-p_gamma-p_j; upper bounds add 17280 p_j+17408 p_gamma. The ball-arithmetic allowance is below 0.000001484930. The exact learned sampler's expected-risk upper bound is below 0.000000008250. These use the original finite-family learning assumptions and cover fresh arrays, averaging over training randomness. They do not imply a per-training-set guarantee.

| Expected joint-KL target | Smallest eligible Heun grid | Other boundary outcome |
|---|---:|---|
| 0.1 | 8 | 4 excluded by lower bound |
| 0.01 | 16 | 8 excluded |
| 0.001 | 32 | 16 excluded |
| 0.0001 | 64 | 32 excluded |
| 0.00001 | unresolved | 64 interval crosses target; smaller grids excluded |
| 0.000001 | none in grid | every grid excluded |

Conditioning on correct recovery, 64 stages also meet 0.00001. The exact learned sampler meets all six expected-risk targets under its own bound. An unresolved or empty comparator set supplies no speedup number. A copied analytic decoder ties the proposed exact model and prevents strict dominance over all normalized generators.

The source file's tiny radius around `full_joint_upper_ball` describes an enclosing calculation of an upper bound. It is not the width of the unknown KL interval. `compact_joint_ball`, the tail allowance, and the conservative parsed endpoints supply the actual two-sided enclosure.

## Connection to completed PSC costs

The already published sampler benchmark used one archived fitted model, three source seeds and two batch sizes. Across each corresponding source seed, endpoint Heun8/16/32/64 took respectively 1.70-1.72, 3.05-3.08, 5.74-5.79 and 11.02-11.17 times the optimized exact sampler's batch-one latency. At batch64 the ratios were 2.00-2.05, 3.98-4.15, 8.08-8.35 and 16.24-16.77. These are descriptive latency ratios for the floating implementations of the certified ideal maps. The complete benchmark retains all 17 implementations and all source-specific raw times.

The strongest optimized four-stage comparator remains faster than optimized exact sampling in every measured case. Its quality bound excludes it at these declared targets. That fact makes a restricted quality-matched comparison possible; it does not erase the four-stage latency loss or the exact sampler's greater traced allocation. The costs share root, context, source, chart, validations and archived fit. They exclude a claim of expected runtime over independently retrained models, cross-hardware uniform speed, training-to-quality improvement, or asymptotic superiority.

This advances the restricted solver witness. Unknown dependence learning, native image improvement, video performance, and superiority to equal-dimensional VAE representations remain separate unfinished requirements.
