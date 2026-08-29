# RMPF-R10-C4G fixed-batch quality boundary

This append-only milestone records the frozen quality-only development evaluation of the exact normalized all-coordinate no-VAE C4G flow after its density and fixed-batch systems gates passed.

## Scientific decision

`DEVELOPMENT_QUALITY_FAIL | R7_ATTRIBUTION_FAIL | NO_SELECTION_HEADROOM | CONFIRMATION_SEALED`

The scientific law was unchanged: exact FSMLF lifting, six RGB Cholesky residual maps, sixty conditional log-radius rational-quadratic splines with identity fallback, and the exact R7 coupled global-copula endpoint. Five paired seeds (9400-9404) used identical data roles, conditions, standard-normal source identities, NFE 16, evaluators, and validation-only baseline search.

### CIFAR-10 32x32

- C4G proper energy: 20.277746.
- Full-flow energy: 20.251464.
- Positive-rate latent-flow energy: 20.234181.
- R7 attributable energy, radial-only minus C4G: 0.0000136 [-0.0000475, 0.0000746].
- R7 attributable dependence: -0.019881 [-0.070789, 0.031026].
- C4G minus full-flow energy: 0.026282 [-0.016370, 0.068934], failing the 0.02 noninferiority margin.
- C4G minus latent-flow energy: 0.043565 [-0.015418, 0.102547], also failing.
- Feature, precision/recall, and zero-catastrophe gates failed.

### Source-separated UCF101 clips

- C4G energy: 48.990836.
- Full-flow energy: 49.897289.
- Positive-rate latent-flow energy: 48.688907.
- C4G significantly beat selected full flow: candidate-minus-full -0.906453 [-1.597221, -0.215685].
- It did not establish latent-flow noninferiority: 0.301929 [-0.284741, 0.888599] under a 0.10 margin.
- R7 attributable energy was 0.0000092 [-0.0000164, 0.0000348]; dependence attribution was -0.005404 [-0.020406, 0.009599].
- Feature, recall, component-attribution, and zero-catastrophe gates failed.

The radial layer retained a large exact-NLL improvement over color-only (0.235696 image; 0.343803 video nat/visible-dimension), but R7 added essentially zero NLL or downstream quality headroom. This preserves the likelihood-versus-proper-quality mismatch.

## Diagnostic boundary

A post-result, deliberately nonpromotable oracle enumerated all 6^5 = 7,776 seed-wise selections among the six already evaluated C3/C4/C4G laws in each domain. No image or video vector passed all 15 unchanged scientific gates. Thus calibration, endpoint thresholding, or selection cannot rescue this finite family.

The first remaining failed layer is approximation/representation: angular, cross-site, semantic, or stage-aligned dependence lies outside conditional radial marginals plus one endpoint R7 coupling.

## Systems and seals

The inherited systems pass remains valid: fixed-batch candidate/full latency was approximately 0.675 for images and 0.527 for video, with much lower stored bytes and FLOPs. The quality result therefore does not arise from systems failure.

Untouched confirmation was absent and unopened. All prior QALT, FiberLift, FIQ-FM, MCQF, RMPF/RMGL/HSRF, and R1-R10/C3/C4/C4F/C4G evidence and failures remain unchanged.
