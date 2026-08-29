# RMPF-R11: exact angular coupling and finite-family boundary

This append-only milestone records the diagnosis-driven child of the exact RMPF-R10-C4G systems-pass model. It does not modify earlier QALT, FiberLift, FIQ-FM, MCQF, RMPF/RMGL/HSRF, R1–R10, C3/C4/C4F/C4G, data, split, result, failure, hash, tag, or PR artifacts.

## Frozen change

Only the angular/cross-site dependence layer changed. The exact FSMLF lifting trunk, RGB Cholesky maps, conditional radial splines, R7 coupled endpoint, standard Gaussian prior, all visible coordinates, exact change-of-variables likelihood, data roles, seeds, controls, and systems limits were preserved.

For frozen trunk coordinates `(a,b)`, the child defines a source-frozen binary coarse state

`S(a)=1{w^T(a-a_bar)>tau}`

and one symmetric positive definite, determinant-one matrix `A_{loS}` per level, orientation, and state. The data-to-base map is

`c_los=A_{loS(a)}^{-1} b_los`,

with exact inverse `b_los=A_{loS(a)} c_los`. Because `a` is unchanged and `det(A)=1`, this triangular layer adds exactly zero log-Jacobian and preserves the normalized all-coordinate no-VAE law.

## Executed evidence

The C4G parent was reproduced from head `37105959db1b1b3a0faa294a4e2623cb22f07bf1` and retained raw/contrast hashes. Across known-truth seeds 9800–9804, R11 achieved:

- exact-NLL gain per visible dimension: `0.1462682174 [0.1451759007, 0.1473605340]`;
- proper-energy gain: `0.0164168869 [0.0125867572, 0.0202470165]`;
- maximum round trip: `1.7763568394002505e-15`;
- hidden angular parity after transformation: `1.0`, preserving the impossibility arm.

The final fused C2 implementation passed every frozen systems gate:

- image fit `0.986507 s`, batch `0.029178 s`, stored `43,526 B`;
- video fit `0.776796 s`, batch `0.021791 s`, stored `132,808 B`;
- paired log-Jacobian cancellation was zero.

The preregistered cheapest development falsifier then used only seed 9400 in each domain. R11 reduced the inherited dependence diagnostic but worsened registered proper energy:

- CIFAR: dependence gain `+0.0464977`, energy gain `-0.00203939`; it missed both flow energy frontiers and the feature frontier.
- UCF: dependence gain `+0.0121198`, energy gain `-0.0972168`; feature, recall, temporal-SWD, and support gates failed.

No smoke route passed, so seeds 9401–9404 and untouched confirmation were not opened. A byte/scientific replay produced identical verdicts.

## Diagnosis children

A rank-eight exact state-conditioned circular-plane child doubled affected mass and achieved NLL gain `0.0269999 [0.0268436, 0.0271562]`, yet proper-energy gain remained `-0.00006797 [-0.0023653, 0.0022294]`.

An independent exact axial-spherical C3 child changed only fallback-selection granularity. It retained proper-energy gain `0.0063305 [0.0060108, 0.0066501]` and dependence reduction, but its NLL lower endpoint was `0.0187892`, below the frozen `0.02` threshold, and its predicted stage pattern failed in one seed. Hidden azimuth parity remained exactly one.

## Scientific decision

`KNOWN_TRUTH_PASS | SYSTEMS_PASS | IMAGE_SMOKE_FAIL | VIDEO_SMOKE_FAIL | CHILDREN_FAIL | CONFIRMATION_SEALED`

The first failed layer is target mismatch followed by approximation/representation. Shared state-conditioned second-order shape and low-rank plane-wise angular maps can improve exact likelihood or a chosen dependence statistic without improving whole-sample proper quality. The next legitimate layer would be a separately frozen joint cross-site or stage-aligned multivariate angular transport—not another covariance ridge, rank, plane, knot, threshold, or selector adjustment.
