# Independent review of the compact Arb calculation

The reviewed source is mathematically consistent with the stated restricted, uniform-tilt forward-KL calculation. The unchanged 64-call rerun returned the same enclosing strings and callback counts as the original report. This is a qualified trusted-computation certificate, conditional on the earlier global derivative/tail proof and the correctness of python-flint/Arb. It is neither a proof-assistant result nor an image-model or runtime claim.

## Analytic callbacks

I checked the installed python-flint 0.8.0 docstrings for `acb.integral`, `acb.log`, `arb.upper`, and `arb.str`. The [official complex integration documentation](https://python-flint.readthedocs.io/en/latest/acb.html) gives the same relevant callback contract; the online version currently describes 0.9.0. An analytic callback must reject a ball on which its function cannot be established analytic. The [real-ball documentation](https://python-flint.readthedocs.io/en/latest/arb.html) describes outward endpoints and enclosing string conversion.

In the nested integral the full complex `e` ball remains captured by the inner callback. Both logs receive `inner_analytic or outer_analytic`, including inner nonanalytic enclosure evaluations when the outer integrator demands holomorphy. Thus every accepted finite inner integration enclosure covers the entire outer parameter ball and checks both logarithm branches over the integration path and any accepted complex neighborhoods. The finite union of accepted subdivisions supplies local domination on the finite path, so integration preserves holomorphy in the parameter. Adaptive subdivision does not replace this function by a midpoint evaluation. No callback-contract defect was found.

All square roots involve fixed positive real time constants; they introduce no branch in either integration variable. Exponential and erf are entire. Rational denominators are handled as meromorphic expressions; poles that cannot be excluded yield nonfinite enclosures. On the real path, the separately established positive Heun Jacobian and positive tilt select the desired real logs. Checking the real path alone would have been insufficient, but the code additionally forwards the complex analyticity flag.

## Derivative and KL identity

Write a=1-t, S=a²+t², d²=2a²+t², k=t/d, u=ky, and D=1+e erf(u/sqrt(2)). The field is w=2ea phi(u)/(S d D). Since D_y=2e k phi(u), its derivative is w_y=-k w[u+2e phi(u)/D], exactly as implemented. For one Heun step the derivative multiplier is 1+h[A+B(1+hA)]/2, where A=w_y(y,t) and B=w_y(y+h w,t+h). The accumulated product is the derivative of the actual discrete map. All declared time grids have exact dyadic h=2/N and include the endpoint evaluation.

For the increasing onto Heun map H with derivative J, ell=log[p_e(H(z))J(z)/phi(z)]. Substitution gives forward KL as the Gaussian integral of exp(ell)ell. Subtracting the zero normalization integral of expm1(ell) gives F(ell)=exp(ell)ell-expm1(ell), a nonnegative function. The program integrates phi(z)F(ell). Multiplication by five averages e over [2/5,3/5]; multiplication by 2880 gives the specified additive joint KL. This average is not a uniform upper bound for every individual tilt.

## Tail bound

The prior analytic estimates |H-z|≤1.7 and |log J|<1.1 imply |ell|≤1.7|z|+3.5, using 0.4≤1+e erf(H/sqrt(2))≤1.6. These constants depend on the global field/Jacobian bounds; this numerical program does not prove those bounds anew.

For A=1.7, b=3.5, T=10, F(ell)≤exp(A|z|+b)(A|z|+b+1)+1. Completing the Gaussian square yields the implemented two-sided tail upper bound

2 exp(b+A²/2) [A phi(T-A)+(A²+b+1) barPhi(T-A)] + 2 barPhi(T).

This derivation checks both the shifted first moment and the extra A² term. The bound is uniform over the declared tilt interval and is multiplied by 2880 once. No quadrature convergence or grid minimum is used to certify the tail.

## Serialization and independent reproduction

Source SHA256: `78ddfa87cabb675d4a4a2b9e02f62c58e8533cea4ecd985319c3d243e97cf870`. All five original average reports match it. The new report is `work/pro10_arb_independent64.json`; it used unchanged source, python-flint 0.8.0, and 96-bit arithmetic, with 31,515 integrand callbacks and 32 outer callbacks. Its elapsed local certification time was about 14.4 seconds.

The reproduced full joint upper-bound ball is `[8.710696864367661329019035882e-6 +/- 6.36e-34]`, with exactly zero imaginary enclosure. It is an enclosure of a computed upper bound, not a narrow two-sided enclosure of the true KL. The compact joint integral itself is reported as `[8.710e-6 +/- 5.72e-10]`; its visible width is much larger. `upper()` first extracts an exact upward-rounded binary endpoint, and the subsequent arithmetic remains enclosing. The retained string radius must accompany the center. A safe simpler rational statement is that the restricted oracle average joint KL is less than 0.000008711, conditional on the stated assumptions. The exact 1/50000 comparison also passes.

Requested tolerances and evaluation caps are not accuracy certificates by themselves. A finite wide enclosure is still valid. As a robustness improvement for future reuse, final status could explicitly require finite tail/final upper and an imaginary enclosure containing zero. The reviewed fixed computations already satisfy these conditions; this is not a defect in their reported bound. Source, library, platform, and upstream analytic assumptions remain part of the trusted scope.

The separately proved wrong-learning probability correction may be added to this oracle bound to obtain the restricted expected-risk statement. This review does not establish those learning probabilities again and does not transfer the conclusion to neural image or video baselines.
