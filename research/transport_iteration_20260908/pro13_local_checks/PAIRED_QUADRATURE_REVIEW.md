# Separate paired-quadrature diagnosis and modified validation

The unchanged failure remains preserved. A separate evaluation reproduced ordinary covariance integral0 and Gaussianized covariance integral−2.2762716439005403e−12, with reported quadrature error6.848622806616631e−9 and the50-subdivision warning. The latter exceeds the original assertion tolerance1e−12. This is an unreliable numerical integral for an exactly zero quantity, not evidence of nonzero covariance.

The public feature obeys psi(1−u)=psi(u): its knots and heights are reflection symmetric and each reflected affine segment agrees. Standard Gaussian symmetry gives Phi-inverse(1−u)=−Phi-inverse(u). The product f(u)=Phi-inverse(u)psi(u) is antisymmetric. It is absolutely integrable because |psi|≤sqrt(1.5) and the integral of |Phi-inverse(u)| is E|Z|<infinity. Substitution u→1−u proves its integral equals its negative, hence zero. The ordinary factor (u−1/2)psi(u) has the same argument.

A stable paired form integrates over0<u<1/2:

Phi-inverse(u) [psi(u)−psi(1−u)].

This uses exact Gaussian antisymmetry before floating evaluation and avoids subtracting two separately large endpoint contributions. Segmented quadrature returned4.8151969231301034e−18 with estimated error3.3777931463518116e−17 and no warning. At2004 explicit paired nodes, maximum feature-symmetry error was6.67e−16 and separately evaluated inverse-normal antisymmetry error2.80e−14. These are finite-arithmetic checks supporting the algebra; no interval certificate is claimed.

Only the `cov_z` quadrature expression was changed in a separate copied `check_math.py`. `paired_check.diff` and `modified_receipt.json` retain the exact one-line difference and original/modified hashes. The local `pro13.py` remained unchanged. The modified validation exited0 and reached all later assertions:100057 inverse cases per dtype,425 Gaussianized tail cases, exact packed-integer parity,323 KL-series/sandwich checks and the declared numerical bound. Maximum independently recomputed CDF residuals were1.11e−16 in float64 and1.04e−7 in float32; the largest Gaussian tail roundtrip error was1.23e−9, within the original2e−9 tolerance. KL quadrature/series difference was7.78e−16. No fit or missing-state regeneration occurred.

This is a successful modified numerical validation, not an unchanged reproduction. The unchanged original `check_math.py` still failed locally, and the original `check_bijection.py` covered only its fixed8D Jacobian because six fitted states are absent. The original52 recovered files remain untouched. Dependencies are the recorded local versions, not the complete delivered requirements. These checks do not complete the missing21-file delivery, reproduce compiled benchmarks, or establish native quality or interval-certified guarantees.
