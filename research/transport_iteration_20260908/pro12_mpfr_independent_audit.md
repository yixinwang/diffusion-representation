# Independent source audit of the Pro12 MPFR certificate

This is a source and mathematical review, **not an independent execution of the delivered certificate**. No compilation, integration, fit, job, or historical timing replay was performed for this review. The publication is `a8954f35cf3dc09c3337d41a45187a7b7e462b11`, delivered locally by cherry-pick `90ed6da`. Reviewed source hashes are in `pro12_mpfr_independent_audit.json`; full delivery verification is a separate audit. Original artifacts are unchanged.

No blocking mathematical defect was found in the registered path: Gauss order 6, 128-bit endpoints, positive initial meshes, the five even stage counts, with precision selected before any interval objects are allocated. The order-8/192-bit path has the same argument. This finding does not make the custom interval implementation a formally verified or generally hardened numerical library.

## Discrete law and tail

The field and its derivative in `field` agree with the independent Gaussian-reference derivation. In particular, the first field is exactly $e/\sqrt\pi$ and the final field vanishes. The derivative of a step is $1+h[d_1+d_2(1+hd_1)]/2$, including the derivative of the predictor. The code multiplies these discrete factors; it does not substitute a continuous-flow log determinant.

The elementary proof of $\lambda k\le3/2$ checks algebraically. The two displayed decompositions of $3r^4-4r^3+r^2-4r+6$ are exact and nonnegative on their respective domains. For negative $q$, the two derivative terms have opposite signs, so the maximum of their magnitudes is a valid bound. The positive, negative-near, and negative-far cases in `global_bounds.cpp` correspond to the written inequalities. The relaxed bound $L=.6$ gives $r_h\le .345$, $\sum r_h\le .69$, and $|\log J|\le .69/.655<1.1$. Positive derivatives and bounded displacement make the scalar endpoint map an increasing global bijection.

For $\ell=\log[p_e(H)J/\phi(z)]$, change of variables gives forward KL as $E_\phi e^\ell\ell$. Normalization permits subtraction of $E_\phi\operatorname{expm1}(\ell)=0$. Thus the implemented nonnegative mathematical integrand $\phi(z)[\ell e^\ell-\operatorname{expm1}(\ell)]$ has the correct direction. Its interval enclosure may have a negative lower endpoint without invalidating the certificate. The displacement, log-Jacobian, and density-ratio bounds imply $|\ell|\le1.7|z|+3.5$. Integrating its exponential envelope gives the delivered tail formula. The factor 14400 is $2880/(.6-.4)$, while the full tail uses 2880. These normalizations agree.

## Interval derivatives and quadrature remainder

The jet coefficient convention is $f^{(k)}/k!$. Product convolution, reciprocal recurrence, exponential recurrence, and integration of the log/erf derivative series are finite derivative identities. They enclose derivatives at every point of the rectangle when their zeroth coefficients enclose that rectangle. They are not Taylor approximations with a missing remainder. `FJ` uses $F'(\ell)=\ell e^\ell$ with the chain rule for higher coefficients; its zeroth coefficient uses outward exponential and expm1 evaluations.

For Gauss order $n$ and interval width $d$, the integral of the squared monic node polynomial is

$$d^{2n+1}\frac{(n!)^4}{(2n+1)((2n)!)^2}.$$

Multiplying this by the supremum of the normalized derivative $f^{(2n)}/(2n)!$ proves the implemented remainder constant. There is no missing factorial. Applying $(I_z-Q_z)I_e+Q_z(I_e-Q_e)$ and positive Gauss weights gives the two-term rectangle remainder in the code without mixed derivatives. Summing actual interval remainder upper bounds, rather than relying on the requested tolerance or mesh agreement, is the correct final certificate.

Double-precision Newton values only propose roots. Certified opposite Legendre signs, disjoint brackets, and degree $n$ establish all roots; the weight formula encloses their exact weights. Registered orders also receive the explicit $(-1,1)$ check in `global_bounds`. The sum-of-weights and polynomial-moment tests are useful consistency checks but are not the proof of node validity.

Cell endpoints represent fixed rational endpoints and recursively fixed rational midpoints. Their intervals need not be point intervals. Each arithmetic expression encloses the corresponding exact endpoint, width, midpoint and evaluation node. Adjacent cell enclosures may overlap at rounded boundaries, but the underlying exact rational partition does not: each child refers to the same exact midpoint. Thus the sum encloses the integral on that partition rather than introducing an unaccounted overlap integral. Root bisection is likewise safe when its midpoint is a narrow interval whose entire Legendre image has a definite sign.

The allowance denominator 69120 equals the full-array multiplier 14400 times domain area $24/5$. Adaptive priorities and binary64 comparisons affect efficiency; accepting a cell requires an outward upper bound below an outward lower allowance. Final `conditional_KL` includes the quadrature enclosure, summed remainder, and nonnegative tail. The much narrower `quadrature_value_enclosure` alone is not an integral certificate.

## Statistical correction

The new uniform wrong-model bound is sound. Write $r=H_f(z)$ and $v=r-z\in[0,B]$. Convexity gives $-rv+v^2/2\le B(B/2-r)_+$. Since $F_e\le\Phi$, the tilted normal stochastically dominates a standard normal and the expectation of this decreasing hinge can only fall. The remaining density term has expectation $\mathrm{KL}(p_e\|\phi)\le e^2/3\le.12$. Therefore

$$M=B[(B/2)\Phi(B/2)+\phi(B/2)]+.12+1.1$$

is a valid uniform scalar bound. Its supplied upper bound is **2.8519519316104041**, not a bound below 2.85. This uniformly handles the wrongly fitted root, without assuming the resulting summary is uniform.

The sharper correct-root/wrong-head correction agrees with the separate earlier audit: $p_e/p_f\le3/2$, $\mathrm{KL}(p_e\|p_f)\le1/30$, and $E_{p_f}(\log(p_f/q_f))_-\le\mathrm{TV}(p_f,q_f)\le\sqrt{\kappa_f/2}$. Conditional on any fitted public row and fresh data, the fitted tilt is uniform when the root is correct. Jensen therefore gives $W_N(b)=96+1.5b+1.5\sqrt{1440b}$. No independence of true and fitted tilts is needed. Root/head fitting samples must remain independent, and the true class must retain its explicitly independent Gaussian root coordinates and conditionally independent residual coordinates.

Consequently the source formula $(1-p_\gamma-p_j)a\le E_{\rm train}\mathrm{KL}\le b+p_jW_N(b)+p_\gamma(128+2880M)$ is valid. The supplied N64 expected upper endpoint is $8.7180979953948377\,10^{-6}$, below $10^{-5}$; this paragraph reports the delivered receipt and checked formula, not a new numerical execution. Correct-recovery risk, expected fitted-model risk, and the KL of a mixture over fitted models remain different quantities.

## Implementation trust and limits

Endpoint add/multiply/inverse, monotone functions, erf/erfc and decimal parsing use MPFR's directed rounding. The official [MPFR rounding documentation](https://www.mpfr.org/mpfr-current/mpfr.html#Rounding) states the relevant rounding semantics. This library contract does not verify the surrounding custom interval code.

- `Real` copies round to nearest. Copies are exact in the registered process because all endpoints have the same fixed precision. Changing `PREC` while objects remain alive could shrink enclosures; that generalized usage is not justified.
- The fallback header manually declares MPFR's public ABI. It matches the usual documented layout/types, but ABI compatibility with the actual linked build is an additional deployment obligation. Prefer the installed vendor header. The reported runtime/compiler environment is historical evidence, not a locally reproduced environment.
- Invalid strings, arbitrary Gauss orders or nonpositive meshes are not comprehensively rejected. For example, empty mesh loops are not a meaningful certificate. The reviewed claim is confined to the fixed valid arguments used by `run.py`; the CLI should not be advertised as robust for arbitrary inputs.
- Several arithmetic helpers do not independently reject every nonfinite intermediate. Registered-path positivity and finite-error gates plus JSON checks cover the expected finite calculations; a general malicious-input/fault-tolerance claim would require more hardening.
- The tests explicitly check interval overlap for identities. That is appropriate as a consistency check, but overlap is weaker than containment and cannot prove all interval operations sound. The source-level recurrence argument carries that burden.
- `run.py` records local source hashes but does not itself compare them to an externally pinned manifest. Reproduction must first authenticate the delivered source closure. The parent delivery audit supplies that separate obligation.

This real-interval approach avoids complex branch assumptions, but adds a custom interval/jet/quadrature implementation to the trusted base. Our previously executed and reviewed Arb certificate remains independent evidence with a different trusted numerical implementation. Agreement between their results is useful; agreement alone proves neither certificate. Neither establishes floating-generator-law KL, rounded maximum-likelihood selection, native-image quality, VAE superiority, universal FM dominance, or certified runtime advantage. Exact-copy decoders still tie.
