# Independent Pro14 reference comparison and local replay

The delivered reference was authenticated against publication `ccc7844b0270fddbfd42bf7d4943449bf7425f79` and cherry-pick `ce2b6ba`: all 20 files matched both Git trees, and all 19 SHA-manifest entries matched. Its original `checks.py` writes `fabricated_results.json` beside itself with replacement semantics. Accordingly, unchanged `checks.py` and `mechanism.py` were copied to a **new** `work/pro14-local-checks` directory before execution. Original receipts were neither copied into that output directory nor overwritten. Authentication, stdout, stderr, process status and the fresh receipt are preserved there.

## Implementation comparison before native freeze

The production implementation was written independently from the supplied equations before reference delivery. It is **not byte-equivalent**, and has two material prospective design differences:

| Item | Delivered `mechanism.py` | Independent production implementation at review |
|---|---|---|
| Default anchors | Stride 45: `0,45,...,675` | Endpoint-inclusive nearest integers: `0,48,...,671,719`; integer half ties up |
| Hidden activation | `tanh` | `SiLU` |
| Default head | `48 -> 32 -> 32` | Same |
| Innovation features | `[h,tanh(u),tanh(u)^2]` | Same |
| Prefix-only features | `[h,tanh(h),tanh(h)^2]` | Same at rank16 |
| Output initialization | Zero | Zero |
| Follower loadings | Existing frame follower rows | Same construction, with the different declared follower set |
| Scale | `log(2)*tanh(U_f b)` | Same, hence positive scale in `(1/2,2)` for finite arguments |
| Smaller fabricated rank | Summary dimension equals rank; adjustable width | Summary remains16; first rank summary entries supply control features; width32 |
| Surrounding model | Independent fabricated scalar/SPD/root/rotation algebra | Existing production scalar, QR/Cayley, exact root, learned analysis and prefix conditioner |
| Cache plumbing | Toy summary passed explicitly | Existing summary returned in the same conditioner pass; same ephemeral context/frame reused |
| Finite guards | Immediate host checks including hidden preactivation | Aggregated device validity, including hidden output and pre-tanh scale projection |

Both modes in each implementation have equal active parameter counts and operation shapes. Both default implementations add 2624 parameters/block, 10496 total, and give full-model count539120 when integrated with the stated baseline. Actual production construction separately verified scalar width42 count544080. There is no claim that separately seeded reference and production objects share weights or return the same samples.

The activation difference affects the family: for finite fixed head weights the reference's hidden tanh uniformly bounds its output coefficients over arbitrary history summaries; production SiLU does not. Both still have bounded final scales, and at each fixed finite history their bounded anchor features give finite mean modulation. The exact inverse and normalization do not depend on choosing tanh versus SiLU, but a bound on coefficients uniformly over histories must not be transferred. Anchor placement changes which innovation coordinates can condition followers. **The parent's prospective decision is to retain production SiLU and endpoint-inclusive anchors for both controls.** This is an explicitly declared variant, not a reference-parity claim. No activation/anchor production change was made during this audit.

The production mean-shift witness in `innovation_response_math.md` uses the exact SiLU identity to realize a centered `tanh(u)^2` feature. The reference instead realizes the scale witness $s(u)=\log2\tanh(\tanh(\tanh^2u))$. They are distinct normalized examples, not a failed parity test. The retained SiLU choice preserves the production witness. The independent math note states the actual linear-in-history coefficient bound and does not transfer the reference's uniform coefficient bound.

## Proof and protocol comparison

The reference's anchor/missing-history/follower-TC/scalar-shape/Gaussian-fit KL decomposition agrees with the independent note, conditional on fixed fitted analysis/base/frame and its finite-moment/entropy assumptions. The covariance correction has rank at most2r before arbitrary marginalization or nonlinear inverse analysis; neither note turns it into a positive semidefinite factor claim. The restricted independent-Gaussian energy witness and fixed-base correlated-follower obstruction are valid. No theorem establishes learnability or the stated native gates.

The reference uses a loose energy-score population bound $2\mathrm{TV}$; the independently derived $\mathrm{TV}^2$ excess bound in `innovation_response_math.md` is sharper on the normalized pixel cube. This is not a contradiction. Its covariance and bounded-kernel MMD qualifications otherwise agree. Its mean/scale likelihood scores are correctly $-z e^{-s}$ and $1-z^2$; response-direction feature-weighted diagnostics can see conditional energy errors despite zero ordinary covariance. After projection and nonlinear activation, these diagnostics are useful adequacy tests rather than complete head gradients or independence certificates.

The delivered protocol is a proposed factorial native screen, not an executed experiment. It suggests a charged compile-versus-eager policy; the parent's separately declared next screen uses eager response and the checked dense-eager RQS backend, deferring compilation. These must be recorded as separate implementation policies, without presenting the original proposal as an executed registration. The original reference explicitly uses fabricated surroundings and does not validate production prefix masking, root/analysis gradients or cache lifecycle; the production test suite supplies separate fabricated evidence for those integration points.

## Unchanged reference replay

The unchanged reference completed successfully locally with no failed check or timeout. Python3.12.14/Torch2.14.0/NumPy2.5.3/SciPy1.18.1 differ from the historical Python3.13.5/Torch2.10.0/NumPy2.3.5/SciPy1.17.0. The process took2.0231 seconds; the reference's internal clock reports approximately .124 seconds. Neither is a sampler benchmark or native training cost.

- Small prefix/innovation dense-Jacobian maximum errors: $2.8311\,10^{-15}$ and $1.0270\,10^{-15}$. Round-trip errors were below $1.2\,10^{-15}$; directional head-gradient errors were below $1.3\,10^{-10}$.
- Full3072 float64: round trip $7.1054\,10^{-15}$, determinant cancellation $4.8850\,10^{-15}$. Float32: round trip $3.8147\,10^{-6}$, determinant cancellation $2.3842\,10^{-6}$. Same-source repeated outputs were exactly equal on these fabricated banks.
- The reference nonlinear scale witness gives $E[V^2]=1.61566209858$, $\operatorname{Cov}(U^2,V^2)=.536401466412$, and best independent-Gaussian KL gap $.0226021514214$. Its ordinary density integral was $1.0000000000000027$; the quadrature error estimate is not an interval certificate.
- The fixed-base follower-correlation floor $.223143551314$ was retained. The intentionally omitted determinant was rejected. Five nonfinite cases were rejected, and the reused-frame directional gradient error was $4.7376\,10^{-11}$.

The source hashes in the fresh receipt match the authenticated original two files. No qalt module was imported by the reference execution, no fit or native observations were read, and no production source, checkpoint or job was changed in this comparison.
