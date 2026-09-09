# Independent partial-delivery source review

This is a read-only source/provenance review. No C compilation, Python artifact execution, observation generation, fitting or timing was performed. The21 missing fixture/state files prevent independent reproduction of the delivered benchmark and fitted-state claims. Original README numbers remain author-reported local results until the missing material and subsequent checks are available.

## Packed graph semantics

The Python and C packed paths implement the same sign statistic: each coordinate contributes a bit for x≤0.25 or x≥0.75; XOR popcount gives Hamming distance h and signed correlation numerator n−2h. The exact integer threshold is |n−2h|×1000≥159n. Entire blocks fail to the product model unless every vertex has degree one. C computes only unordered pairs, and its wrapper explicitly pads the final uint64 word with zero bits. The kernel's finite-size/threshold guards bound products safely in the registered n=2000,d=720 use. It remains a low-level pointer API relying on the wrapper for buffer shape, allocation and zero padding; do not expose the C entry point as a general untrusted-buffer interface.

The packed statistic is not an exact implementation of the dense psi Gram statistic. The dense path uses real-valued psi correlations and threshold≥0.175; the packed path uses signs and threshold≥0.159. Their fitted graphs may agree in the declared positive cases, but are not universally identical algorithms. Claimed C/Python packed parity concerns the identical sign statistic. Their small parity checks include sample sizes not divisible by64. Full fitted-parameter parity checks intentionally exclude timings, method labels and graph diagnostics; they compare the resulting roots, pairs and coefficients after JSON normalization. The original failed pre-normalization assertion/log is preserved and must not be hidden.

The C implementation compiles with `-O3 -march=native`, builds a temporary shared object and loads it using ctypes. Compilation and compiler-version inspection are explicit subprocesses. This review did not invoke them. Platform-specific code generation and popcount availability matter to performance; a local speed ratio does not transfer to PSC or GPUs. Runtime shared objects are explicitly excluded original artifacts, not among the21 newly missing delivery files.

## Timing accounting and scope

The compiled graph benchmark rotates three method orders, limits BLAS threads and times validation, feature/sign construction, packing and degree handling. It regenerates its own fabricated arrays and checks recorded hashes, rather than receiving native data or passing fixture truth into the learner. Only three repetitions per condition are present; these are descriptive timings, not a robust cross-machine complexity result. Compile/load is separately reported and excluded from warm rows, so cold use must include it.

The stronger SYRK benchmark computes one triangle and tests one/four BLAS threads; the C kernel itself remains single-thread. It is limited to positive fixtures. The outer-fit benchmark includes the initial whole-array validation scan omitted from the learner's internal summed stage timer, and excludes generation/serialization/import cost explicitly. Its order alternates but gives compiled-first twice and dense-first once. These limits are disclosed rather than evidence of a universal runtime win. Preserved Python-packed losses, initial comparison failure and later stronger-comparator results are separate measurements.

The current partial package lacks all three fixture JSON files and18 fitted-state JSON files referenced by these runners. Hash-verified source and summary files do not make the runs reproducible without those dependencies. `reproduce.py` should not be executed in the preserved publication directory; existing results and sources are provenance.

## Zero-mean failure and regularity

The fabricated generator computes sine means by integrating each phase under the actual first-root eight-bin histogram, then subtracts those means for the stress law. This is a correctly targeted marginal-correlation falsifier rather than assuming a nonuniform root centers sine automatically. The positive law keeps signed amplitudes bounded away from zero. Learner fitting accepts only observed arrays and public dimensions/split/configuration; fixture matching/phases remain confined to generation and evaluation.

For the centered law, integrating out context removes the pair signal from unconditional correlations. Failed matching and product fallback therefore expose a real limitation of the proposed unconditional graph statistic, not a proof that conditional information is absent. The source preserves the corresponding negative result. The packed expected-risk guarantee assumes a nonzero mean signal; it does not solve phase-unknown zero-mean graph discovery.

Coefficient lookup is still piecewise constant by context bin in the delivered learned map. Its full joint transform generally has context-boundary discontinuities; scalar conditional-CDF continuity does not imply a globally C1 learned flow. The original README rejects a globally C1 claim; its statement of declared-law/inverse validity must be read separately from learned-map regularity. This package also differs from the later continuous-context variant developed in the main workspace.

No native-image/video, PSC, unrestricted full-flow, VAE-representation or global-FM dominance result follows from these sources. The analytic exact-copy control ties. The useful prospective contribution is a lower-cost sign-correlation implementation for a restricted identifiable family, with an explicit zero-mean failure and local hardware/compiler qualifications.
