# Pro11 delivered text subset: independent source and theory audit

The delivered directory is a verified **text subset**, not a verified complete fitted-state package. Original publication commit21f3997aed1447b05f3643032dc133b3e0cf9305 and local cherry-pick3a181cc6803584a39231078929682bbfbd3fccf8 contain the same16 published files. All15 original text files plus the new DELIVERY_SUBSET.md match both Git blobs byte-for-byte. The15 originals total117,659bytes. Of20 entries in the unchanged original SHA256SUMS,14 are present and match; six fitted-state NPZs are missing exactly as the delivery note declares. The manifest itself is not one of its20 entries.

The independent read-only checker is `pro11_check_delivery.py`, with machine report `pro11_delivery_independent_check.json`. It does not import or execute fitting/evaluation code. It also verifies all three seed records' model.py/run_checks.py hashes against the delivered source. Source hashes:

* model.py:866318dfb73bbe2d247a45603cd57c61a784649f3857370f1e1965a2abb20a88
* run_checks.py:ff553b4fc710b206c027f66dd2ad44da70ab150c87c4cc52645aa7ca293c1708
* SHA256SUMS:1806ccef2ab1baaed43c46a8540278ed5690519c21382f0b252a6b799268abf6

No fit, mathematical test suite, saved-bank regeneration or population-KL recomputation was rerun for this audit. The six absent states prevent independent execution of audit_saved.py against the originally saved fits. Their presence in an unavailable original ZIP is reported by the delivery note, not independently established by this subset audit. The archival saved_audit.json records are source-matching claims, not a substitute for the missing arrays. Do not mark the complete20-entry manifest verified or silently ignore its missing entries.

## Continuous scalar interpolant: the previous uncertainty is resolved

The actual sampler calls inverse_cdf(...,linear_grid=True). Both FittedModel.sample and Fixture.sample pass True explicitly. After dyadic bisection identifies a cell, the implementation interpolates inside it using the exact real-arithmetic cell-average density

1+theta psi(u) sqrt2 sinc(h) cos(2pi(lo+h/2)), h=2^-bits,

where NumPy's sinc(h)=sin(pi h)/(pi h). This is the correct integral average of sqrt2 cos(2pi v). It does not return a rounded bisection midpoint. An optional midpoint path exists but is not used by either registered sampler. Consequently the previous continuous-grid density argument applies to the ideal real-arithmetic operation, including its M*2pi*kappa/[J(1-2kappa)] discrepancy bound.

The explicit grid_pdf and grid_cdf agree with that same ideal interpolant. Gaussian-source inversion calls grid_cdf. Density/Jacobian checks explicitly call logpdf(...,grid_bits=32), and saved sample generation uses32bits. No2^32-entry table is constructed, but the32 bisection evaluations per pair are real computational work that must be charged.

There is an important API distinction: sample() defaults to the32-bit continuous-grid model, whereas logpdf() defaults to the *ungridded* cosine density. Callers must supply matching grid_bits when claiming the density of the sampled model. This is not a default-coherent sampler/log-density pair. The original evaluation intentionally reports the ungridded fit and documents the difference; future production integration should make the distinction explicit in method names/defaults rather than silently combine them.

The inverse has an absolute2e-14 bracket tolerance and no output clipping. It is ordinary float64 arithmetic; this audit does not certify all floating-point brackets, CDF endpoints, transcendental evaluations or tails. The theorem concerns the ideal continuous interpolant, not atomic floating output KL. Several low-level functions and logpdf lack comprehensive nonfinite/parameter validation; for example logpdf's comparisons do not reject NaNs. Registered fit() validates finite observations, but this archival reference should not be treated as a hardened production API.

## Positive and cancelled-signal fixture membership is resolved by source

Fixture.__init__ explicitly assigns `p[0]=1/8`. Thus C_1 is truly uniform in the declared fixture, even though its root law is still fitted and the other191 roots have randomized eight-bin masses. The estimator is not given this fact as a replacement for fitting its first root probabilities. The expected sine over C_1 is exactlyzero for every random phase.

For the positive world, theta_g(c)=sign_g[.35+.075 sin(2pi c+phase_g)]. Therefore |E theta_g|=.35, |theta_g|<=.425<.45, and its Lipschitz constant is2pi*.075<.5. C_1 densityone meets the .5 lower bound and is uniform inside every32-bin regression cell. The other root coordinates need not meet the C_1-specific density bound. For the changed world with offsetzero, E theta_g=0 exactly, so the registered nonzero-mean structure signal assumption fails. These conclusions follow from source, not from the reported successful fits or fallback outcomes.

The general warning remains valid outside this fixture: arbitrary nonuniform root histograms do not center sine, and positive offset.35 alone does not ensure |E theta|>=.35. The source resolves that dependency for these two particular laws by the explicit uniform first root.

## Learning API and proof assumptions

fit receives only an observation matrix and public hyperparameters: root dimension, group partition, root/context bin counts, structure split and bounds. It receives no fixture object, true matching, phases, signs, latent bank or teacher strengths. It estimates root histograms from all observations; graph scores use only the first split; conditional pooled moments use only the independent second split. It checks equal even group sizes, context-grid refinement and clipping bounds. Failed degree-one graphs use a deterministic pairing with theta identicallyzero, yielding the declared product fallback. An erroneous accepted perfect matching is possible and is handled by the theorem's bad-graph term.

The class supplies observed C_1 as the known context coordinate, known uniform residual marginals, a fixed group partition and the cosine feature. It is not a learned analysis/summary result for arbitrary data. Matching recovery requires the nonzero mean condition in each block and conditional independence of true pairs. The random context-count proof and variance constant1/12 require the aligned histogram model; these are correctly stated in THEORY.md. No false independence of360 responses with the same context is needed.

The earlier independent derivation reproduces the formula0.61910165641061, including all root, context, binomial-count, empty-bin, bad-graph and grid terms, versus the fixed-chart product floor88.2. These are symbolic theorems with ordinary numerical constants, not interval-certified decimal bounds. Context-constant pair baselines are not uniformly separated by the Lipschitz class, which includes constant functions; any empirical improvement against them is fixture-specific.

## What the text reports, and what is not independently verified

The three positive joint-KL values near0.21 and constant-context values near3.08 are ideal-density population evaluations of the *ungridded fitted parameters* against the ideal analytic fixture. The source computes these using an entropy series plus context quadrature, with a separate saved-fit audit based on two-dimensional quadrature. This audit checked the formulas and source separation but has not rerun those evaluations on the missing states.

Actual synthetic fitting observations were generated using44-bit continuous-CDF interpolation, with a separately reported uniform per-array log-density discrepancy bound around2.3144e-9 from the ideal law; fitted sample checks use32bits. This is appropriately disclosed in the source and THEORY.md. It is not literally sampling the exact analytic cosine law in real arithmetic. Applying the ideal-law finite-learning theorem verbatim to the perturbed fitting distribution would need a further perturbation argument. The small discrepancy is not itself such a proof, especially for an event bound much smaller than a training-bank total-variation bound.

The binned context estimator is piecewise constant and can make the full transport discontinuous as C_1 crosses a context-bin boundary. Its conditional scalar inverse is continuous in its own source, but that does not establish a continuous full joint flow. The independently proposed continuous-context trapezoid variant addresses a different model/estimator and must not retroactively revise this archive.

The all-fallback changed-law conditional KL2.028220434893 and other adverse values remain preserved as reported, not independently bank-verified. Their source-level explanation is consistent with the zero-mean score obstruction. The archive disclaims a matched global-FM experiment, native image/video superiority, exact-copy advantage and a comparative speed result. Those restrictions must remain with any citation of this artifact.
