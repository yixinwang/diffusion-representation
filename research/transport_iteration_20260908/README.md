# September 8 transport and observation-score iteration

The requested joint quality, speed, and representation advantage remains unestablished. This iteration repairs a sampler's invertibility, supplies a way to reject output corrections too small for an energy-score target, and evaluates the previously frozen covariance experiment on PSC.

## Existing evidence and scope

The default GitHub branch was `e00a034`. PSC had seven additional committed development steps through `4553e04`, which this branch preserves. The existing user checkout on PSC has unrelated changes; execution uses an isolated worktree. Historical R20--R22 branches remain separate. Their image-quality failures are retained as constraints on further angular repairs.

The newer radial development result improved conditional density relative to B4 but lost to its Student control and left direction and band-share errors unchanged. Its frozen covariance child tests whether cross-band second moments explain those errors. That study uses already examined repair-development images. Its estimates cannot establish confirmation performance. The official CIFAR test batch remains excluded. This is a conditional-density study; it does not train a full-image generator or establish image/video superiority to latent diffusion.

## Changes

- `qalt/src/qalt/radial_transport.py` maps three Gaussian coordinates invertibly to the existing RGB scale-mixture density by radial inverse CDF. It returns inverse maps and numerical log determinants. The previous mixture sampler is preserved. This is an alternative realization of the same law, with no quality gain claim.
- `qalt/src/qalt/output_screen.py` rejects an energy-score improvement margin when twice an upper bound on paired output displacement is below that margin. It takes generated pairs only. It requires finite first moments, frozen generators, fresh iid sources, and an analytically justified displacement bound. It cannot establish an improvement.
- `qalt/experiments/transport_validation/run.py` checks source-only nonlinear, non-Gaussian numerical fixtures and compares transport costs. No observed-data loader is imported.
- The existing covariance child and its thresholds are unchanged.

Local checks: the full QALT suite passed all 129 tests under Python 3.12, including the 14 new transport/screen tests and 23 covariance tests. The numerical fixture has a Gaussian source and a nonlinear triangular six-dimensional inverse. Its maximum round-trip error was about 2.2e-12. The radial inverse CDF took 44 probability-evaluation rounds. On this local diagnostic it was much slower than the legacy mixture sampler; it supplies no speed advantage. The synthetic displacement upper bound was 0.000185 in normalized energy-score units, below the predeclared 0.005 margin. That rejection concerns the deliberately tiny synthetic correction only.

The initial numerical runner failed while serializing a NumPy Boolean. The screen now converts scalar inputs to native floats; a JSON serialization regression check passes. No experimental outcome or scientific threshold changed.

## Fresh ChatGPT Pro conversations

The first three were launched through the signed-in ChatGPT page with the visible `6 Pro` selection. Each is a distinct conversation.

1. [Conditional flow construction and strongest baselines](https://chatgpt.com/c/6aa05d00-b5a8-83e9-b027-551012e78bcc).
2. [Observation-space mechanism after R20--R22 failures](https://chatgpt.com/c/6aa05d79-8a20-83e9-9e12-b6e9c4e52a25).
3. [Frozen covariance experiment review](https://chatgpt.com/c/6aa05f19-db10-83e9-8701-577f8f0b5fb7).

Their responses remain subject to independent mathematical and implementation checking. Launching a conversation is not evidence that its analysis is complete.

## Theory and independent checks

`energy_derivation.md` gives complete definitions, the sharp factor-two displacement inequality, finite-source confidence assumptions, and an exact smooth non-Gaussian example of observation-score dilution. `energy_independent_check.md` independently verifies its coefficients and limits. `radial_derivation.md` gives the inverse-CDF construction, origin behavior, numerical limitations, and cost.

The algorithmic-method writing comparison used hash-verified ranges from Ranganath, Gerrish, and Blei (2014), methods pages 2--6, and Hoffman, Blei, Wang, and Paisley (2013), methods pages 5--31. The note follows their target-first method exposition and separates mathematical assumptions from execution evidence. These are writing references only; no current generative theorem is attributed to them.

The novelty review remains conservative. Conditional multiscale factorization is established prior work; Guth et al. (2023), *Conditionally Strongly Log-Concave Generative Models*, Section 2, provides a closely related factorization and conditional learning/sampling analysis ([primary paper](https://proceedings.mlr.press/v202/guth23a/guth23a.pdf)). The present repair uses elementary radial probability transforms and energy-score inequalities. No priority claim is made. An unrestricted stochastic latent decoder can implement the same conditional law and tie it.

## Completed PSC results and next decision

The covariance child completed as job `45550810` at source `4553e04`, with first failed criterion `paired_nll`. It gained 0.3184877 nats/detail over B4, gained -0.00000284 over block covariance, and gained -0.1834003 over Student. The block and Student comparisons failed. The local runtime ratios passed. The model still failed second-moment, band-share, band-correlation, and radial-PIT checks. The independent score recomputation reproduces all per-image contrasts and statistical decisions and verifies all payload hashes.

The holdout Gaussian optimum offers only 0.00009782 nats/detail above fitted full covariance on these same development arrays. This excludes closing the observed Student gap by another global zero-mean covariance fit on this chart. It is a same-array diagnostic, with no population coverage claim.

The numerical fixture completed as PSC job `45551078` at source `db4429e`. All 14 focused tests passed. The nonlinear six-dimensional round-trip error was 2.17e-12. The radial inverse remains substantially slower than the legacy sampler. The exact fixture and source hashes are in `psc_fixture.json`.

All three Pro responses have now been read. Their mathematical conclusions agree with the limited interpretation above. Pro 1 proposes a complete learned multiscale conditional spline flow; its coarse distribution must also be learned. The proposed construction uses the conditional factorization of Wavelet Flow, and a stochastic latent decoder can copy it exactly. Pro 2 derives a directional output-energy exclusion bound. Pro 3 verifies the covariance algebra and predicts failure if full covariance does not surpass block covariance and the fixed Student rival. None ran our PSC jobs or verified source that was not yet published at their review time.

A [fresh fourth Pro iteration](https://chatgpt.com/c/6aa0633a-e1f8-83e9-812d-73bdd9749c78) is developing a complete observation-only generative experiment after this failed covariance result. It must quantify learning and approximation under local conditional structure, include a distant-dependence failure case, learn the coarse law, and include the exact stochastic-latent copy. No previously used image split will be relabeled as untouched confirmation.

## Execution record

The initial eight-GiB submission exceeded the PSC four-core memory limit and was rejected before allocation. Job `45550358` then exited before data access because the batch environment omitted the result path. Explicit environment export corrected that launch issue for `45550810`; the method, thresholds, and data rules were unchanged. See `execution_record.md` for the complete startup record. The original dirty PSC checkout remains untouched; separate worktrees isolate both completed jobs.

## Completed conditional learning and the next implementation

PSC job `45552606` completed all 36 frozen conditional-CDF cells at revision `ffec36e`; all result and source hashes were independently checked. Tree density error per coordinate decreased from approximately 0.0303 to 0.0075 across training sizes 256 to 4,096. This descriptive trend does not establish an asymptotic rate. The distant-context study remains subject to a positive joint information-loss floor. Every stochastic latent copy tied bitwise. Raw records and the independent review are in `psc_cdf/` and `psc_cdf_review.md`.

The fourth Pro response is complete. It suggests positive spline densities with a convex likelihood objective, supplying an explicit optimization gap. The reviewed theorem is in `positive_spline_learning.md`. Its large finite constants fail to certify a practical separation at ordinary dataset sizes, as shown in `positive_spline_finite_constants.md` and its companion JSON. The theorem does not apply to the neural model simply because both use splines.

`positive_density_spline.py` implements quadratic context bases and positive linear response densities; `positive_spline_flow.py` supplies the complete Gaussian-to-observed-coordinate transport, including fitted root densities and shared conditional groups. Fitting counts independent arrays separately from correlated pooled sites. Independent numerical references verify inversion, Jacobians, normalization, constrained minimization, and optimization gaps. The full QALT suite passes 175 tests. A test invocation initially omitted the local package path and failed collection; the corrected environment passed.

A [fresh fifth Pro conversation](https://chatgpt.com/c/6aa06c18-b268-83ea-8f3b-e5f7890bb8db) is reviewing finite practical guarantees and a strong real-data comparison. PSC GPU job `45552255` remains queued at this update. No real-image or video generation advantage has been established.
