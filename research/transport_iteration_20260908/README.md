# September 8 transport and observation-score iteration

The requested joint quality, speed, and representation advantage remains unestablished. This iteration repairs a sampler's invertibility, supplies a way to reject output corrections too small for an energy-score target, and resumes the previously frozen covariance experiment on PSC.

## Existing evidence and scope

The default GitHub branch was `e00a034`. PSC had seven additional committed development steps through `4553e04`, which this branch preserves. The existing user checkout on PSC has unrelated changes; execution uses an isolated worktree. Historical R20--R22 branches remain separate. Their image-quality failures are retained as constraints on further angular repairs.

The newer radial development result improved conditional density relative to B4 but lost to its Student control and left direction and band-share errors unchanged. Its frozen covariance child tests whether cross-band second moments explain those errors. That study uses already examined repair-development images. Its estimates cannot establish confirmation performance. The official CIFAR test batch remains excluded. This is a conditional-density study; it does not train a full-image generator or establish image/video superiority to latent diffusion.

## Changes

- `qalt/src/qalt/radial_transport.py` maps three Gaussian coordinates invertibly to the existing RGB scale-mixture density by radial inverse CDF. It returns inverse maps and numerical log determinants. The previous mixture sampler is preserved. This is an alternative realization of the same law, with no quality gain claim.
- `qalt/src/qalt/output_screen.py` rejects an energy-score improvement margin when twice an upper bound on paired output displacement is below that margin. It takes generated pairs only. It requires finite first moments, frozen generators, fresh iid sources, and an analytically justified displacement bound. It cannot establish an improvement.
- `qalt/experiments/transport_validation/run.py` checks source-only nonlinear, non-Gaussian numerical fixtures and compares transport costs. No observed-data loader is imported.
- The existing covariance child and its thresholds are unchanged.

Local checks: 14 new transport/screen tests and 23 existing covariance tests passed. The numerical fixture has a Gaussian source and a nonlinear triangular six-dimensional inverse. Its maximum round-trip error was about 2.2e-12. The radial inverse CDF took 44 probability-evaluation rounds. On this local diagnostic it was much slower than the legacy mixture sampler; it supplies no speed advantage. The synthetic displacement upper bound was 0.000185 in normalized energy-score units, below the predeclared 0.005 margin. That rejection concerns the deliberately tiny synthetic correction only.

The initial numerical runner failed while serializing a NumPy Boolean. The screen now converts scalar inputs to native floats; a JSON serialization regression check passes. No experimental outcome or scientific threshold changed.

## Fresh ChatGPT Pro conversations

All three were launched through the signed-in ChatGPT page with the visible `6 Pro` selection. Each is a distinct conversation.

1. [Conditional flow construction and strongest baselines](https://chatgpt.com/c/6aa05d00-b5a8-83e9-b027-551012e78bcc).
2. [Observation-space mechanism after R20--R22 failures](https://chatgpt.com/c/6aa05d79-8a20-83e9-9e12-b6e9c4e52a25).
3. [Frozen covariance experiment review](https://chatgpt.com/c/6aa05f19-db10-83e9-8701-577f8f0b5fb7).

Their responses remain subject to independent mathematical and implementation checking. Launching a conversation is not evidence that its analysis is complete.

## Theory and independent checks

`energy_derivation.md` gives complete definitions, the sharp factor-two displacement inequality, finite-source confidence assumptions, and an exact smooth non-Gaussian example of observation-score dilution. `energy_independent_check.md` independently verifies its coefficients and limits. `radial_derivation.md` gives the inverse-CDF construction, origin behavior, numerical limitations, and cost.

The algorithmic-method writing comparison used hash-verified ranges from Ranganath, Gerrish, and Blei (2014), methods pages 2--6, and Hoffman, Blei, Wang, and Paisley (2013), methods pages 5--31. The note follows their target-first method exposition and separates mathematical assumptions from execution evidence. These are writing references only; no current generative theorem is attributed to them.

The novelty review remains conservative. Conditional multiscale factorization is established prior work; Guth et al. (2023), *Conditionally Strongly Log-Concave Generative Models*, Section 2, provides a closely related factorization and conditional learning/sampling analysis ([primary paper](https://proceedings.mlr.press/v202/guth23a/guth23a.pdf)). The present repair uses elementary radial probability transforms and energy-score inequalities. No priority claim is made. An unrestricted stochastic latent decoder can implement the same conditional law and tie it.

## Next decision

Inspect the frozen PSC covariance result and its first failed criterion. Conditional means, conditional covariance, and higher-order direction errors require different interventions. Any new intervention requires a fresh Pro conversation, a checked quantitative model result, a prospective development specification, and fair controls. Untouched confirmation stays closed until the required development comparisons pass.

## PSC execution record

The frozen covariance child was submitted as job `45550358`, at source `4553e04f71c545e401409d18d82731589974760b`, from an isolated worktree. The first submission was rejected before allocation because 8 GiB exceeded the 2,000 MiB-per-core limit for four CPUs. Resubmission requests 8,000 MiB, below the frozen 8 GiB ceiling. No code, data, threshold, or method changed.

ChatGPT subsequently displayed a temporary too-many-requests restriction on all three launched conversations. Their final reviews have not yet been retrieved. No Pro conclusion is credited before it can be read and checked.
