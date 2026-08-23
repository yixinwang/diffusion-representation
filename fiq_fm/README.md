# Audited Flow-Induced Quotient Flow Matching

This folder is a clean research package added without modifying the repository's legacy MNIST/manifold scripts.

The implemented algorithm learns a low-dimensional **iterative** quotient from ordinary flow-matching labels, groups already-fixed residual axes using a cross-fitted conditional-covariance graph, trains the same latent flow used by latent baselines, and retains all discarded randomness in a one-shot block-Cholesky fiber. The graph does not recover a generic rotation inside the residual subspace. This implementation is a restricted special case of the broader moving-chart FIQ-FM draft.

## Current audit status

- The package contains five theorem/leakage tests and self-contained synthetic and digits runners.
- The mathematical result is conditional on a fixed aligned chart and a declared block-Gaussian fiber class.
- The old five-seed tables are historical snapshots: their referenced seed JSON files are absent, their summary schemas do not match the current runners, and the old digits runner failed during aggregation.
- Fresh smoke runs exercise the repaired code, but no empirical result is promoted until a clean confirmation run preserves raw records and regenerates every summary.
- Image-scale parity and orders-of-magnitude speedups have not been established.

## Historical findings, not current evidence

The archived exact-quotient table reported held-out sliced-W2 `0.0400 ± 0.0016` for FIQ, compared with `0.0614 ± 0.0016` for the parameter-matched full flow, `0.0708 ± 0.0016` for diagonal VAE-LFM, and `0.0638 ± 0.0013` for block RAE-LFM. It also reported a `1.292`-nat fitted block-vs-diagonal NLL gain and a `1.291`-nat analytic gap. Missing raw artifacts prevent treating those values as reproduced evidence.

The archived digits table reported better sliced-W2 and requested-label accuracy than the two finite-recipe latent baselines, but full FM had better raw sliced-W2 and RAE had slightly better linear-probe accuracy. The five overlapping train/test resamples are not independent scientific replications. These values are descriptive history only.

## Install and reproduce

```bash
python -m pip install -e .
pytest -q
python experiments/synthetic_exact/run.py --seeds 0 1 2 3 4 \
  --output results/synthetic_exact_confirmation
python experiments/sklearn_digits/run.py --seeds 0 1 2 3 4 \
  --output results/sklearn_digits_confirmation
```

## Structure

- `theory/FIQ_FM_verified.tex`: audited conditional theory note with archived result tables; compile with `pdflatex` to produce the PDF.
- `theory/FIQ_FM_original_draft.tex`: supplied broader draft, retained for comparison.
- `theory/approach_registry.md`: approach families and blocked routes.
- `theory/adversarial_audit.md`: failure-driven proof and implementation audit.
- `theory/experimental_protocol.md`: fairness and no-leakage protocol.
- `src/fiqfm/`: implementation.
- `experiments/`: self-contained experiment entry points.
- `tests/`: theorem-to-code and split tests.
- `results/`: archived pre-audit outputs; fresh confirmations must use new directories.
