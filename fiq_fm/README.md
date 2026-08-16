# Gauge-Fixed Flow-Induced Quotient Flow Matching

This folder is a clean research package added without modifying the repository's legacy MNIST/manifold scripts.

The implemented algorithm learns a low-dimensional **iterative** quotient from ordinary flow-matching labels, fixes the residual coordinate gauge using a cross-fitted conditional-covariance graph, trains the same latent flow used by latent baselines, and retains all discarded randomness in a one-shot block-Cholesky fiber. It is a restricted, auditable special case of the broader moving-chart FIQ-FM draft.

## Verified status

- Six theorem/leakage tests pass.
- Five-seed exact-quotient synthetic benchmark completed.
- Five-seed sklearn digits benchmark completed.
- Full configurations, per-seed JSON, long-form CSV, and paired summaries are committed; rerunning the scripts regenerates plots.
- Claims are conditional. Image-scale parity and orders-of-magnitude speedups are not claimed.

## Main findings

On the exact-quotient benchmark (`D=18`, active `d=2`), FIQ achieves held-out sliced-W2 `0.0400 ± 0.0016`, compared with `0.0614 ± 0.0016` for the parameter-matched full flow, `0.0708 ± 0.0016` for diagonal VAE-LFM, and `0.0638 ± 0.0013` for block RAE-LFM. The learned block-vs-diagonal NLL gain, `1.292` nats, matches the analytic `1.291`-nat KL gap.

On digits (active `d=16`), FIQ significantly improves held-out sliced-W2 and requested-label accuracy over both latent baselines. It has the best mean class-conditional feature Fréchet, but that advantage is not statistically established at five seeds. Its z-only reconstruction MSE is `0.190`, versus `0.420` for VAE-LFM and `0.373` for RAE-LFM.

## Install and reproduce

```bash
python -m pip install -e .
pytest -q
python experiments/synthetic_exact/run.py --seeds 0 1 2 3 4 \
  --output results/synthetic_exact_verified
python experiments/sklearn_digits/run.py --seeds 0 1 2 3 4 \
  --output results/sklearn_digits_verified
```

## Structure

- `theory/FIQ_FM_verified.tex`: coherent verified theory/results note; compile with `pdflatex` to produce the PDF.
- `theory/FIQ_FM_original_draft.tex`: supplied broader draft, retained for comparison.
- `theory/approach_registry.md`: approach families and blocked routes.
- `theory/adversarial_audit.md`: failure-driven proof and implementation audit.
- `theory/experimental_protocol.md`: fairness and no-leakage protocol.
- `src/fiqfm/`: implementation.
- `experiments/`: self-contained experiment entry points.
- `tests/`: theorem-to-code and split tests.
- `results/`: complete verified outputs.
