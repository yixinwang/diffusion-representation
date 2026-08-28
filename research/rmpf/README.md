# Reversible Multirate Pyramid Flow (RMPF)

This append-only package records the first verified milestone for a no-VAE, full-dimensional multiresolution flow. Every coordinate is retained in one exact normalized law. Efficiency comes from sparse-in-time global communication and streamed tensor-tree coupling, not dimension reduction.

## Frozen known-truth result

The target has 32 visible dimensions, non-Gaussian componentwise texture, 16 exact invertible stages, local detail updates at every stage, and a rank-4 parity-like global copula active at four frozen stages. Development seeds were 9000--9002 and untouched confirmation seeds were 9100--9104.

Across confirmation seeds:

- no-global minus RMPF NLL: `0.07664 [0.07433, 0.07895]` nat/dimension;
- local MCQF minus RMPF NLL: `0.07908 [0.07703, 0.08114]`;
- full-attention minus RMPF energy score: `0.01065 [0.00635, 0.01495]`;
- mean RMPF/full-attention batch-latency ratio: `0.188`;
- single-sample latency ratio: `0.425 [0.399,0.452]`;
- peak-allocation ratio: `0.1906`;
- workspace ratio: `0.2222`;
- stored-file ratio: `0.3106`;
- exact copied-mechanism mismatch: zero.

The smoothed even-parity adversary had maximum pairwise signal below 0.05. RMPF achieved perfect sign accuracy in all five seeds; no-global and MCQF remained near chance.

This is a finite known-truth mechanism result, not a CIFAR, high-resolution image, or real-video promotion. An unrestricted full model that copies the complete RMPF schedule and mixer is the equivalence control and ties exactly.

## Reproduce

```bash
export PYTHONPATH=research/rmpf/src
python research/rmpf/run_known_truth.py development
python research/rmpf/run_known_truth.py confirmation
python research/rmpf/verify_known_truth.py
```

See `PREREGISTRATION.md`, `RMPF_THEORY_AND_KNOWN_TRUTH.md`, and `INDEPENDENT_PROOF_AUDIT.md` for the frozen design, derivations, failure conditions, and audit.