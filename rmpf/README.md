# Reversible Multirate Pyramid Flow (RMPF)

RMPF is an all-dimensional exact flow hypothesis: an orthonormal pyramid retains every coordinate; coarse global variables receive expressive updates; details receive local updates every stage plus a periodic structured global communication tree. It is not a VAE, quotient, stochastic fiber, or route selector.

## Verified known-truth milestone

The finite `D=32`, `K=16` experiment contains heavy-tailed non-Gaussian texture, local detail structure, and a tunable rank-four parity-like global copula. The frozen confirmation uses seeds 9100--9104.

Primary confirmation results:

- no-global minus RMPF NLL: `0.05296 [0.05173, 0.05419]` nat/dimension;
- no-global minus RMPF global-bit error: `0.32570 [0.30553, 0.34587]`;
- full-attention minus RMPF proper energy score: `0.004286 [0.002411, 0.006160]`;
- RMPF/full-attention batch latency: `0.10764 [0.09386, 0.12143]`;
- peak-memory ratio: `0.21452`;
- stored-byte ratio: `0.31522 [0.31506,0.31538]`;
- unrestricted equivalence-copy mismatch: exactly zero.

All frozen known-truth gates passed. This promotes only to genuine-data development. It does not establish CIFAR/ImageNet/video quality or production GPU efficiency.

## Reproduce

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e rmpf
pytest -q rmpf/tests

python rmpf/experiments/known_truth/run_development.py \
  --output rmpf/results/development_seed9000 --seeds 9000

python rmpf/experiments/known_truth/aggregate_development.py \
  --root rmpf --output rmpf/results/development_frozen

python rmpf/experiments/known_truth/run_confirmation_seed.py \
  --freeze rmpf/results/development_frozen/freeze.json \
  --seed 9100 --output rmpf/results/confirmation_seed9100

python rmpf/experiments/known_truth/aggregate_confirmation.py \
  --root rmpf --output rmpf/results/confirmation_aggregated
```

The split orchestration notes record why confirmation seeds 9102--9104 were executed in byte-identical method groups after foreground timeouts. No method, metric, margin, seed, or test split changed.