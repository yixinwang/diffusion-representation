# Observed B1-v2 joint-RGB repair

This directory reproduces the frozen adaptive-development study in `qalt/theory/OBSERVED_B1_RGB_BLOCK_PROTOCOL.md`. It uses only CIFAR-10's five training batches. The old 5,000-image discovery split is excluded from every downstream fitting and score array; the official test batch remains sealed.

The seed runner writes one `seed_2100` through `seed_2104` directory containing raw per-image/per-band log scores, diagnostics, exact hashes, the opened-file ledger, and source provenance. The aggregator refuses missing seeds, identity/order/hash drift, schema or normalization drift, test-batch access, and failure of the samplewise B4/E4 tie. All inferential outputs are labeled adaptive development with no coverage.

From the repository root, verify on an allocated node:

```bash
export PYTHONPATH="$PWD/qalt/src"
/ocean/projects/mth250006p/ywang26/pytorch/bin/python -m pytest -q \
  qalt/tests/test_rgb_block.py \
  qalt/tests/test_observed_block.py \
  qalt/tests/test_observed_block_statistics.py \
  qalt/tests/test_observed_block_runner.py \
  qalt/tests/test_observed_block_aggregate.py
```

Submit the five frozen seeds only from the committed source snapshot:

```bash
bash qalt/experiments/observed_b1_rgb_block/submit.sh \
  qalt/results/observed_b1_rgb_block_development_20260824
```

After all five jobs finish, aggregate without loading a dataset:

```bash
export PYTHONPATH="$PWD/qalt/src"
/ocean/projects/mth250006p/ywang26/pytorch/bin/python \
  qalt/experiments/observed_b1_rgb_block/aggregate.py \
  --input-root qalt/results/observed_b1_rgb_block_development_20260824 \
  --output qalt/results/observed_b1_rgb_block_development_20260824/aggregate
```

The registered distribution-free pseudo-bound is expected to abstain because its analytic range term is much larger than `0.01`; do not change that threshold. Advancement is decided only by the frozen Welch/TOST/Holm development gates, followed by a separate untouched confirmation freeze.

The exploratory optimization child is frozen in
`qalt/theory/OBSERVED_B1_RGB_BLOCK_OPTIMIZATION_CHILD_PROTOCOL.md`. It changes
only the common maximum EM iterations from 200 to 1,000 and reuses development
data, so it has no confirmation coverage. Submit it to a new result directory:

```bash
MAX_ITERATIONS=1000 bash qalt/experiments/observed_b1_rgb_block/submit.sh \
  qalt/results/observed_b1_rgb_block_optimization_child_20260830
```
