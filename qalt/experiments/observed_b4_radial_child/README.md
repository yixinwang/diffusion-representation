# Observed B4 reversible radial child

This adaptive seed-2100 screen implements the frozen protocol in
`qalt/theory/OBSERVED_B4_RADIAL_CHILD_PROTOCOL.md`. It uses CIFAR training
batches only. It must not deserialize `test_batch`.

Focused checks:

```bash
OMP_NUM_THREADS=1 PYTHONPATH=qalt/src \
  /ocean/projects/mth250006p/ywang26/pytorch/bin/python -m pytest -q \
  qalt/tests/test_radial_b4_flow.py \
  qalt/tests/test_observed_b4_radial_child_runner.py \
  qalt/tests/test_observed_block.py::test_b4_only_fitter_is_bitwise_identical_to_full_fitter
```

The committed runner command will be:

```bash
source_commit=$(git rev-parse HEAD)
OMP_NUM_THREADS=1 PYTHONPATH=qalt/src \
  /ocean/projects/mth250006p/ywang26/pytorch/bin/python \
  qalt/experiments/observed_b4_radial_child/run.py \
  --seed 2100 \
  --source-commit "$source_commit" \
  --output qalt/results/observed_b4_radial_child_20260830/seed_2100
```

For Slurm, choose a new result root and bind the same commit at submission:

```bash
sbatch --export=ALL,RESULT_ROOT="$PWD/qalt/results/observed_b4_radial_child_20260830",SOURCE_COMMIT="$(git rev-parse HEAD)" \
  qalt/experiments/observed_b4_radial_child/run_seed.slurm
```
