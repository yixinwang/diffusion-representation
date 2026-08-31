# Observed B4 covariance-only child

This seed-2100 adaptive repair screen implements
`qalt/theory/OBSERVED_B4_COVARIANCE_CHILD_PROTOCOL.md`. It reproduces and
hash-checks the completed radial parent before fitting any covariance arm. It
uses CIFAR training batches only and must not deserialize `test_batch`.

Focused checks (no CIFAR access):

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=qalt/src \
  /ocean/projects/mth250006p/ywang26/pytorch/bin/python -m pytest -q \
  qalt/tests/test_covariance_b4_flow.py \
  qalt/tests/test_observed_b4_covariance_child_runner.py \
  qalt/tests/test_observed_block.py::test_b4_only_fitter_is_bitwise_identical_to_full_fitter
```

After the protocol, runner, tests, and launch files are committed together,
submit a new result root while binding the exact source revision:

```bash
sbatch \
  --export=ALL,RESULT_ROOT="$PWD/qalt/results/observed_b4_covariance_child_20260831",SOURCE_COMMIT="$(git rev-parse HEAD)" \
  qalt/experiments/observed_b4_covariance_child/run_seed.slurm
```

The runner refuses an existing seed directory. It durably writes scores,
diagnostics, fitted covariance matrices, source and result hashes, Slurm and
compute provenance, a strict-JSON completion record, or a strict-JSON failure
record. The official CIFAR test split remains untouched confirmation data.
