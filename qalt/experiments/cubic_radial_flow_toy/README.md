# Cubic radial flow aligned toy

This experiment executes the frozen protocol in
`qalt/theory/CUBIC_RADIAL_FLOW_TOY_PROTOCOL.md`. It is an aligned
representational check, not a CIFAR endpoint or confirmation study.

Run the unit checks first:

```bash
OMP_NUM_THREADS=1 PYTHONPATH=qalt/src \
  /ocean/projects/mth250006p/ywang26/pytorch/bin/python -m pytest -q \
  qalt/tests/test_cubic_radial_flow.py \
  qalt/tests/test_cubic_radial_flow_runner.py
```

After committing the source, run the witness:

```bash
OMP_NUM_THREADS=1 PYTHONPATH=qalt/src \
  /ocean/projects/mth250006p/ywang26/pytorch/bin/python \
  qalt/experiments/cubic_radial_flow_toy/run.py \
  --stage witness --output qalt/results/cubic_radial_flow_toy_20260830/witness.json
```

Run the frozen controls only if `all_checks_pass` is true:

```bash
OMP_NUM_THREADS=1 PYTHONPATH=qalt/src \
  /ocean/projects/mth250006p/ywang26/pytorch/bin/python \
  qalt/experiments/cubic_radial_flow_toy/run.py \
  --stage controls \
  --witness qalt/results/cubic_radial_flow_toy_20260830/witness.json \
  --output qalt/results/cubic_radial_flow_toy_20260830/controls.json
```

The runner refuses uncommitted registered source and refuses controls whose
witness metadata or pass decision does not match the frozen child.
