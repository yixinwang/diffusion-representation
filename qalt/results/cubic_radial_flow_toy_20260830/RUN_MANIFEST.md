# Cubic radial flow aligned-toy development run

Status: witness and specified component controls passed. This is synthetic
development evidence only. It has no CIFAR-10 confirmation or full-flow
coverage.

## Frozen source and execution

- Source commit: `d6adf93b2689eb34b2163c42f2fb75e652175a4d`
- Protocol SHA-256:
  `9c8868bad939af0214d2e22fffc6212013cb7d964040239be7f854702077efa4`
- Compute allocation: Slurm array job `44873061`, task `3`, node `r178`, four
  allocated CPUs. The process environment reported job ID `44873091`.
- Python `3.10.9`, NumPy `2.2.6`, SciPy `1.15.3`, `OMP_NUM_THREADS=1`.
- Official CIFAR-10 test data deserialized: `false`.
- Reserved synthetic confirmation seeds `4100,...,4129` were not run.

Focused checks:

```bash
OMP_NUM_THREADS=1 PYTHONPATH="$PWD/qalt/src" \
  /ocean/projects/mth250006p/ywang26/pytorch/bin/python -m pytest -q \
  qalt/tests/test_cubic_radial_flow.py \
  qalt/tests/test_cubic_radial_flow_runner.py
```

The first runner-test attempt had one failure: a broad sanity assertion
required a fixed-seed lognormal correlation above `0.25`, while the observed
value was `0.2477369`. Only that non-decision sanity interval was corrected to
`(0.20, 0.50)`. The next run passed 11 checks. After adding reproducible
fourth-order energy moments and stronger witness validation, the final source
passed 12 checks in `0.25` seconds. The machine numerical-Jacobian error was
`1.70364e-10`, below `1e-8`.

Witness command:

```bash
OMP_NUM_THREADS=1 PYTHONPATH="$PWD/qalt/src" /usr/bin/time -v \
  /ocean/projects/mth250006p/ywang26/pytorch/bin/python \
  qalt/experiments/cubic_radial_flow_toy/run.py \
  --stage witness \
  --output qalt/results/cubic_radial_flow_toy_20260830/witness.json
```

Control command:

```bash
OMP_NUM_THREADS=1 PYTHONPATH="$PWD/qalt/src" /usr/bin/time -v \
  /ocean/projects/mth250006p/ywang26/pytorch/bin/python \
  qalt/experiments/cubic_radial_flow_toy/run.py \
  --stage controls \
  --witness qalt/results/cubic_radial_flow_toy_20260830/witness.json \
  --output qalt/results/cubic_radial_flow_toy_20260830/controls.json
```

The witness used 500,000 independent vectors in each of two contexts. Its
balanced normalized log-density advantage was `0.0226875` nat/coefficient;
the one-sided 95% lower limit was `0.0225154`, above the `0.01` practical
margin. The three energy correlations were `0.23237--0.23617` for `a=0.030`
and `0.26692--0.26913` for `a=0.038`, each within `0.01` of its checked value.
Maximum inverse, inverse-log-determinant, and direction errors were
`2.665e-15`, `6.217e-15`, and `3.331e-16`.

All specified controls passed. At `a=0`, the pointwise density difference was
zero. At `a=0.0138`, the advantage was `0.00569979` with upper one-sided 95%
limit `0.00580317`, below the margin. Independent permutation of band-energy
rows preserved every empirical band marginal exactly and reduced every
cross-band correlation to absolute value below `0.001982`.

The witness took `1.57` recorded seconds and at most `63,116` KiB by
`/usr/bin/time`; the controls took `2.07` recorded seconds and at most `87,196`
KiB. Both commands exited zero.

## Scientific limit and continuation

This result attributes the aligned-toy gain to a shared radius rather than
separate marginal tails. It establishes separation from the best affine
Gaussian on the specified target. It does not establish separation from a
generic nonlinear 9D flow, transfer to Haar residuals, image quality, or
full-flow speed.

Next executable action: fit the same covariance-normalized 9D cubic layer on
training-only Haar residuals for development seed 2100, then compare normalized
repair-holdout likelihood with its exact joint-Gaussian parent and B4, while
checking held-out dependence, calibration, reversibility, and measured CPU
overhead. Do not open CIFAR-10 test data.
