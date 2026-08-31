# Observed B4 radial child run record

## Frozen inputs

- Source commit: `a05c4ebb57d99e7e1420e17b36e761d5f7b41a5d`
- Protocol SHA-256: `b74b9e0187e4c68a2413fe4864797fec2b3ae1b35237785a00776f763bc1edbd`
- Development seed: `2100`
- Slurm job: `44873091` (`44873061_3`), node `r178`, four allocated CPUs
- Opened data: the five CIFAR training batches only; `test_batch` was not deserialized

## Command

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=$PWD/qalt/src \
  /usr/bin/time -v \
  /ocean/projects/mth250006p/ywang26/pytorch/bin/python \
  qalt/experiments/observed_b4_radial_child/run.py \
  --seed 2100 \
  --source-commit a05c4ebb57d99e7e1420e17b36e761d5f7b41a5d \
  --output qalt/results/observed_b4_radial_child_20260830/seed_2100
```

## Result

The run completed and failed first at `fit_boundary`. The cubic fit selected
the frozen upper limit `a=0.1`; the interval was not widened. Cubic improved
the B4 conditional NLL by `0.3136291` nat per detail coefficient with
image-level standard error `0.0014847`, but it was worse than the registered
Student rival by `0.1882589` nat per detail coefficient with standard error
`0.0022367`. The Student fit selected `nu=2.1176545`.

The complete normalized-density and reversibility checks passed. The
global-isotropic direction and band-share checks could not improve because
both radial maps preserve those quantities samplewise. Their held-out B4
deviations were `0.0929461` and `0.2205200`; every global isotropic radial law
therefore fails the frozen `0.03` limits on this B4 coordinate system. The
Student map also left a maximum latent band-energy correlation of `0.3360007`.

Compute use was 14 minutes 29.67 seconds wall time, 862.51 seconds user CPU,
1.95 seconds system CPU, and 2,323,104 KiB maximum resident memory. No swap or
major page fault occurred.

## Integrity

`sha256sum -c seed_2100/SHA256SUMS` passed. The checksum file has SHA-256
`9a9900d64e5a1004a59a7009faf7ee29684e845940ce1459d838e4c7bb6301ad`.
`seed_2100/COMPLETE.json` has SHA-256
`827fd24188d45cea7a72c7eae0af2d09691d8f2cc87137f868eb33f194efc69f`.

## Executable continuation

Register and run one covariance-only reversible child. Keep the B4 coordinate
repair, common fitting sites, data split, conditioning strata, and all
information fixed. Replace only the failed global isotropic radial density
layer with a checked nine-dimensional covariance layer so that the revision
can alter the angular and band-share failures. Do not open confirmation data.
