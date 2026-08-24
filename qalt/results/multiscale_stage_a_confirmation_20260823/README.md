# Multiscale QALT Stage-A confirmation

Frozen commit: `8bd9c61`. Allocated Slurm job: `44295561`. Confirmation seeds: `800..829`, exactly once.

All 16 preregistered paired one-sided Student components passed Holm correction at familywise level 0.05, and all deterministic hard checks passed.

| quantity | image mean [95% CI] | video mean [95% CI] |
|---|---:|---:|
| QALT minus oracle NLL | 0.000121 [0.000081, 0.000162] | 0.000080 [0.000063, 0.000097] |
| exact split minus QALT NLL | 0 [0, 0] | 0 [0, 0] |
| diagonal VAE minus QALT NLL | 0.10163 [0.09995, 0.10332] | 0.10612 [0.10198, 0.11027] |
| coarse-only minus QALT NLL | 0.10541 [0.10326, 0.10755] | 0.11406 [0.11050, 0.11762] |
| finite-Euler minus QALT NLL | 0.000147 [0.000117, 0.000176] | 0.000097 [0.000073, 0.000120] |
| latency ratio | 0.3274 [0.3268, 0.3279] | 0.1894 [0.1890, 0.1897] |
| explicit peak live-array ratio | 0.750 | 0.625 |

NLL is in nats per full endpoint dimension. Latency includes matched local iterative kernels and the shared inverse Haar decoder on the allocated CPU node. Memory counts explicit simultaneous live NumPy arrays, not accelerator allocator telemetry.

The maximum defensible claim is narrow: on the declared nonlinear, non-Gaussian procedural image/video law, a train-fitted equal-dimensional multiscale representation matches the exact full-token endpoint to the registered tolerance with latent-order repeated work and beats fixed diagonal, lossy, and finite-Euler controls. An optimized same-information latent method ties exactly. This result does not establish observed image/video quality, neural-network training behavior, FID/FVD, or production GPU speed.
