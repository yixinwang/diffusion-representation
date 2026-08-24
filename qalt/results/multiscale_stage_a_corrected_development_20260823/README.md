# Corrected Multiscale QALT Stage-A development

This run uses preregistered development seeds `700..704` and includes the shared inverse Haar decoder in both latency paths. It is development evidence on a procedural nonlinear, non-Gaussian image/video law, not confirmation or observed-data evidence.

| quantity | image mean [95% CI] | video mean [95% CI] | frozen gate |
|---|---:|---:|:---|
| QALT minus oracle NLL | 0.000015 [-0.000024, 0.000053] | 0.000044 [-0.000009, 0.000097] | pass |
| exact split minus QALT NLL | 0 [0, 0] | 0 [0, 0] | pass |
| diagonal VAE minus QALT NLL | 0.1030 [0.0956, 0.1105] | 0.1073 [0.0916, 0.1230] | pass |
| coarse-only minus QALT NLL | 0.1074 [0.0987, 0.1161] | 0.1156 [0.1023, 0.1289] | pass |
| finite-Euler minus QALT NLL | 0.000204 [0.000082, 0.000327] | 0.000129 [0.000062, 0.000196] | pass |
| latency ratio | 0.328 [0.325, 0.330] | 0.192 [0.185, 0.199] | pass |
| explicit peak live-array ratio | 0.750 [0.750, 0.750] | 0.625 [0.625, 0.625] | pass |

All comparisons are paired by seed. NLL is in nats per full image/video dimension. The full-token exact baseline and exact same-information split baseline tie QALT exactly, as required by the countertheorem. Strict quality gains apply only to the registered restricted diagonal, lossy, and finite-Euler controls.

The explicit memory ratio counts simultaneously live NumPy coefficient/state arrays, including the endpoint array; it is not GPU allocator telemetry. The latency timer executes the matched local update kernel and shared inverse Haar decoder on the allocated CPU node. Consequently, this stage validates the mechanism and accounting, not production neural-network or accelerator speed.
