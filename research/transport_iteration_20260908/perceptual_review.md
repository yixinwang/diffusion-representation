# Frozen native perceptual appendix audit

Job **45571074 completed, exit 0**, on v024 in 3:07; batch peak RSS943,048KiB (~921MiB). It uses frozen source **22ac114f90f5250fddced652c6a3bccc8942dd9d** and the pinned FID-specific Inception weights SHA256 `6726825d0af5f729cebd5821db510b11b1cfad8faad88a03f1befd49fb9129b2`. This is a reused-development appendix, not confirmation or a benchmark/SOTA study.

All output payload hashes and 9 source Git hashes verify. The pinned input terminal-status hash and full declared native input inventory match the preserved native study. The 1,000 repair IDs and data ledger match exactly, without rereading pixels. Independently recomputed all reported full-bank polynomial KID and PRDC statistics from the saved float 32 features using a separate NumPy implementation, including strict neighborhoods, duplicate-row zero distances, and tie counts. No feature extractor, model, image-loading path or training was run by this audit. Machine checks and exact discrepancies are in `work/psc-perceptual/machinecheck.json`; full results `work/psc-perceptual/20260909-native-perceptual`, audit script `work/audit-perceptual-results.py`.

Every arm uses its entire 2,000 generated-image feature bank against the same 1,000 reused-repair features. External pretrained information is common evaluation-only information; it was not supplied to model fitting or selection. All seven prescribed arms remain reported.

| Arm | KID (lower better) | Precision | Recall | Density | Coverage |
|---|---:|---:|---:|---:|---:|
| Analysis only |0.276993|0.2415|0.000|0.0790|0.034|
| Coupling |0.150935|0.7010|0.002|0.4311|0.141|
| Residual FM4 |0.140904|0.6540|0.017|0.3901|0.154|
| Residual FM8 |0.148181|0.5950|0.007|0.3242|0.145|
| Residual FM16 |0.152523|0.5635|0.005|0.2930|0.131|
| Residual FM32 |0.153901|0.5540|0.005|0.2863|0.133|
| Residual FM64 |0.154237|0.5545|0.005|0.2845|0.132|

**No uniform quality win:** coupling has worse KID and coverage than FM4andFM8, and worse recall than every residual-FM arm. Coupling has higher precision/density than those arms and better KID than FM16/32/64, so the outcome is mixed. Recall and coverage are low throughout; these feature-neighborhood metrics are descriptive, not probability-support theorems. They do not establish VAE representation superiority or superiority over modern latent diffusion/flow systems.

The separate deterministic first 500-versus-last 500 real-feature control has KID0.00015214, precision 0.78, recall 0.79, density 0.996 and coverage 0.972. It is a smaller 500/500 pair with possible class-composition differences, not a same-size calibrated floor or success threshold. No FID or significance claims are made. Single-training-seed results, reused development records, small banks, and selected shared model architecture limit inference. These scores may not be fed back into selection while calling this same study frozen.

Exact audit totals: 11 payload hashes, 9 source hashes; maximum absolute recomputed metric discrepancy 2.220446049250313e-15.
