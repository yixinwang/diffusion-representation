# Tilted-normal synthetic reference audit

Job **45572966 completed, exit 0**, on r263, elapsed 2:51. The two-core/4000M allocation used one numerical-library thread; batch peak RSS 61,104 KiB (~59.67MiB). Perceptual 45571074 was still PENDING with no assigned node in the accompanying status query. No job was resubmitted or altered.

Audit PASS for preservation, source identity, reported selection consistency, numerical gates and timing summaries. All 83 payload hashes match the exact recursive closure. All 5 source snapshots match their recorded hashes and Git blobs at **82bb8bf6dac93e9ad7ea63c70ec7ca4ac77bccfa**. The original reference remains byte-identical to SHA256 **38cb7d9b79111b99e3af961aa28cdf2ca6f5a2b427245434e2cea407c7b6c73b**. The complete small result tree and log are preserved under `work/psc-tilted-reference/20260909-tilted-reference`; audit code is `work/audit-tilted-reference-results.py`, machine checks `work/psc-tilted-reference/machinecheck.json`.

All 35 reported fits selected the correct sign/index: 3 original reference runs at the prescribed seeds plus 32 different sign/index recovery cells. All 32 recovery score vectors have 16 finite entries; independently taking each saved vector's argmax gives the reported/correct index, and all reported score gaps recompute. Reported signs equal the predeclared truths. This is a report/score consistency audit, not a new fit: no fitting function or original timed run was called. Three repeated runs for (+0.5,index 7), plus one run for each of 32 conditions, do not empirically establish an extremely small error probability for every condition.

The three source-roundtrip maxima range 3.9968e-15–8.4377e-15, below 1e-9. Tail-roundtrip error through |source|50 is 1.4211e-14, below 1e-10. Dictionary orthogonality error 7.7716e-16 is below 1e-12. All three reports record a bitwise-equal conditional copy. These passed executable numerical checks are not a global interval proof; successful output arrays were not preserved for independent copy/roundtrip recomputation.

All 324 raw positive timing observations are present (3 seeds × 2 batch sizes × 6 arms × 9 repetitions). Every median and FM/exact ratio independently recomputes exactly. All six prescribed arms and both batch sizes remain included.

| Arm | Batch 1 median range across 3 seeds (ms) | Batch 64 median range across 3 seeds (ms) |
|---|---:|---:|
| Exact |0.6840–0.7421|31.4744–31.6045|
| Heun 4 actual velocity calls |0.6369–0.6999|26.3259–26.5448|
| Heun 8 |1.0669–1.1693|49.0963–49.3234|
| Heun 16 |1.8737–2.0494|92.9887–93.3137|
| Heun 32 |3.4678–3.7853|180.8945–181.6141|
| Heun 64 |6.6210–7.2268|357.9964–358.8996|

**Adverse finding preserved:** Heun 4 is faster than exact in every seed at both batch sizes. Exact is faster than Heun 8/16/32/64. This timing study does not measure a matched-quality frontier; it uses the common fitted finite-family model and locked canonical velocity/grid. It cannot claim faster generation than all flow-matching solvers or trained neural baselines.

## Provenance reconstruction limits

Using the verified original primitive code with `fit` and `run` replaced by functions that raise, the audit regenerated only the existing frozen synthetic streams and observations. It used the preserved public dictionary to avoid changing the dictionary through local QR rounding. The three 9-round interleaving sequences reproduced exactly. Independently regenerated QR dictionary differs by at most 2.01e-16 from the preserved one.

Cross-platform byte matching is incomplete: root source hashes 33/35, head source 10/35, roundtrip source 2/3, batch 1 timing source 3/3, batch64timing source 2/3. The preserved dictionary hash matches 35/35; recreated observation hashes match 0/35 for both root/head arrays. These are exact hash comparisons, not tolerant numerical comparisons: successful original observation/source arrays were not saved, so their numerical discrepancies cannot be measured locally from hashes alone. Different NumPy/SciPy/platform arithmetic is a plausible explanation, but this audit does not establish the exact cause. Source identity, seed/order/count records and published payload preservation are verified; byte-exact cross-platform regeneration of every synthetic array is **not** claimed. No refit, new timing, adaptive evaluation or result selection followed these mismatches.

This validates the frozen synthetic fixture's delivered results and their limits. It provides no native-image/video quality, GPU performance, learned-neural-baseline superiority, unrestricted algorithmic separation or empirical verification of the tiny theoretical probability bound. The separate mathematical/interval audit is not repeated here.
