# Independent four-case GPU synthetic-generator audit

Frozen source `2b58285826a74e32882a6de0af2a1d951a85ba58`; reported PSC job 45558567 on V100/v024. All four registered cases and all three fitted arms per case are present. This audit did not contact PSC or independently query scheduler/GPU metadata.

All 12 distinct source hashes/Git blobs verify in each case. All 40 expected saved artifacts are present; JSON snapshots agree, checkpoints contain finite values and the reported parameter counts, and all 20 generated arrays have the prescribed shape/dtype with finite values. Every saved copied-stochastic-latent sample array is bitwise equal to its saved spline sample array.

The older runner has no original COMPLETE/payload-hash manifest. A fresh local hash/size ledger for all downloaded artifacts is included in the machine-check output. This is preservation plus cross-file/recomputation evidence, not verification against a nonexistent remote payload ledger.

## Independent evaluation recomputation

Only the four frozen evaluation seeds were recreated using an independent implementation of the recorded local/distant teacher. Independent NumPy/SciPy score formulas recomputed all 120 metrics from saved generated samples. Maximum absolute discrepancy: `7.91624188e-09`. No fitting, training-stream regeneration, new seeds/cells, real datasets, or checkpoint sampling occurred. Local torch 2.14.0 differs from PSC 2.10.0+cu128; a bitwise evaluation-array match cannot be checked because original evaluation arrays/hashes were not retained.

The four-channel cases are synthetic multichannel arrays, not real video, and their 8x8 spatial field is small enough for the networks to have broad receptive fields. The distant case is a stress test, not a demonstrated limitation of each neural architecture. The hierarchical FM retains fixed Haar analysis; it is not the newer learned-analysis/global-decoder baseline.

## Numerical gates (saved records)

| Case | Gaussian-source RT max | Logdet cancellation max | Saved copied samples |
|---|---:|---:|---|
| local_image | 4.8875809e-06 | 3.0517578e-05 | bitwise equal |
| distant_image | 5.1856041e-06 | 2.2888184e-05 | bitwise equal |
| local_multiframe | 5.6326389e-06 | 9.1552734e-05 | bitwise equal |
| distant_multiframe | 6.1392784e-06 | 9.1552734e-05 | bitwise equal |

All recorded numerical gates pass fixed limits .001 for source roundtrip and .01 for logdet cancellation. These values were integrity/threshold checked, not independently recomputed by sampling checkpoints. Same prior coordinate accounting is 64 for one-channel and 256 for four-channel arrays; the copied decoder has no extra uncounted noise.

## Complete descriptive score and cost table

All fixed scales and Heun grids are retained. Lower energy/moment/MMD is preferable, but there is one training seed per case and only 256 evaluation tensors; no significance or uniform quality ordering is established. Generation timings are one batch, not repeated latency measurements.

| Case | Arm/grid | Energy | Mean RMS | Second moment | MMD .1 | MMD .3 | MMD1.0 | Seconds | Calls |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| local_image | spline | 0.17894858 | 0.02816222 | 0.01911403 | 0.00045249 | 0.00346313 | 0.00025002 | 0.040610 | 0 |
| local_image | full_fm_heun_8 | 0.17964212 | 0.02565966 | 0.03391557 | 0.00363105 | 0.00688491 | 0.00027764 | 0.696860 | 16 |
| local_image | full_fm_heun_32 | 0.17976137 | 0.02558266 | 0.03378176 | 0.00422874 | 0.00744744 | 0.00028850 | 0.015875 | 64 |
| local_image | hierarchical_fm_heun_8 | 0.17888330 | 0.02883369 | 0.00684758 | 0.00053308 | 0.00243132 | 0.00031429 | 0.015935 | 48 |
| local_image | hierarchical_fm_heun_32 | 0.17893479 | 0.02871463 | 0.00731369 | 0.00080651 | 0.00267775 | 0.00031654 | 0.041519 | 192 |
| distant_image | spline | 0.18889643 | 0.03295497 | 0.01740049 | 0.00054528 | 0.00380706 | 0.00045135 | 0.030733 | 0 |
| distant_image | full_fm_heun_8 | 0.18964666 | 0.03554308 | 0.03533905 | 0.00227617 | 0.00707718 | 0.00071696 | 0.019101 | 16 |
| distant_image | full_fm_heun_32 | 0.18966993 | 0.03553410 | 0.03537422 | 0.00227672 | 0.00721990 | 0.00072373 | 0.016699 | 64 |
| distant_image | hierarchical_fm_heun_8 | 0.18872942 | 0.03315555 | 0.01229491 | 0.00034887 | 0.00244279 | 0.00046900 | 0.016239 | 48 |
| distant_image | hierarchical_fm_heun_32 | 0.18872029 | 0.03296070 | 0.01231558 | 0.00033095 | 0.00241949 | 0.00046242 | 0.043161 | 192 |
| local_multiframe | spline | 0.17929205 | 0.02680544 | 0.01568126 | 0.00055510 | 0.00317009 | 0.00019083 | 0.031565 | 0 |
| local_multiframe | full_fm_heun_8 | 0.17967747 | 0.03203820 | 0.03322558 | 0.00064571 | 0.00423234 | 0.00052327 | 0.016240 | 16 |
| local_multiframe | full_fm_heun_32 | 0.17969926 | 0.03208486 | 0.03320146 | 0.00068418 | 0.00433447 | 0.00052908 | 0.016212 | 64 |
| local_multiframe | hierarchical_fm_heun_8 | 0.18027461 | 0.03837670 | 0.01061832 | 0.00061909 | 0.00649372 | 0.00095290 | 0.016197 | 48 |
| local_multiframe | hierarchical_fm_heun_32 | 0.18033639 | 0.03832642 | 0.01096848 | 0.00077557 | 0.00681926 | 0.00095842 | 0.043171 | 192 |
| distant_multiframe | spline | 0.18840344 | 0.02489197 | 0.01845273 | 0.00026332 | 0.00278248 | 0.00005396 | 0.031433 | 0 |
| distant_multiframe | full_fm_heun_8 | 0.18853755 | 0.02653058 | 0.03513465 | 0.00043269 | 0.00285183 | 0.00019445 | 0.015557 | 16 |
| distant_multiframe | full_fm_heun_32 | 0.18855582 | 0.02593335 | 0.03498882 | 0.00050127 | 0.00302856 | 0.00017583 | 0.016189 | 64 |
| distant_multiframe | hierarchical_fm_heun_8 | 0.18905606 | 0.03200739 | 0.01463639 | 0.00059896 | 0.00492248 | 0.00051864 | 0.015880 | 48 |
| distant_multiframe | hierarchical_fm_heun_32 | 0.18913494 | 0.03190693 | 0.01509496 | 0.00077520 | 0.00535786 | 0.00052392 | 0.041130 | 192 |

Adverse/mixed findings must remain explicit:

- In both image cases, hierarchical FM has lower energy and MMD(.3) than spline. In both multichannel cases, spline has lower energy/MMD(.3), but this is not real-video evidence. Other metrics need not follow those rankings.
- Spline generation takes approximately 31–41ms. Full FM 32 and hierarchical FM 8 each take about 16 ms in these recorded batches; the CPU spline timing advantage does not transfer to this GPU result.
- Spline also uses more measured peak allocated GPU training memory than both FM controls in all four cases: 6.23 MiB versus 2.52/1.46 MiB for image, and 19.36 MiB versus 4.18/3.36 MiB for multichannel. It has substantially more parameters; this experiment does not support a memory-efficiency advantage.
- The local-image full-FM 8 batch takes about 697 ms, while 32 steps take about 16 ms. This inversion is a warmup/cache/timing anomaly warning; it cannot support a clean solver-cost frontier. Preserve both observations without selecting the favorable one.
- Heun 8/32 are step counts: full FM uses 16/64 velocity calls, hierarchical FM 48/192 across three spatial resolutions. Call counts at different resolutions are not equivalent FLOPs or latency.

## Training/resource accounting

| Case | Fitted arm | Parameters | Updates | Training+init seconds | Cap overrun seconds | Peak GPU allocated MiB |
|---|---|---:|---:|---:|---:|---:|
| local_image | spline | 55,390 | 1,364 | 90.010476 | 0.010476 | 6.23 |
| local_image | full_fm | 5,689 | 51,662 | 90.001121 | 0.001121 | 2.52 |
| local_image | hierarchical_fm | 18,463 | 25,389 | 90.002421 | 0.002421 | 1.46 |
| distant_image | spline | 55,390 | 1,959 | 90.015822 | 0.015822 | 6.23 |
| distant_image | full_fm | 5,689 | 51,706 | 90.001200 | 0.001200 | 2.52 |
| distant_image | hierarchical_fm | 18,463 | 24,800 | 90.001171 | 0.001171 | 1.46 |
| local_multiframe | spline | 95,992 | 1,961 | 90.011244 | 0.011244 | 19.36 |
| local_multiframe | full_fm | 6,412 | 51,613 | 90.001567 | 0.001567 | 4.18 |
| local_multiframe | hierarchical_fm | 24,820 | 24,753 | 90.001883 | 0.001883 | 3.36 |
| distant_multiframe | spline | 95,992 | 2,081 | 90.017521 | 0.017521 | 19.36 |
| distant_multiframe | full_fm | 6,412 | 52,714 | 90.000330 | 0.000330 | 4.18 |
| distant_multiframe | hierarchical_fm | 24,820 | 25,221 | 90.000948 | 0.000948 | 3.36 |

All update counts remain below 100,000. The 90-second budget includes initialization and allows a final in-flight update overrun; actual overruns are retained above. These were different-size networks with unequal update counts. Measured PyTorch allocated GPU memory is not total driver/process/device memory. Summed case runtime is 1096.973s; reported scheduler elapsed 18:52 includes launch/testing/other overhead and is not independently queried here.

No theoretical bound is demonstrated by these finite-cap neural fits; all advantages in the original summary remain false. The results establish complete execution, passed recorded numerical gates, reproducible exploratory metrics, and mixed quality/cost outcomes. They do not establish a generation/representation/efficiency win against optimized latent flow/diffusion or the stronger new learned-analysis baseline.

Audit source is portable: `audit_complete_gpu.py --results RESULTS --repo REPO --output OUTPUT --scratch SCRATCH`. The machine-check JSON binds local payload hashes and records all recomputed scores/discrepancies. No repository mutation, commit, checkpoint sampling, real-data access or PSC action occurred.
