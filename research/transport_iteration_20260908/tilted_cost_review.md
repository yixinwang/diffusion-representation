The frozen CPU sampler benchmark completed and its saved numerical results passed independent audit. This is a comparison of implementations of one archived fitted model (gamma 0.5, dictionary index 7), not three independent fits, a new quality study, or evidence of uniform latent-model superiority.

Job 45576004 completed with exit 0:0 in 7:00 on r026, reserving two CPU cores and 4000M for a 20-minute limit; numerical computation used one thread. Source commit: `9391d3ab89ad0667036d39a85f2e59409da34dc6`. Submission/accounting, original log, machine audit, and original source closure are preserved alongside this report.

All 129 payload hashes and ten source snapshots match the frozen Git revision. The original reference source and fitted artifact remain byte-identical to the archived Pro8 files. All six registered Gaussian source banks regenerate byte-exact locally. The public dictionary is orthonormal to the checked tolerance. All 102 saved output arrays have the expected finite float64 shape. The audit independently recomputed 66 saved-output parity comparisons and twelve exact inverse roundtrips: maximum parity error 4.44e-16 and roundtrip error 6.66e-15. Replaying the 36 original-reference outputs from their frozen inputs differed by at most 4.44e-16. No fitting or benchmark timing was rerun.

Timing rows comprise 3060 primary calls (17 arms × 30 repetitions × six source/batch cases), plus 900 separately labeled amortized calls (five cached all-call Heun N values × 30 × six). All primary permutation orders, continuing secondary permutation streams, row counts and reported primary medians match independently. Each arm receives five warmups. Primary calls charge construction of schedules, the root transform, summary, Haar transform, sigmoid and output validity check. Secondary calls precompute schedules outside the timed call; their setup costs must remain separate. These are cached all-call Heun evaluations with N actual field calls, including both endpoints. Independent untimed reconstruction of all 30 corresponding source/N outputs agrees with saved cached_fm_N outputs to 3.33e-16; it does not use endpoint specialization. These observations are descriptive measurements on one CPU allocation, not asymptotic complexity evidence.

Median primary wall times in milliseconds, shown as the minimum–maximum of the three sampling-seed medians:

| Arm | Batch 1 | Batch 64 |
|---|---:|---:|
| original_exact | 0.7039–0.7732 | 34.8995–39.3015 |
| central_exact | 0.5482–0.6032 | 21.5243–24.3248 |
| original_fm_4 | 0.6608–0.7183 | 30.6080–34.7646 |
| original_fm_8 | 1.0837–1.1872 | 54.1486–60.4360 |
| original_fm_16 | 1.8843–2.0707 | 99.2459–111.1201 |
| original_fm_32 | 3.4672–3.8079 | 189.9802–212.1031 |
| original_fm_64 | 6.5940–7.2529 | 371.3522–416.7029 |
| cached_fm_4 | 0.7051–0.7742 | 30.1713–34.1345 |
| cached_fm_8 | 1.1006–1.2157 | 53.6160–60.1019 |
| cached_fm_16 | 1.8398–2.0383 | 98.6249–110.4203 |
| cached_fm_32 | 3.3046–3.6661 | 189.2575–211.3811 |
| cached_fm_64 | 6.2138–6.9314 | 370.6220–413.8574 |
| endpoint_fm_4 | 0.5388–0.5913 | 20.0542–22.6374 |
| endpoint_fm_8 | 0.9385–1.0322 | 44.1128–49.1876 |
| endpoint_fm_16 | 1.6816–1.8540 | 89.1986–99.2302 |
| endpoint_fm_32 | 3.1481–3.4914 | 179.6624–200.4152 |
| endpoint_fm_64 | 6.0458–6.7326 | 360.8973–402.4482 |

Central exact improves over original exact by 1.264–1.621× across all six cases. However endpoint-specialized Heun4 is still 1.71–7.85% faster than central exact in every case. Cached all-call Heun4 is slower than original Heun4 at batch one for every seed. Caching therefore does not establish a universal speed improvement. All arms and adverse cases remain in the table and raw records.

The endpoint implementation has N mathematical stages and N−2 actual nontrivial field-kernel calls, with two analytic endpoint stages; the final predictor is not allocated. Independent counted dispatches verified 2, 6, 14, 30 and 62 calls for N=4,8,16,32,64, while ordinary cached Heun makes N calls. Calling the endpoint implementation “N field evaluations” without this distinction would overstate its compute. Both optimized exact and optimized Heun comparators use the same optimized exact root; Heun parity here is with the original same-N approximation, not a claim that Heun equals the exact output.

Secondary schedule-amortized cached all-call Heun medians in milliseconds (schedule setup excluded):

| N | Batch 1 | Batch 64 |
|---|---:|---:|
| 4 | 0.6671–0.7094 | 28.1299–33.0987 |
| 8 | 1.0465–1.1375 | 50.7991–58.7040 |
| 16 | 1.7692–1.9374 | 94.1657–108.5123 |
| 32 | 3.2050–3.5359 | 181.8230–209.7138 |
| 64 | 6.0431–6.6217 | 356.7141–412.5690 |

Preparing all five schedules costs 0.1233–0.3582 ms across six recorded setup calls. No break-even is inferred from these values.

Memory diagnostics are separate tracemalloc peak increments above their baseline, not whole-process allocation or intrinsic model complexity. Full per-arm diagnostics remain preserved. For the principal arms, minimum–maximum bytes across sampling seeds:

| Arm | Batch 1 traced increment | Batch 64 traced increment |
|---|---:|---:|
| original_exact | 210,617–210,617 | 11,896,881–11,896,938 |
| central_exact | 285,913–285,913 | 16,689,713–16,689,770 |
| original_fm_4 | 165,377–165,441 | 10,488,889–10,489,057 |
| cached_fm_4 | 188,993–189,138 | 11,897,529–11,897,842 |
| endpoint_fm_4 | 165,825–165,882 | 10,422,777–10,422,834 |

Each output itself uses 24,576 bytes at batch one or 1,572,864 bytes at batch 64. Slurm sampled batch MaxRSS is 117,780 KiB; the process-reported high-water RSS is 150,536 KiB. These are different measurement mechanisms and should not be interchanged. Central exact has a larger traced memory increment than original exact in these diagnostics, so this study does not support a uniform memory win.

The original full result tree occupies 86,955,735 bytes. `psc_tilted_cost` publishes an explicit subset with all 51 batch-one actual outputs, all six Gaussian banks, dictionary, all source snapshots, all raw timing rows and numerical records. Its original closure is named `upstream_COMPLETE.json`; the 51 batch-64 output hashes and full local/PSC locations are recorded in `PUBLICATION_SUBSET.json`. The full archive was audited before subsetting and remains available at those locations. The portable checker requires that full archive, and performs no fitting or timing rerun.

Example from the repository root: `python research/transport_iteration_20260908/check_tilted_cost_results.py --results /path/to/full/20260909-tilted-cost --repo . --output /new/path/audit.json`. The checker uses NumPy/SciPy and frozen source snapshots; no real data are imported.
