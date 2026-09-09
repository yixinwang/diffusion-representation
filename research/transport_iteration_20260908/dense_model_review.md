# Frozen checkpoint dense-backend audit

Job **45573581 completed, exit 0**, on v024 in 3:48; batch peak RSS1,280,532KiB (~1.221GiB). Source commit **3d801dcc1508cc3e203df5ef97c3834f2462c50e** includes the reviewed launcher warning-delivery fix. No retraining, optimizer step, observations, or new generation-quality evaluation occurred in this audit.

Preservation/source audit passes: all 37 payload hashes,16 source snapshots and Git hashes, pinned native terminal-status hash and 4 accessed native payload hashes verify. The common saved Gaussian array exactly matches the allowed Gaussian field in the frozen native source chunk. No image field was deserialized. Compiler caches/tmp remain on PSC and are explicitly excluded from the scientific payload manifest. Results are local under `work/psc-dense-model/20260909-dense-model`; portable audit `work/audit-dense-model-results.py`, machine checks `work/psc-dense-model/machinecheck.json`.

Independently compared all 162 saved output/value/gradient arrays against the saved reference. Largest pipeline pixel discrepancy is 3.7253e-6 (limit 1e-4), largest parameter-gradient discrepancy 1.3271e-8 (allowance 1e-4+1e-3|reference|). All declared encoding, logdet, NLL and loss tolerances pass. Recomputed conditional-noise and analysis-code roundtrip errors directly from preserved arrays:6.1989e-6 and 1.03235e-4, both below 1e-3. Recorded conditional/analysis logdet cancellation errors are 1.83105e-4/1.46484e-3, both below 1e-2. Forward logdet arrays needed to independently recompute these two cancellation errors were not saved; their source-enforced gates/provenance verify. The job explicitly makes no coarse-Heun inverse or full-source-inverse claim.

All 270timing samples and 90 seeded order permutations verify. Every summary median/peak recomputes exactly; no model call or timing was rerun. Three cases each have 30 repetitions for reference, dense eager and dense compiled, after 10 warmups.

| Case | Reference median (ms) | Dense eager (ms) | Dense compiled (ms) | Reference/compiled ratio |
|---|---:|---:|---:|---:|
| Full generation, batch 1 |56.684|59.761|52.527|1.079|
| Full generation, batch 64 |121.686|118.146|92.190|1.320|
| Generated-input encode/NLL/backward, batch 32 |56.227|42.375|16.179|3.475|

Post-call finite-value/gradient checks were measured separately from the durations above. Including those checks gives reference/compiled medians 56.794/52.638ms,121.807/92.305ms and 59.335/19.273ms respectively; the last ratio is about 3.079. Do not present 3.475 as a fully checked training-loop acceleration. Inter-arm device movement also lies outside those timing samples and is preserved separately (median approximately 5.3–6.8ms per switch). Inference output checks and dense validity checks internal to the generator remain charged within the call.

**Adverse/limited findings:** dense eager is slower than reference at batch 1. Pipeline incremental peak memory is unchanged by compilation:1,649,664 bytes at batch 1 and 106,588,160 bytes at batch 64 for every arm. Generated-input backward incremental peak decreases from 186,740,224to 103,013,376 bytes for compiled (55.16% of reference), while dense eager slightly increases it to 188,275,200 bytes. Persistent inputs/parameters and cached allocator state remain in baselines, so these are not total deployment-memory estimates.

Compiled first uses total 51.9145 seconds, separately from warmups/movement. This includes shape specialization and execution, not a pure compilation invoice. The small standalone kernel's 16–17×ratio does **not** carry over to the complete generator: the measured generation gain is only 1.08–1.32× here. Backward on fixed generated inputs is not a production training loop and does not show equal optimization trajectories, training time-to-quality, or successful retraining.

The audit supports numerical compatibility and bounded timing benefits for the preserved checkpoint/backend on this hardware. It does not establish superior generation quality, an algorithmic complexity theorem, or a favorable amortization threshold. No further job or source change was made.

Absolute and incremental memory maxima from all steady calls (MiB); reserved values include allocator caching:

| Case/arm | Allocated peak | Reserved peak | Incremental allocated peak | Incremental reserved peak |
|---|---:|---:|---:|---:|
|pipeline_1_reference|22.676|236.000|1.573|0.000|
|pipeline_1_dense_eager|22.676|236.000|1.573|0.000|
|pipeline_1_dense_compiled|22.676|236.000|1.573|0.000|
|pipeline_64_reference|122.753|236.000|101.650|0.000|
|pipeline_64_dense_eager|122.753|236.000|101.650|0.000|
|pipeline_64_dense_compiled|122.753|236.000|101.650|0.000|
|encode_nll_backward_32_reference|198.150|236.000|178.089|0.000|
|encode_nll_backward_32_dense_eager|199.614|236.000|179.553|0.000|
|encode_nll_backward_32_dense_compiled|118.302|236.000|98.241|0.000|
