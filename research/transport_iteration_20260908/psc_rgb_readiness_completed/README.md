# Complete data-free codec/FM readiness, job 45633712

The one-hour request completed successfully on V100-32 node v021 and released after **3:07** (worker3:04). The runner's measured scope took **102.13 seconds**. Its 10 preflight tests passed in13.47seconds. The difference between worker/allocation/runner times is retained; none is substituted for another. Requested QOS was gpuinteract, but PSC recorded effective QOS gpu. The allocation waited approximately95seconds and was not duplicated.

All53 transferred files (52,665,894bytes),45 original status-manifest payload hashes and7 frozen source files were authenticated against source84c0cad383be64d6632a5fa61e5d1bc4b605ca5f. Root independently verified the hashes, recomputed timing summaries and saved-cache moments, and checked exact repeated-source and checkpoint-reload outputs. All checks passed. Full original checkpoints, synthetic arrays, worker logs, startup errors and source bytes are included; nothing from the authenticated capture is omitted.

## Measured component timings

These are synthetic inputs and briefly updated models, not trained image generators or a matched-quality comparison. Twenty measured batch32 training steps follow five warmups. Each listed batch125 value is a single readiness step, not warmed throughput.

| Component | Batch32 training median | Batch32 training p90 | One batch125 step |
|---|---:|---:|---:|
| Codec |27.52ms|28.85ms|650.03ms|
| Latent field, cached input |22.27ms|22.70ms|32.89ms|
| Pixel field |94.90ms|95.65ms|787.60ms|

With32Heun steps/64field calls, median complete sampling time for32synthetic outputs was **0.41144s for latent FM including its decoder**, and **1.86909s for pixel FM**. At batch1 the medians were0.40309s and0.60739s. There are only three measured repeats; raw timings and warmups are preserved. No significance or candidate speed/quality claim follows.

Allocated training peaks forbatch125 were approximately1.925GiB(codec),0.313GiB(latentfield) and5.444GiB(pixelfield). Reserved memory retains earlier allocator blocks and is reported separately in raw results; it is not fresh incremental model memory. The latent encoder remains resident during sampling. Shared synthetic inputs/cache and host finite-check costs remain included in their disclosed scopes. Optimizer construction is outside individual-step timers but inside whole-screen elapsed time.

The256-image synthetic normalized cache required1,048,704bytes and0.10435seconds including encoding, transfer andwrite. Independently recomputed per-channel mean error was3.01e-8 and population-variance error8.81e-8. This is not a measured4000-image cache cost. All intended gradients passed finite checks, and same-device saved-source/checkpoint outputs matched exactly.

## Limits and failures retained

No canonical image loader, real-image training, development/test scoring, VAE comparison, or candidate comparison was run. This qualifies mechanical readiness of the specified components; it does not establish adequate optimization, the review's proposed3000-second budget, a reconstruction floor on realimages, equal-dimension representation quality, or superiority to latent diffusion/FM.

The first pre-allocation wrapper failed on loginPython3.6's unsupportedsubprocess keyword, before anyallocation. The compatibility fix left the frozen model/profile source unchanged. An scp retrieval attempt failed; the subsequent read-onlySSH transfer was checked against the exact archive hash. Neither error caused another allocation or an experimental retry.
