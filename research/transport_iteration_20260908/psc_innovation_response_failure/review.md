# Innovation response pilot failure audit

PSC job 45582364 failed with exit 1:0 after 17:35 on v024, in seed 78201 RQS_prefix. Frozen scientific source: `168efc227e361a86a2a6aa6952786e5d0e13e30f`. Preflight passed all 25 focused tests in 153.84 seconds. The thrown error is the dense conditional spline aggregated numerical validation guard during residual log-probability encoding, before loss backward or the next optimizer step.

All 114 status-manifest payload hashes and all 88 copied source hashes/Git blobs pass independently. The full 44,800,000-byte transport archive, original Slurm log, all checkpoints, failure/status JSON and independent file inventory are retained here. No remote originals were changed. The initial local archive extraction used system Python lacking the `filter` API and failed before extraction; extraction was then performed with Python3.12 and the safe data filter. Transfer itself succeeded once.

| Stage | Status | Successful updates | Loop seconds |
|---|---:|---:|---:|
| I_frozen | completed | 1961 | 89.899 |
| I_joint | completed | 426 | 89.961 |
| I_prefix | completed | 1949 | 89.989 |
| P_frozen | completed | 1991 | 89.883 |
| P_joint | completed | 430 | 89.845 |
| P_prefix | completed | 1811 | 89.967 |
| RQS_prefix | failed | 820 | 40.125 |
| analysis | completed | 276 | 83.868 |
| root | completed | 2170 | 85.191 |

The failed minibatch is draw 821 after 820 successful updates. Replaying PCG64 seed 78231 for 821 draws of 32 indices from the registered 4,000 fit IDs reproduces the saved post-draw RNG exactly and the complete record-order SHA256 exactly. Every saved Adam step is 820. `failed-minibatch-identities.npz` contains the exact subset indices and canonical record IDs; `machinecheck.json` includes these in readable form and checkpoint architecture metadata. No optimizer or model calculation was rerun. Actual fit pixels are absent from the saved outputs, so reconstructing numeric input tensors still requires these canonical training records and the frozen record-keyed dequantization/common logit transform; the identities alone are not claimed to reproduce the numerical exception.

No ALL_FITS_FROZEN marker or generated/feature/quality/numerical evaluation payload exists. Only seed78201 was entered. The strict canonical loader did load its selected recycled repair inputs at setup; thus “no repair access” would be inaccurate. However, source control flow and preserved artifacts show no repair evaluation or extractor creation was reached. No protected test or excluded-discovery evaluation occurred. This failed run yields no generation-quality result and cannot be used to favor either learned response arm.
