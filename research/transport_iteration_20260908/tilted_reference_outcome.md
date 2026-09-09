# PSC replication: recovery passes, speed comparison is mixed

The frozen three-seed reference and 32-condition recovery study completed in PSC job 45572966. All 35 reported sign/index selections are correct, and all numerical checks pass. The independent report audit verified the 83-file payload, five source snapshots, all recorded scores and all 324 timing observations. Cross-platform regeneration does not reproduce all source/observation hashes; the review states exactly what was and was not independently reconstructed.

Unlike the local CPU result, the PSC CPU analytic sampler is slower than four-call Heun in every seed and at both batch sizes. At batch 64, exact sampling takes 31.47–31.60 ms versus 26.33–26.54 ms for Heun4. Exact sampling is faster than Heun8 and above. Every solver point remains in the report.

The theoretical quality separation is still scoped to the finite tilted-normal family and fixed canonical Gaussian-reference velocity with the declared Heun solver. Its arithmetic lower bound does not depend on this processor's timing. Combining a quality bound with a conditional operation-cost inequality does not prove a hardware-independent runtime gain. No native-image, video or neural-representation superiority follows from this synthetic study.
