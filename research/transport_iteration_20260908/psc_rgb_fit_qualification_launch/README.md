# FIT-only codec qualification launch

Job **45636236** was submitted once from frozen commit `6ae48a91706c97e65c833f94bd51a7b54f74c653`. At this launch capture it is **pending resources**, with no real-image result. One V10032, four CPUs, 16GB host memory and a one-hour allocation were requested through `salloc`, followed by one `srun` worker and immediate allocation release. Account `cis260243p` is authorized; requested QOS `gpuinteract` must be distinguished from the scheduler's effective QOS when known. The 600-second FIT stage and explicit final overrun are specified in the frozen protocol.

A fresh sparse checkout was independently prepared in `mth260022p`. All 11 source SHA256 values and Git bytes matched; the original dirty user checkout was unchanged. Both staging and output parent have verified Lustre project559736 and passed exclusive write/fsync/read/remove probes. No old-project output or Git mutation occurred.

This is a launch receipt, not a pass or generation result. Root's 14 local fabricated checks passed before source freeze. The companion Review21 summary and frozen runner document the algorithm, gates, FIT-only scope and independent replay plan. Later results must retain the same source/seed/gates and preserve failures.

The included allocation logs are submission-time snapshots; final logs will supersede their status while retaining this history.
