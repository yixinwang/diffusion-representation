# Complete output-quota failure, job 45619353

The unchanged v2 study at source `3b8c0f7bef604a90ff329fd9608639527b67a972` failed after 58:45 on V100 node v024. All 34 preflight tests passed. The disk quota was exceeded while writing progress for seed 78203 S42; the last preserved progress reports 1,580 updates. The attempted fallback checkpoint also failed. Partial temporary files are retained as failures, not usable checkpoints.

Exactly 20 final arm checkpoints exist. Neither `ALL_FITS_FROZEN.json` nor `ALL_NUMERICS_ADMITTED.json` exists. There are no numerical, generated, feature, or quality banks. No partial arm is admissible for evaluation or reuse. This is an infrastructure failure and supplies no quality or algorithm superiority result.

`original/` preserves the full 180-file result directory and original Slurm output. All 178 hashes in the original status payload manifest were checked, and all 94 archived sources match the frozen Git revision. The two status/failure records are outside their own payload manifest and are independently hashed here. The full original temporary-file failure is included; no upstream outputs were omitted. Authentication scripts and retrieval receipts accompany the data. The retrieval script is historical evidence, not a command to rerun this experiment.

A fresh full study must use an authorized output project with verified write access and room, including its scheduler output path. Seeds, stage budgets, scientific gates, and production source remain fixed. Old partial fits will not be resumed. Allocation balances alone do not establish filesystem quota headroom.
