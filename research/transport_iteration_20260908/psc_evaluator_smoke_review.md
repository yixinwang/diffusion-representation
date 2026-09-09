# PSC fabricated evaluator smoke audit

The single authorized job 45569096 completed exit 0 at frozen revision 22ac114f90f5250fddced652c6a3bccc8942dd9d. Slurm elapsed 6:22, reserved 2 CPUs/4000M, batch peak RSS 464,960 KiB (454.06MiB). Torch and BLAS computations were one thread. The script recorded 309.129629 seconds from its start; this is not isolated forward latency and does not determine which setup stage caused the delay.

Downloaded only `smoke.json`, `COMPLETE.json` and the empty job log to `work/psc-evaluator-smoke`. COMPLETE's one payload hash verifies. All three recorded source hashes independently match Git blobs at the frozen commit. The fixed linspace fabricated input hash independently reproduces locally. Local preservation hashes and checks are in `work/psc-evaluator-smoke/machinecheck.json`.

The result records exactly one CPU forward on two fabricated RGB32 inputs, yielding finite float32 features with shape (2,2048), min 0, max 5.068574905395508. Feature-byte SHA256: `3c917b3c4d0a5f7ab887c25e43e0f5e9adae89c67b19d0591a902d3118e101db`. No feature arrays were saved, so this audit verifies the result's provenance and executable checks rather than independently recomputing feature values. The actual frozen program asserts shape/dtype/finiteness before emitting completion.

Dependency versions: Python 3.10.9, NumPy 2.2.6, Torch 2.10.0+cu128, torchvision 0.25.0+cu128. It used the pinned FID-specific source SHA256 `c6183fff54dd240fe66d53d207f4bd28c06fde98c21b5525f10ca0cc5cef7780` and weights SHA256 `6726825d0af5f729cebd5821db510b11b1cfad8faad88a03f1befd49fb9129b2`, loaded through the frozen local-only restricted loader.

This establishes CPU dependency/model-loading/fabricated-forward compatibility in the existing PSC environment. It does not establish GPU compatibility, real-bank throughput, numerical equivalence to another FID implementation, generation quality, or superiority. No actual dataset, generated bank, fitting operation, or second job was used. The original dirty checkout and existing Python environment were untouched; the detached smoke checkout and isolated evaluator assets are separate task directories.
