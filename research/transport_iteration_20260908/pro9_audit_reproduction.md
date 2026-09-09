# Pro9 delivered audit reproduction

**PASS for the unchanged delivered CPU fabricated-input audit.** All 13 files match Git blobs at local cherry-pick `c61e1a08a964d05820be38ef62752f26759aafb1` (delivered remote revision `ceafcce57628076ee43cbc91bab5d1e459697176`). All 12 SHA256SUMS entries verify. The original delivery and its outputs were copied intact to `work/pro9-reference-rerun/original`; execution used a separate folder and did not overwrite delivered results.

The reference SHA256 is `ca6158776bf49fd9425aba46a865f251694a7d162419c44e9e7d8ca3dc95ab28`; auditor SHA256 is `a11fe4e1c7e1a6bb43626dfd92f21beaa7ea116cbc3bb98054b4bf12993dab8b`. The audit script was inspected before execution: it imports the local reference, creates fabricated tensors, runs scalar quadrature/gradient/Jacobian/invariance checks, and writes only its adjacent JSON/stdout. It has no native-data loader, network call, optimizer, checkpoint read, or scheduler action. The local command used the existing `work/venv312/bin/python` with numerical libraries restricted to one thread. No source or assertion was patched.

All **18 check groups passed** under Python 3.12.14 / Torch 2.14.0, CUDA unavailable. The delivered output used Python 3.13.5 / Torch 2.10.0+cpu. Platform-dependent small numerical differences are preserved in `work/pro9-reference-rerun/comparison.json`, not relabeled as the original result. Local stdout parses exactly to the new audit JSON; stderr is empty.

| Check | Delivered final | Independent unchanged rerun |
|---|---:|---:|
| Residual-module trainable parameters |200,436|200,436|
| Duplicate float32 frame-cache bytes |184,320|184,320|
| Fabricated (2,45,8,8) inverse max error |7.1526e-7|4.7684e-7|
| Fabricated-shape logdet cancellation |0|9.5367e-7|
| Tiny 32-coordinate dense-Jacobian logdet error |2.2204e-16|2.2204e-16|
| Independent scalar density integral |1|1|

The byte count independently agrees with **4 blocks × 720 active coordinates × rank 16 × 4 bytes**. It is duplicate basis-cache storage only, not total inference memory. Parameter count is for the residual module, not the whole generator. The final count exceeds the retained pre-rotation count of199,956 by480 parameters, agreeing with four rank-16 skew-rotation parameter vectors of16×15/2 entries.

The rerun preserves exact bitwise same-noise-copy outputs/logdets, finite parameter gradients, independent NumPy forward/inverse comparisons, scalar finite-difference gradient checks (including the zero-slope case), tiny full Jacobian agreement, masked current/future-input invariance, nonzero within-block dependence, exact scalar ablation, invalid-input rejection, and cached/uncached equality. The cache tests cover invalidation on training-mode entry and checkpoint loading.

## Failure history and limits

The first retained attempt failed the exact cached/uncached output assertion. The delivery explicitly says its original failed source was not separately snapshotted, so the retained traceback and auditor do not constitute a fully reproducible historical run. The README attributes the fix to making the QR/Cayley frame contiguous without relaxing the equality gate. The pre-rotation results are retained as intermediate history, not evidence about the final source. The final audit and this rerun share exactly the final source/auditor hashes and both pass.

Passing this auditor does **not** establish safety after arbitrary in-place parameter mutation or dtype/device conversion of an already prepared cache: those transitions are absent from its assertions, and the delivered frame lacks comprehensive mutation/version and conversion invalidation. Root reports that the separate production implementation fixes the independently discovered cache/dtype issues. This task did not substitute that production code or claim its tests were reproduced by running the reference auditor. Its verification must remain separate.

No fitting, native images, empirical quality metrics, GPU execution, performance benchmark, PSC job, or commit occurred. These checks substantiate the reported module counts and tested numerical behavior; they provide no full-generator latency, realistic quality, or broad latent-model superiority result.

Preserved machine comparison: `work/pro9-reference-rerun/comparison.json`. Reusable comparison script: `work/compare-pro9-audit.py`; raw independent result/stdout/stderr: `work/pro9-reference-rerun/execution/`.
