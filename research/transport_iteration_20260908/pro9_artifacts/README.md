# Pro9 native dependence repair — review package

Read-only review of yixinwang/diffusion-representation at
bb024437a761c096375a042239f77262724dca14.

**Main recommendation:** an exact-root, cached-prefix conditional decoder with
learned rank16 global innovation mixing, a nine-output direct real-line
piecewise-quadratic scalar head, and a final joint analysis/decoder full-model likelihood phase (root weights fixed).
The native success claim is not established. This package contains no native
images, trained weights, external evaluator weights, data loader, cluster
launcher, repository write, or completed GPU benchmark.

## Files

- `reference.py`: self-contained PyTorch conditional residual module, analytic
  forward/inverse/logdet, cache and masking logic, and an in-memory replacement
  helper for the repo's `GlobalInnovationFlow.residual_decoder` interface.
- `audit.py`: fabricated-input CPU audit with independent NumPy formulas,
  scalar density quadrature, finite-difference gradient checks, tiny dense
  Jacobian, source inversion, current/future leakage and exact-copy/cache tests.
- `audit_results.json`: actual final audit output, including source hashes.
- `audit_stdout.txt`, `audit_stderr.txt`: final audit streams.
- `integration.py`: prospective phase/whole-parameter-matching helpers. These
  are syntax checked, but complete integration with the pinned repo is not run.
- `THEORY.md`: full construction, inverses, approximation bound, structural KL
  identity, non-Gaussian positive family, learning/quality bridge and primary
  references. No general superiority theorem over FM or exact copies.
- `PROTOCOL.md`: one proposed frozen three-seed, staged native falsification
  experiment. It is not an executed or committed registration.
- `audit_attempt1_stderr.txt`, `audit_attempt1.py.txt`: retained earlier cache
  bitwise-mismatch trace and auditor. The original pre-patch module was not
  separately snapshotted; these are not a complete reproducible historical run.
- `audit_pre_rotation_results.json`: intermediate local result, not final code
  provenance. Final results and source hashes are in `audit_results.json`.
- `SHA256SUMS`: delivery file hashes; excludes itself.

## Actual local result

Python3.13.5, PyTorch2.10.0+cpu; CUDA unavailable. Fabricated-input CPU audit PASS.
At the RGB32 residual shape (2,45,8,8), not native images: max inverse error
7.152557373046875e-7, logdet cancellation0, finite parameter gradients. The
module has200436 parameters and184320 bytes of duplicate float32 basis cache.
These are RESIDUAL-module quantities, not whole-generator costs. The32D tiny
model's analytic log determinant matches a full dense Jacobian to2.22e-16.
No latency or GPU-efficiency claim is supported by these tests.

An initial exact-cache-output check failed because cached and uncached QR frames
had different tensor layouts and hence slightly different floating-point GEMM
results. Returning a contiguous frame fixed this without relaxing the bitwise
criterion. The final frame includes a learned small Cayley rotation: an
unrotated QR subspace is not a general eigenbasis. Cache invalidation on checkpoint
loading is explicitly tested. All final tests are rerun on the final module.

## Reproduce the CPU audit

Requires Python, torch, numpy and scipy already available in the environment.
It uses no internet or native data and does not fit a model.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python audit.py
```

`audit.py` writes `audit_results.json` next to itself. The delivery manifest
covers the delivered result; rerunning may change platform-specific values or
metadata. A different result is not to be relabeled as the delivered run.

## In-memory interface example (prospective, not executed here)

On an independently verified pinned checkout with qalt importable:

```python
from qalt.global_innovation_flow import GlobalInnovationFlow
from reference import replace_in_memory
from integration import set_endpoint_phase, joint_endpoint_loss

base = GlobalInnovationFlow()  # exact root, no coarse Heun sampler
candidate = replace_in_memory(base, width=32, rank=16)
# Load/copy only the independently checked shared A/root prefix state as the
# registered runner specifies. This example intentionally does not load data.
params = set_endpoint_phase(candidate, "joint")
# Discard old analysis-code caches; create the registered optimizer from params.
# loss = joint_endpoint_loss(candidate, current_fit_logits)
```

The repository wrapper supplies the retained analysis and learned exact root;
this package does not replace their existing implementation. The original
source module is not overwritten. Native source guarding, full root/A numerical
integration, canonical loading, the target-GPU preflight and the registered
study runner remain work for the experiment implementation, not completed work
in this review.
