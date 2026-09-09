# Native training finite-check operation audit

This is a source and model-shape audit of `qalt/experiments/observed_flow_pilot/run_shared.py`, not a latency measurement. No data, forward/backward training, or GPU job was used. The frozen native configurations were instantiated on CPU solely to enumerate parameter tensors.

On a successful update, `train_stage` performs one host Boolean conversion of the finite-loss reduction, then one per present gradient, then one per updated parameter. Each CUDA scalar conversion can require a device-to-host transfer and synchronization. The code additionally calls an explicit device synchronize and converts the detached loss to a Python float for logging. Those two actions are separate from the counts below. Model-internal validation, optimizer internals, and periodically written JSON are also excluded from this table.

| Stage | Scalar parameters | Parameter tensors | Finite-gradient scalar reads, when all gradients present | Updated-parameter scalar reads | Finite-loss scalar reads | Total finite-check scalar reads |
|---|---:|---:|---:|---:|---:|---:|
| Retained pre-analysis + multiscale A | 255,952 | 104 | 104 | 104 | 1 | 209 |
| Shared coarse FM | 10,531 | 6 | 6 | 6 | 1 | 13 |
| Four-layer conditional coupling | 261,524 | 48 | 48 | 48 | 1 | 97 |
| Global residual FM, width 124 | 263,421 | 12 | 12 | 12 | 1 | 25 |

The gradient count is exactly the number of parameters with `p.grad is not None`; the all-present column is the architecture's expected fully connected-loss case, not an observed gradient trace. Zero-valued gradients still count if their tensors exist. Failure paths short-circuit the Python `any` and can stop earlier. These counts do not imply that each read waits for an entire new GPU workload, nor do they measure latency.

Despite nearly equal scalar parameter counts, coupling performs four times as many gradient/parameter finite-check host reads as residual FM: 96 versus 24, or 72 extra per successful update. Including the shared single loss check gives 97 versus 25. This is an implementation-dependent audit cost, not an intrinsic mathematical complexity lower bound for either estimator. Because training stops by measured wall time, the overhead can alter update counts and examples processed. The completed comparison remains a valid measurement of those exact checked implementations; it cannot attribute an observed training-speed difference entirely to the underlying algorithms.

A future additive implementation can preserve the same acceptance predicate while aggregating each class of checks on the device. Reduce each present gradient's `isfinite` tensor to a scalar, stack or logically combine those device scalars, reduce with tensor `all`, and perform one host Boolean conversion before `optimizer.step`. Separately do the analogous single conversion for all updated parameter tensors after the step. The empty-gradient list must represent true. Keep the finite-loss check before backward. This changes the successful finite-check host conversions to three per update (loss, gradients, updated parameters), regardless of tensor count. It does not eliminate the per-tensor finite reductions, stacking/workspace, or their device execution cost.

The two gradient/parameter barriers cannot be collapsed into one after the optimizer without changing failure semantics: nonfinite gradients must prevent the update. A single host check per class preserves that requirement and the existing generic failure messages. Do not accidentally use Python `all`/`any` on device tensors inside the aggregate implementation. Preserve loss logging, explicit timing boundaries, source-index order, optimizer, and caps; charge aggregation allocations and compile setup if compilation is used.

Before switching any study, compare outputs, gradients, parameter updates, and deliberate nonfinite failure behavior on fixed fabricated cases. Then benchmark complete updates and separately record finite-check time with warmup and interleaved order. The current counts motivate this measurement; they do not establish that synchronization dominates or predict a particular speedup. The original frozen runner was not modified.
