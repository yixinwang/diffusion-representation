# Opt-in dense conditional backend

The additive DenseGlobalConditionalSplineDecoder inherits the original decoder's layers, parameters, masks, state dictionary and validation. It retains the same affine and rational-quadratic transformations. Its explicit dense_eager and compiled choices use a device validity flag aggregated across layers, then reject the entire batch if any arithmetic check fails. The compiled option has no fallback. Existing model defaults and frozen experiments are unchanged.

Six focused tests check nonidentity outputs and log determinants against the original, forward/inverse behavior, a small full Jacobian, source/context/parameter gradients, checkpoint loading and deliberate numerical failure. A CPU tracing test verifies graph capture only; the separate PSC scalar-kernel study supplies actual GPU compilation evidence. The full local qalt suite passes 249 tests in 20.61 seconds.

The kernel speedup is not a whole-generator speedup. Context networks, shared analysis, coarse generation, sigmoid, allocation, first compilation, backward computation and repeated shape compilation still need charging. A separately frozen checkpoint-level benchmark is being prepared before training uses this backend. No native-quality improvement is inferred from changing an implementation backend.
