# RMPF-R12: stage-aligned cross-site angular boundary

This append-only milestone records the diagnosis-driven RMPF-R12 round. It preserves all earlier QALT, FiberLift, FIQ-FM, MCQF, RMPF/RMGL/HSRF, R1-R11, C3/C4/C4F/C4G evidence and does not modify prior data, splits, results, failures, hashes, tags, or PRs.

## Frozen change

R12 changes only the rejected angular/cross-site dependence layer of the exact normalized all-coordinate no-VAE C4G/FSMLF flow. A rank-eight stage-aligned cross-site spherical Möbius transport was inserted before the unchanged R7 endpoint. The map keeps every coordinate, has an explicit inverse, and contributes the exact spherical conformal log-Jacobian. Identity is recovered at zero parameter; an unrestricted copied-mechanism control ties exactly.

## Main result

- Matched known truth passes: identity-minus-R12 NLL/dimension `0.0239029 [0.0232096,0.0245962]`; proper-energy gain `0.152080 [0.147676,0.156483]`.
- Systems pass on the frozen CPU: image batch `0.030227 s`, video batch `0.022884 s`; exact round trips below `2e-12`; logdet cancellation zero.
- Opened seed-9400 CIFAR/UCF smoke fails: image proper-energy gain is only `0.0001056`; video gain is effectively zero. Replication seeds 9401-9404 and untouched confirmation remain unopened.

## Diagnosis chain

Successive exact angular families—ACG covariance, axial, projective scatter, extensive trace-zero, and fitted stage patterns—improve likelihood and registered dependence diagnostics but do not deliver the required whole-sample proper quality. A QMC audit shows that sufficiently strong aligned teachers can yield a proper-energy gain, but real video has insufficient source-separated identification and hierarchical pooling creates a likelihood/target mismatch: video NLL improves by `0.225691` while the registered pattern error worsens by `-1.30655`.

## Decision

`KNOWN_TRUTH_AND_SYSTEMS_PASS__REAL_IMAGE_VIDEO_SMOKE_FAIL__LEARNED_PATTERN_IMAGE_HEADROOM__HIERARCHICAL_VIDEO_NLL_TARGET_MISMATCH__REPLICATION_CONFIRMATION_SEALED`

The finite-family boundary is target mismatch followed by approximation/representation, not normalization, invertibility, or systems. The next scientifically distinct layer is a joint cross-level/site semantic or multimodal stage-feedback transport inside the lifting conditioners.