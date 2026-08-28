# RMPF-R9 source-frozen sketch-fit rejection

This append-only milestone records the diagnosis-driven RMPF-R9 fitting round. It does not modify prior QALT, FiberLift, FIQ-FM, MCQF, RMPF/RMGL/HSRF, R5-R8, data, split, result, failure, PR, or confirmation artifacts.

R9 changes only the numerical fitter for the exact R8 SSRLT + unchanged R7 coupled-copula flow. The probability family, exact likelihood, observed dimensions, seeds, roles, hardware, controls and frozen systems limits are unchanged.

## Decision

```text
R9_SKETCH_PARAMETER_AND_ROBUST_COST_FAIL_QUALITY_NOT_OPENED_CONFIRMATION_SEALED
```

The 16,384-row source-frozen leverage sketch and one permitted balanced source-block child preserved the full-row regression objectives and validation NLL, but failed the preregistered local-parameter stability gate. The balanced child also failed robust fitting-cost promotion:

- image fit median 1.677744 s, bootstrap interval [1.552918, 1.851390], frozen limit 1.447782 s, 1/7 passing repetitions;
- video fit median 1.933483 s, interval [1.559867, 2.262751], limit 2.214945 s, 5/7 passing repetitions.

Full-row recovery is bit/numerically exact and returns to the original R8 sufficient-statistic fitter. No sampled sketch dimension reached the frozen 0.08 local-parameter error gate before losing the fit-time margin. No R9 CIFAR/UCF sample-quality result and no confirmation result were opened.

Local scientific commits:

- evidence: `69544fc62350976cf8c7a487c9db538aae916ce8`;
- portable handoff: `466cf4604104080ae1e6c51bc69304db3ef6c8c6`;
- parent R8: `8d21f7d9f4ee82ffd038572e191bb336b79db3ea`.

See `THEORY_AND_FALSIFIER.md`, `reproduce.py`, and `MACHINE_VERDICT.json`.