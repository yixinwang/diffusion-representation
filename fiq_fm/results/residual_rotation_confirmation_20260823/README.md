# Failed residual-rotation confirmation

This is the first frozen 30-unit confirmation run from commit `c8ad680`, executed on allocated Slurm job `44283346`. It completed every registered seed `100..129` and both paired rotation arms, then exited nonzero because two preregistered gates failed. The result is preserved as a failure; no seed was removed and no threshold was changed.

Passed results include:

- Haar JBD minus oracle-block NLL: `0.0000433` nat per residual dimension, 95% interval `[-0.00000450, 0.0000911]`; equivalence passed.
- Haar permutation minus JBD NLL: `0.01594`, interval `[0.01355, 0.01833]`; Holm-adjusted superiority passed.
- Haar oracle-diagonal minus oracle-block NLL: `0.02387`, interval `[0.02338, 0.02435]`; Holm-adjusted separation passed.
- Haar provisional-full minus oracle-block NLL: `0.001251`, interval `[0.001037, 0.001466]`; equivalence passed.
- Haar JBD normalized true cross-block leakage: `0.001178`, interval `[0.000756, 0.001601]`; the `<0.05` gate passed.

Failed results are:

- Haar JBD projector error: mean `0.09368`, interval `[0.06553, 0.12183]`; the corrected one-sided `<0.10` test failed (`Holm p=0.325`).
- Seed `119` failed the hard chart rule because its response relative eigengap was `0.0556`, below the frozen `0.10` minimum. All other chart diagnostics passed.

Therefore the learned chart achieves the registered NLL/leakage benefits on this assay but does not confirm reliable recovery of the true latent residual subspaces. The residual-rotation mechanism is not promoted, and image/video scaling is paused pending a separately registered repair or rejection.

Exact source hashes and Slurm provenance are in `config.json`; all 30 unit files and 300 method rows are retained; `SHA256SUMS` covers the generated artifact.
