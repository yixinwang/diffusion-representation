# Reversible Multirate Pyramid Flow: audited reset milestone

This folder is the compact reviewable milestone for the RMPF reset. RMPF is a single normalized, all-coordinate exact flow. It uses an orthonormal Haar organization, local triangular detail updates, and periodic parity-capable structured global communication. It has no VAE, encoder/decoder bottleneck, discarded dimensions, quotient/fiber factorization, or route selector.

## Minimal reproduction

```bash
python rmpf_minimal.py > minimal_result.json
```

The script has only NumPy as a dependency. It verifies:

- exact all-coordinate round trip;
- the triangular analytic log-Jacobian against a finite-difference dense Jacobian;
- finite exact likelihood;
- the even/odd parity counterexample: zero pairwise and `(m-1)`-subset gaps but global separation two;
- a smooth strength transition crossing `0.05` nat/dimension at strength 12;
- a monotone rank transition with zero error at the true rank four;
- an exact copied-mechanism tie.

The complete five-seed experiment, dense-autograd Jacobian oracle, controls, measured systems data, and genuine-development ladder are in the release bundle linked from the research handoff.

## Scientific outcome

Known truth passes. Across untouched seeds 9100--9104, the complete implementation obtained no-global-minus-RMPF NLL `0.05296 [0.05173,0.05419]`, MCQF-local-minus-RMPF global-bit error `0.32334 [0.30073,0.34595]`, and strong measured CPU efficiency against the finite validation-selected dense control.

Realistic development does not promote the method. Stable parity/FWHT features solved the exact adversary but did not generalize on opened CIFAR-10 32x32 or source-separated UCF101 clips. Correct Haar parentage and larger NFE did not recover endpoint quality. A final full-rank structured coarse-state head improved opened CIFAR validation energy, but the registered global component `A` contributed only `0.00004082`, below `0.001`, while `B+S+A` used `1.4985x` full stored bytes, `6.0176x` batch latency, `28.0102x` single latency, `1.6228x` peak allocation, and `3.1088x` workspace. Genuine confirmation remains sealed.

No real-data Pareto or novelty claim is made.
