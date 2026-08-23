# Quotient generative modeling research program

## One-sentence spine

The program asks when a reversible quotient--fiber representation can preserve the endpoint law of a full-dimensional generator while reserving repeated neural transport for a lower-dimensional active state, and it tests that mechanism against optimized same-information baselines before moving from controlled nonlinear data to images and videos.

## Evidence levels

- **Proved:** a mathematical statement with explicit assumptions and a checked proof.
- **Simulated:** a registered controlled experiment with known ground truth.
- **Empirically tested:** a frozen implementation evaluated on untouched observed data.
- **Proposal:** an unproved algorithm, bridge, or experiment.
- **Closed:** a claim contradicted by a proof, counterexample, or registered experiment.

## Main-text and appendix routing ledger

| Material | Main note | Appendix or audit |
|---|---|---|
| Problem, endpoint factorization, principal theorem, decisive comparison, limitations | Keep | -- |
| Complete derivations and proof attacks | State result and assumptions | Keep every step |
| Registered primary experiment and central result | Keep | Full seed-level record |
| Secondary metrics, robustness, and ablations | One sentence and pointer | Full tables and plots |
| Image/video implementation details | Short operational description | Configurations, provenance, and compute logs |
| Related-work inventory and novelty audit | Grouped synthesis | Paper-by-paper ledger |

The current program has two related deliverables. FIQ-FM learns a VAE-free ambient quotient and stochastic fiber. QALT applies the same quotient--fiber mechanism inside a fixed latent generator. QALT is a specialization, not a second independent novelty claim.
