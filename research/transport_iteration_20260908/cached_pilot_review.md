The native cached-innovation stage1 run completed successfully as an execution, but failed its declared engineering success criterion for every seed. Job 45576312 completed on v024 in 52:56 with exit 0:0; Slurm sampled batch MaxRSS was 1,949,576 KiB. All fits were frozen before evaluation. This is reused-repair development evidence, not confirmatory testing.

Full independent audit verified all 85 frozen source files against commit `1f6be303c8b93ba45ac63c43e7c5971fa38a0082`, all 317 payload hashes (1,820,749,149 bytes), all 18 complete generated banks and all 12 saved numerical banks. Maximum source inversion error is 0.0001245104 (gate 0.001); maximum determinant cancellation error is 0.001953125 (gate 0.01). Numerical outputs are finite and exact-copy controls are bitwise equal. Every formerly missing file has now been retrieved and verified; the earlier compact review remains as historical evidence in compact-review.md.

The 27 engineering booleans were independently recomputed from hash-verified summary values and match the per-seed and completion reports. Seeds 77201/77202 pass six of nine gates; seed 77203 passes five of nine. All three fail covariance-error halving, J beating RQS KID, and J gradient energy within 10% of repair. Seed 77203 additionally fails J beating S KID. No averaging rescues these failures.

M improves residual NLL over S and J improves complete NLL over M for all three seeds. All three energy comparisons satisfy the registered +0.0005 tolerance, but this does not imply J has the lowest energy. RQS has better complete NLL and KID than J in all three seeds. J residual NLL must not be compared directly with M as evidence of complete-quality improvement because J changes analysis coordinates.

| Seed | Arm | Complete NLL/coordinate | Residual NLL/coordinate | KID | Covariance error | Paired pixel energy |
|---|---|---:|---:|---:|---:|---:|
| 77201 | S | -2.5532289 | 1.3030687 | 0.2016193 | 0.0014812 | 0.1703187 |
| 77201 | M | -2.5580767 | 1.2978977 | 0.2000149 | 0.0015200 | 0.1702073 |
| 77201 | J | -2.6275817 | 1.0480189 | 0.1779058 | 0.0013732 | 0.1700305 |
| 77201 | RQS | -2.6433819 | 1.2069054 | 0.1642864 | 0.0013534 | 0.1701409 |
| 77201 | A_only | -2.4669340 | 1.3711300 | 0.2278631 | 0.0066966 | 0.1720000 |
| 77201 | root_only | -2.4894214 | 1.3711300 | 0.2051058 | 0.0014770 | 0.1703214 |
| 77202 | S | -2.5643206 | 1.3525064 | 0.1848752 | 0.0028259 | 0.1692973 |
| 77202 | M | -2.5703266 | 1.3460999 | 0.1971256 | 0.0029145 | 0.1694500 |
| 77202 | J | -2.6233042 | 1.0647787 | 0.1832486 | 0.0027342 | 0.1695029 |
| 77202 | RQS | -2.6471957 | 1.2641063 | 0.1700417 | 0.0028879 | 0.1693483 |
| 77202 | A_only | -2.5011143 | 1.3969195 | 0.2235418 | 0.0075270 | 0.1714845 |
| 77202 | root_only | -2.5226833 | 1.3969195 | 0.1934007 | 0.0028128 | 0.1692866 |
| 77203 | S | -2.5710727 | 1.3679713 | 0.1944719 | 0.0018177 | 0.1713724 |
| 77203 | M | -2.5785236 | 1.3600237 | 0.1955489 | 0.0017991 | 0.1714570 |
| 77203 | J | -2.6289418 | 1.0627716 | 0.2000879 | 0.0021577 | 0.1714372 |
| 77203 | RQS | -2.6519977 | 1.2816513 | 0.1724920 | 0.0018533 | 0.1712370 |
| 77203 | A_only | -2.5150778 | 1.4071290 | 0.2293614 | 0.0074973 | 0.1712907 |
| 77203 | root_only | -2.5343624 | 1.4071290 | 0.2036799 | 0.0017726 | 0.1714541 |

All secondary PRDC, gradient means, endpoint counts, parameter counts, and fitting charges remain in the downloaded per-arm/evaluation JSON and compact-machinecheck.json. M/J have 528,624 parameters, S 531,088, RQS 589,712; these include the disclosed 1,152 coarse zero-context input weights. RQS is a larger comparator. This audit does not infer a speed or memory ranking. A-only/root-only are diagnostics with shorter fitting histories, not equally trained full competitors. Float32 sigmoid endpoints remain recorded without clipping or replacement.

The full audit independently recomputed all 18 unbiased full-bank polynomial KID and PRDC records from saved features, with maximum discrepancy across scalar quality/NLL/energy metrics of 3.11e-15. Features were not re-extracted. It recomputed all descriptors from the actual generated pixels (byte-exact), covariance errors and gradient means against the strictly loaded canonical repair arrays, and every paired pixel energy value. It verified each likelihood total equals its four saved components, independently reconstructed the outer logit-chart term, and reproduced complete and residual means. Model likelihood terms themselves were not re-evaluated, so this verifies their preserved arithmetic rather than a model-forward likelihood replay.

The canonical repair array was obtained with the frozen strict loader after all 85 source-file guards passed; its ledger matches the original pilot, and both ID and value hashes match all three per-seed pair records. Only the 1,000 selected repair arrays were exported; no fit/discovery arrays were exported and no official test file was read. Repair value SHA256: `587a5fd9b63fa694599d615993b1c69e1da0c29ed6ad555239da7ba0b45d64a9`; ID SHA256: `da0870e250bc25661e566a4cdf460f64e2e58765d217f7f4490b4f88d3e8d082`.

All 18 saved float64 pixel banks are exact casts of float32 values and lie in [0,1]. Comparing them against a float64 CPU logistic transform of the saved float32 logits gives maximum error 8.89e-8, within the explicit 1.2e-7 numerical audit tolerance. All recorded endpoint counts agree with actual arrays: 16 values at one and none at zero across the 18 banks. No clipping or replacement was introduced. This is a cross-precision consistency check, not a claim that CPU float64 logistic equals CUDA float32 sigmoid byte-for-byte.

All 21 training-stage input-order hashes were regenerated from the declared RNG seeds or saved M/J fork RNG, using only recorded fit IDs and update counts. Their final RNG states match saved checkpoints exactly; no training was replayed. Every stage completed with nonzero finite losses, and its recorded overrun is within one measured maximum step plus 0.01 seconds. Frozen shared analysis/root blocks match across M/S/RQS checkpoints (160 tensors each), while J's fixed coarse root matches (56 tensors). This does not independently replay optimizer moments or prove the full training trajectory.

A local Gaussian replay check initially failed byte equality: this machine's Torch 2.14/ARM differs from the PSC Torch 2.10/x86 environment, and maximum differences are 2.27e-6, 2.34e-6 and 2.62e-6. The authoritative saved source banks and their completion/pair hashes agree exactly. The initial assertion error and discrepancy measurements remain preserved, and no saved source was replaced. The same-PSC-environment source-only replay returned a complete success JSON: all three generation banks and all twelve numerical source banks are byte-exact under Torch 2.10.0+cu128. However the enclosing SSH process did not terminate normally within 300 seconds and was stopped locally; its received success JSON and timeout are both preserved in cached_pilot_native_noise_result.json and cached_pilot_native_noise_transport.json. This distinguishes a completed source assertion result from an unobserved normal transport exit. No retry was performed. The local different-environment check remains explicitly non-byte-exact.

The portable full checker is audit-full.py with independent-feature-metrics.py, accepting --results, --repair, --repo and a new --output path. Full result banks remain in work/psc-cached-pilot/completed-small (the historical directory name no longer describes its size); repair-audit contains only the identity-checked reference arrays. Machine output is full-machinecheck.json; transfer inventory and all failures remain alongside it. Keep these raw arrays outside Git; their exact hashes and remote original location are recorded in the original status. No fit, model generation, evaluator feature extraction, scheduler job or repository commit occurred during the full audit.
