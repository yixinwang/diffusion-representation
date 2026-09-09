# Completed native innovation-response study: negative result

GPU job **45632257** completed with exit0 after **1:10:05** on V100v023, frozen source `3b8c0f7bef604a90ff329fd9608639527b67a972`. All21 fits froze before all21 numerical admissions and any quality evaluation. The numerical repair now permits this bounded study to complete; it does not establish universal numerical correctness or generation superiority.

## Result

All three seeds failed the full set of19 registered engineering gates. Lower KID is better:

| Seed | I joint KID | RQS frozen KID | RQS joint KID | Gates passed |
|---|---:|---:|---:|---:|
| 78201 | 0.201820 | 0.169696 | 0.175045 | 11/19 |
| 78202 | 0.176780 | 0.152790 | 0.154525 | 11/19 |
| 78203 | 0.201085 | 0.163628 | 0.169129 | 10/19 |

Both spline controls beat I_joint on KID in every seed. RQS_joint also has lower complete NLL per coordinate in every seed: I_joint -2.627560/-2.644297/-2.640180 versus RQS_joint -2.687221/-2.702495/-2.699536. I_joint improves complete likelihood over I_frozen in all3 seeds, but innovation-versus-history-only response comparisons are mixed. The covariance-half-control and gradient-within10% gates fail in every seed. The registered5% KID improvement against both spline controls fails throughout. No seed or metric is discarded and no gate is relaxed.

This uses4000 FIT images,1000 reused repair images and2000 shared full3072-dimensional Gaussian sources per seed. It is developmental evidence, not held-out significance. Source dimensions are equal within this study, parameter counts are not exactly equal (I539120 versus RQS589712), and all model-specific fitting work is charged by the frozen protocol. This study includes no latent diffusion or FM arm. Save-inclusive generation clocks vary strongly and are not a clean inference-speed frontier.

## Independent verification and provenance

CPU allocation **45637892** independently executed the exact checker prepared before outputs were reviewed. It completed with exit0 onr033 in **one minute** (worker54.44s); one hour was only the requested cap. The worker verified all94 source/Git bytes and21 frozen-fit/numerical-admission identities before the checker. The checker authenticated402 native payloads and recomputed full-bank KID, numerical checks, NLL component sums/means, saved energy means, pixel descriptors, and all gate formulas. Covariance-error and repair-gradient reference summaries are reported inputs to those formulas, not independently reconstructed canonical statistics; the feature extractor/model itself was not rerun. These limits are explicit in audit.json.

Native terminal status SHA256: `d25b0dcee8475f61c4bf7a9ac412783ed5c9e5d30821e73e20d2653592c03722`.

Independent audit SHA256: `2363e64a500302bfc3f1d1da89a9ee40378f6ae9affc55618819a8b53b0ad104`.

This Git folder contains the complete compact CPU audit records, prepared checker and all resulting seed/arm metrics. It is **not** the full2,176,351,767-byte native raw payload. All original banks/checkpoints remain retained at `/ocean/projects/mth260022p/ywang26/diffusion-results/20260909-innovation-response-v2-attempt3`. The slow full local transfer is still incomplete at this publication. The independent audit ran directly against those complete PSC files, not a partial local copy.

## Decision

The present innovation-response candidate has not earned a generation or efficiency claim over the spline controls. [Fresh Review23](https://chatgpt.com/c/6aa1c67d-d050-83ea-b75f-eb2e10ebec5a) is examining one bounded diagnostic/algorithmic decision from these negative results; it is pending, not evidence of a successful replacement. Stronger full/latent-FM baseline work remains separately prospective. Preserve this failure regardless of future outcomes.
