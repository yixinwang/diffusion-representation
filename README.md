# Diffusion representation

## Research status and iteration summaries

**The requested significant improvement in real image/video generation quality, efficiency, and representation quality has not been established.** Restricted nonlinear, non-Gaussian constructions have proved and measured gains against specified baselines. Equally informed exact-copy flows can tie them. The completed native image comparisons retain negative and mixed results.

This README is the entry point for each research iteration. **For every iteration, update and push a dated Markdown summary here** with the algorithm change, mathematical assumptions, experimental results, failures, source revision, PSC job IDs, evidence links, and next decision. Record pending work as pending. Preserve earlier results rather than replacing failures with later successes. A Pro conversation is a source of proposals until its proofs and artifacts are independently checked.

### Current algorithm and comparison

The current candidate is a normalized, full-dimensional flow without a VAE. A learned invertible multiscale analysis separates a coarse state from residual blocks. A coarse generator and conditional residual transformations retain all **3,072 Gaussian source coordinates** for CIFAR-10.

The innovation-response variant adds a triangular transformation to each cached residual block. It keeps 16 Gaussian anchor coordinates and conditions the remaining coordinates' shifts and bounded scales on those anchors and a history summary. A matched history-only response tests whether access to the new innovations helps. The mathematical argument is a **fixed-chart conditional-KL decomposition**, not a guarantee that training finds a good chart or improves semantic representations over every VAE.

- [Algorithm implementation](qalt/src/qalt/innovation_response.py)
- [Response theory and assumptions](research/transport_iteration_20260908/innovation_response_math.md)
- [Frozen seven-arm comparison protocol](qalt/experiments/innovation_response_pilot_v2/PROTOCOL.md)
- [Numerical qualification evidence](research/transport_iteration_20260908/psc_response_full_qualification)

The study compares innovation/history-only responses with frozen or jointly trained analysis, two strong spline controls, and a scalar control: **seven arms × three seeds = 21 fresh fits**. It uses the same 4,000 fitting images, 1,000 reused development images, and full Gaussian sources. Protected test data remain unopened. All fits and numerical checks must finish before development quality scoring. These short fitting stages are an engineering screen, not an adequately trained state-of-the-art latent-diffusion comparison.

### September 9, 2026 — numerical repair, interactive validation, and fresh study

**Algorithm change.** The reflected spline inverse uses direct distance from the nearer appropriate bin endpoint to avoid an out-of-range rounded root. It preserves the same real-valued rational-quadratic spline family; floating outputs, gradients, and training trajectories can differ. The production validity checks and numerical tolerances were retained.

**Actual failure and validation.** The saved failed GPU batch produced one active inverse coordinate with computed root `1.0000001192092896`, while high-precision inversion gave `0.9999999514812143`. Jobs **45612641, 45614937, and 45614938** reproduced the rejection and checked the candidate. Independent audits verified full 3,072-coordinate CPU/GPU roundtrips, determinant cancellation, exact checkpoint reloads, and all **152 expected joint parameter gradients**. These are bounded checkpoint/input checks, not universal floating-point guarantees. Extreme synthetic roundtrip failures remain documented.

**Preflight failure and correction.** Job **45618049** stopped before canonical data access: 33 tests passed and one test incorrectly required a fabricated legacy-kernel failure on every Torch platform. The corrected test uses independent Decimal80 inversion of each platform's realized knots, with unchanged accuracy tolerances and an additional interior case that detects endpoint snapping. No production algorithm or fitting budget changed.

**Interactive result.** Allocation **45619090** requested one hour, passed all **34 preflight tests** on PSC, and released after **1 minute 57 seconds**. The test process took 42.95 seconds. No training data was accessed.

**Next experiment.** Fresh study **45619353** was submitted at source [`3b8c0f7`](https://github.com/yixinwang/diffusion-representation/commit/3b8c0f7bef604a90ff329fd9608639527b67a972), with all 21 fits from scratch and the unchanged 90-minute allocation envelope. Its last published scheduler check was pending priority at **18:21:35 UTC on September 9**; no quality result from it is recorded here. A [fresh Pro review](https://chatgpt.com/c/6aa19f09-c720-83ea-b1c5-51c6cec1aa1e) is examining stronger latent-FM/diffusion controls and complete cost accounting.

Evidence: [full qualification](research/transport_iteration_20260908/psc_response_full_qualification), [preserved failed preflight](research/transport_iteration_20260908/psc_response_v2_preflight_failure), [interactive verification](research/transport_iteration_20260908/psc_response_v2_interactive_preflight), [fresh submission and scheduler record](research/transport_iteration_20260908/psc_response_v2_attempt2_launch).

### September 9, 2026 — output-quota failure and review correction

**Experiment result.** Job **45619353** failed after **58 minutes 45 seconds** with `OSError(122, 'Disk quota exceeded')` while fitting the final seed's scalar control. Neither the all-fits freeze nor numerical-admission receipt exists; no quality result is admissible. Completed partial arms will not be selected, evaluated, or reused. The [complete original failure payload and independent authentication](research/transport_iteration_20260908/psc_response_v2_quota_failure) are preserved: all 178 payload hashes and 94 frozen source files passed verification. The quota also prevented fallback checkpoint serialization; partial temporary files remain included.

**Fresh run.** Job **45632257** was submitted once at the same source `3b8c0f7`, seeds, numerical gates, and 90-minute budget. It was **RUNNING on v023 at 19:30:52 UTC**, with preflight completion still pending. Both output and scheduler logs now use an authorized project with a passed write/read probe and approximately **985 GiB of actual Lustre quota headroom**. The old project's actual block quota was exceeded despite its earlier rounded allocation display. [Submission, source guards, storage checks, and startup receipt](research/transport_iteration_20260908/psc_response_v2_storage_rerun) are preserved. This is an infrastructure repair, not an algorithm improvement or a successful experiment.

**Review19 failure.** The review finished without the requested checker, tests, or artifact package. Its claimed source summary named seeds 1901–1903 and unrelated method families, contradicting the frozen runner's **78201–78203** and **P_frozen, P_joint, I_frozen, I_joint, RQS_frozen, RQS_joint, S42**. Its source-verification claim is therefore unreliable. The reported nonexistent `hypothesis.md` read also remains a failure. An explicit correction and a smaller inline protocol were requested through the ordinary composer after dismissing the rate notice; the follow-up is pending. No review output was promoted into the implementation.

### September 9, 2026 — resource check and smaller real-image comparison planning

**Earlier snapshot, superseded by the quota failure above.** Study **45619353** passed its 34 GPU preflight tests and was running on one V100. At the recorded 40:08 elapsed snapshot, all seven first-seed fits were complete and the second seed was finishing. Quality results remain unavailable: the protocol requires all 21 fits and numerical admissions before scoring. Source and budgets remain unchanged.

**Initial next-comparison request, before the failed review above.** A [fresh review19](https://chatgpt.com/c/6aa1ae98-91b4-83e9-a30a-4d32ef3e33b3) is developing a bounded comparison of innovation response, history-only response, global splines, full-dimensional flow matching, and a capable codec plus latent generator. It must charge representation training, decoder work, caches, and all rejected fits, with data-free profiling before budget selection. A failed or undertrained baseline is unresolved evidence. The interface displayed “Extra High”; actual Pro model selection could not be verified. This is a submitted review, not an experiment result.

**Available resources and data.** Read-only PSC checks verified four enabled accounts with positive GPU balances, spanning 629–1,087 service units, and eligibility for interactive GPU work. Queue priority is not a start-time guarantee. Public ImageNet and COCO training directories exist; only metadata and README text were inspected. A prospective training-only 64×64 subset can support a smaller new pilot after its split, preprocessing, baseline, and budget freeze. No higher-resolution model or additional dataset experiment has run. Future large outputs need roomier authorized project storage; the current experiment remains untouched.

**Review18 result and limits.** Its proposed broad ImageNet frontier is too large and unprofiled for the next bounded experiment. Independent read-only inspection found a missing full-dimensional prior check for pixel flow matching and acceptance of incomplete method sets. Its 34 reported checker tests do not establish native generation performance or complete protocol coverage. The original 20,408-byte archive was recovered and hash-authenticated locally, but no code was executed or package published here. Its attempted Git tree creation was blocked by automatic safety review: “we couldn't determine the safety status of the request.” That blocked publication has not been retried. The next review is restricted to proposals and local artifacts, with no Git writes.

**Baseline implementation audit.** The existing learned-analysis FM retains a full-dimensional stochastic residual decoder; it is not a conventional compressed latent generator. A capable unconditional RGB codec baseline is still missing. The [source inventory and minimum implementation requirements](research/transport_iteration_20260908/next_real_image_component_audit.md) distinguish reusable components from this missing comparison.

**Decision.** Finish and audit the existing three-seed experiment before using it to motivate a new frozen study. Preserve all negative results. No real-image/video quality, speed, or representation superiority is claimed.

### September 9, 2026 — higher-order conditional dependence (Pro17)

**Algorithm and theory.** A conditional triple copula captures non-Gaussian dependence despite independent univariate and bivariate marginals. A shared, binned response estimator has a checked expected-KL improvement of at least **0.09629 nats per 113-dimensional synthetic observation** over a specified conditional-product baseline. The theorem requires balanced context sampling, valid sharing, and conditional independence of groups within an observation; it does not treat image patches as independent images.

**Reproduced results.** All 20 original tests and the stress runner passed independently. Shared nonlinear data gave KL **0.01242** versus the product oracle's **0.21869**. In the prespecified alternating-sign failure, the shared model gave **0.22066**, worse than the same product oracle; an untied model remained capable. An equally cheap tied triangular copy matched the candidate exactly.

**Decision and limits.** This demonstrates a benefit from a correct dependence/sharing assumption, not a distinct architecture or speed advantage. The binned context map is discontinuous, and additional probes found incomplete NaN validation. It was not promoted into the native implementation. Original failed repository reads and cross-platform reproduction differences are preserved.

Evidence: [original package](research/transport_iteration_20260908/pro17_artifacts), [independent proof and replay](research/transport_iteration_20260908/pro17_local_checks/INDEPENDENT_REVIEW.md).

### Earlier iterations — algorithms, results, and decisions

The rows below summarize earlier work; linked artifacts retain protocols, per-seed results, failures, hashes, and scope qualifications.

| Iteration / experiment | Algorithm or question | Result and decision | Evidence |
|---|---|---|---|
| Initial transport and covariance work, September 8 | Radial inverse-CDF transport and an observation-score screen; covariance repair of residual density | Radial inversion was slow. Covariance repair failed its stronger block/Student and moment comparisons. No joint quality/efficiency claim. | [Historical iteration log](research/transport_iteration_20260908/README.md) |
| Native full-flow/FM pilot, job 45568073 | Shared invertible analysis and coarse generator; global spline residual versus parameter-matched FM | Quality was mixed. The spline was slower than all measured FM settings at batch 64 and used more memory. Later KID: spline 0.15093 versus four-step FM 0.14090. No native win. | [Native evidence](research/transport_iteration_20260908/psc_native_completed) |
| Dense implementation, job 45573581 | Vectorized/compiled spline operations on a frozen complete generator | Measured full-generation speedup was 1.08–1.32×, with unchanged peak memory and 51.91 seconds of first use. This is a checkpoint implementation result, not training-to-quality superiority. | [Independent cost review](research/transport_iteration_20260908/dense_model_review.md) |
| Exact tilted sampler and solver certificates, Pro8–12 | Exact conditional transport versus specified finite-stage Heun samplers on a restricted nonlinear law | Independent interval checks support scoped error bounds. The exact sampler was faster at the matched certified target; four-stage Heun remained faster at looser quality, and an exact-copy decoder ties. No image/video conclusion. | [Cost audit](research/transport_iteration_20260908/tilted_cost_review.md), [interval reproduction](research/transport_iteration_20260908/pro12_local_reproduction/README.md) |
| Cached innovation, job 45576312 | Cache conditional transformations and add a low-rank mixer; compare frozen/joint analysis across three seeds | Overall criteria failed in all three seeds. The strong spline had better complete likelihood and KID in every seed. Retain the failures and test a different response mechanism. | [Complete-study audit and explicit artifact subset](research/transport_iteration_20260908/psc_cached_pilot) |
| Nonlinear pair discovery, job 45579788 | Learn unknown residual pairs and a context-dependent copula in full dimension | Positive-signal cases recovered all 1,440 pairs: KL 0.1855–0.1904 versus approximately 94.67 for the fixed-chart product baseline. Zero-mean discovery failed in all three seeds; all methods fell back to KL approximately 2.193. | [Synthetic evidence](research/transport_iteration_20260908/psc_trapezoid_mechanism), [zero-mean bounds](research/transport_iteration_20260908/zero_mean_discovery) |
| Innovation-response study, job 45582364 | Innovation-conditioned shifts/scales versus a matched history-only response and strong spline controls | Failed during the spline prefix after 820 updates, before quality evaluation. Partial candidate fits were not evaluated or reused. The numerical diagnosis above followed this failure. | [Complete failure evidence](research/transport_iteration_20260908/psc_innovation_response_failure/README.md) |
| Video input feasibility, job 45580328 | Audited decoding and dequantization of one fixed UCF101 training clip | Timestamp policy failed; a separately frozen sequence-index policy passed. Eight observed frames were verified. No video model was trained and no validation/test video payload was opened. | [Input audit](research/transport_iteration_20260908/psc_video_index_input/README.md) |

### Completed native three-seed quality result

Lower KID is better. The following is the completed cached-innovation experiment, **not** the pending innovation-response rerun.

| Seed | Joint candidate KID | Strong spline KID | Passed criteria |
|---|---:|---:|---:|
| 77201 | 0.177906 | 0.164286 | 6/9 |
| 77202 | 0.183249 | 0.170042 | 6/9 |
| 77203 | 0.200088 | 0.172492 | 5/9 |

All 18 saved quality banks were independently recomputed. Reused development data make this exploratory evidence; fresh seeds do not make it confirmatory. No successful-seed selection, protected-test tuning, or claim of superiority over latent diffusion follows.

## Archived research notes

## 12/26/2925

suppose x = f(z) for linear f and low dimensional z, what would the diffusion score, or diffusion denoising function or diffusion mean prediction or noise prediction at different time step look like, what about differences across score trajectories, how would the low rank structure look like? what are would diffusion map and LTSA , and Levina Bickel estimator and how would they work in this? how about for quadratic f (nonlinear)? what if it is a low dimensional manifold but may not have global coordinates, how do I get representation?


What if we represent manifold not using global coordinates, but only constraints?

Also use log space smoothing to emphasize support learning?

The causal changes are small and sparse, so the generation of transformation from one image to another maybe described by low dimensional latents., or can conditional on known transfomration of previous frame.


- Use score-induced riemannian metric for kNN, 
- Levina-Bickel intrinsic dim responds to the score-induced metric
- Not sure why LTSA doesn’t work; no low-dimensional structure at all
- Diffusion map has a low-dimensional structure no matter whether we use score or not

Among score statistics

- E[x0 | xt] has the low-dimensional structure instead.

If (x_0=f(z)) (i.e., the data live exactly on the manifold parameterized by (z)), then the posterior mean is “manifold-projected”:
$$\mathbb E[x_0\mid x_t]=\mathbb E[f(z)\mid x_t]=\int f(z),p(z\mid x_t),dz.$$
In DDPM Gaussian corruption,$$p(x_t\mid z)=\mathcal N!\big(x_t;\ \sqrt{\bar\alpha_t},f(z),\ (1-\bar\alpha_t)I\big),]so by Bayes,[p(z\mid x_t)\ \propto\ p(z),\exp!\left(-\frac{1}{2(1-\bar\alpha_t)}\big|x_t-\sqrt{\bar\alpha_t},f(z)\big|^2\right).$$
What “shows up” in (\mathbb E[x_0\mid x_t])
* It is always in the range/convex hull (in expectation) of (f): it’s an average of points on the manifold ( {f(z)}) under the posterior over (z).
* In the small-noise regime ((1-\bar\alpha_t) small), (p(z\mid x_t)) concentrates near[z^*(x_t)=\arg\min_z |x_t-\sqrt{\bar\alpha_t},f(z)|^2 -2(1-\bar\alpha_t)\log p(z),]and then[\mathbb E[x_0\mid x_t]\approx f(z^*(x_t))](plus small curvature/uncertainty corrections). So it behaves like a projection/denoiser onto the manifold.
* If the mapping is non-injective (multiple (z) give similar (f(z))), then (\mathbb E[x_0\mid x_t]) becomes a mixture average across those modes—potentially landing “between” different manifold points (the classic MMSE vs MAP difference).
If you tell me whether you’re thinking discrete (t) DDPM or continuous VP/VE SDE, I can write the exact same story in that notation (it’s identical conceptually; only the noise parameter changes).



- Shall read the REPA paper, prism hypothesis, and the apple alignment paper
- https://arxiv.org/abs/2410.06940
- https://arxiv.org/abs/2512.19693
- https://www.arxiv.org/abs/2512.07829


https://arxiv.org/pdf/2512.20963
https://arxiv.org/pdf/2510.02305


How to use causality to make video diffusion more efficient? How in general can use causality to improve efficiency in learning?