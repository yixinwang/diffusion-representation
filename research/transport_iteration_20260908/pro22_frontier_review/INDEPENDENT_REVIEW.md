# Independent review of Review22 claims

Prepared from the supplied Review22 claims and current repository source, without opening any experiment output, checkpoint, or data. [Review conversation](https://chatgpt.com/c/6aa1c0d6-6cd4-83e9-8ac5-006795ceca3a). The ordinary available conversation displayed Extra High; Pro was disabled in the preceding model-menu check, so Pro execution is not claimed. This is not an authentication of an undelivered artifact or an empirical result.

## Architecture and time

`qalt/src/qalt/rgb_codec_flow_matching.py:107–127` uses frequencies exp(-log(10000) j/31), j=0,...,31. The first raw sine component is sin(t), strictly increasing on [0,1] since cos(t)>0 there. Thus the raw time feature map is injective in exact arithmetic. Neither a learned MLP nor float32 evaluation is guaranteed injective, and this does not prove adequate time approximation or learned dynamics.

Each velocity block has two 3x3 convolutions with common dilation cycling 1,2,4 (`:93–104`), plus 3x3 input/output convolutions. Ignoring normalization, the maximal convolution-path receptive-field side is 5+4 sum(d_i): pixel12 blocks gives 117, latent8 gives 73 (sum dilations28 and17). Both exceed their respective32 and8 spatial sides. The cycle includes dilation1, so this is not merely a single sparse dilated kernel. This is potential dependency support, not proof that learned weights use it effectively. Actual GroupNorm computes statistics across each group's channels and all spatial locations, creating global spatial dependence already in one normalization; a claim that the *whole network* has only the stated finite local receptive field would be wrong.

## Prospective clocks and diagnostics

Pixel field checkpoints360/780/1200/1800 and codec600 plus field180/600/1200/1800 yield nominal latent totals780/1200/1800/2400. Those arithmetic sums omit measured codec finalization, normalization/cache, optimizer/model setup, checkpoint IO and other charged work. Shared nominal totals780/1200/1800 are not exact cost matches. Publish actual cumulative whole-pipeline clocks, and reserve evaluation/generation costs consistently. Do not silently relabel field-only time as complete cost.

The raw FM target is Y-Z and input is (1-t)Z+tY (`_loss`, :130–136). A zero-field baseline has positive expected target-square loss. A learned field's loss exceeding that baseline on a fixed independent evaluation bank is a useful relative failure flag; it is not a proof that every learned field below the baseline is adequate. Literal loss greater than zero is not an invalidity criterion: the conditional target variance generally stays positive even for the population-optimal velocity. Specify the zero-*velocity* baseline explicitly, use paired identical noise/time/targets, freeze before inspecting repair results, and disclose reused repair status. A final loss still improving over its penultimate checkpoint by2% in two of three seeds is an engineering indication compatible with budget limitation, not a calibrated certificate of either convergence or nonconvergence. Changing to further training after observing repair invalidates the original frozen scope unless declared a new development iteration. No FM-loss threshold certifies generation quality.

## Exact sign-test arithmetic

For n independent non-tied seed-level differences, under the sign-null with success probability1/2, an all-same-sign result has two-sided exact p=2^(1-n). Hence n=5 gives.0625 and n=6 gives.03125: six is the minimum for unanimous significance at.05 for this *particular two-sided exact sign test*. A prospectively one-sided unanimous test at n=5 has p=.03125. This is not a universal minimum number of seeds or a power guarantee. Multiple endpoints, selected comparisons, ties, dependence, and adaptive choice of direction require separate treatment; images within a shared seed do not become independent training seeds.

## Feasible-witness optimization diagnostic

Let Lhat be one common, complete FIT negative log likelihood, normalized by3072 and evaluated deterministically on exactly the same observations. If a frozen-analysis checkpoint theta_F can be represented unchanged in the allowed joint parameter class C_J (same model parameterization, root weights and all relevant constraints), then

    Lhat(theta_J) - inf_{theta in C_J} Lhat(theta)
      >= Lhat(theta_J) - Lhat(theta_F).

Thus a positive joint-minus-frozen gap, larger than a justified bound on their combined numerical evaluation error, demonstrates an empirical optimization gap with a known feasible better point. If each measured loss has error at most eta, subtract2eta from the measured gap. Repeated floating agreement alone is not a rigorous error bound; report ordinary numerical precision honestly. The common-prefix checkpoint is another feasible witness under the same conditions. None establishes the global optimum or a structural restriction of the joint architecture. Failure to beat a witness is not evidence that the larger class cannot express a better model.

Source `qalt/experiments/innovation_response_pilot_v2/run.py:315–359` confirms the frozen branch trains conditional residual NLL divided by2880, while the joint stage trains complete NLL divided by3072 with analysis unfrozen and root parameters fixed but input gradients live. On a fixed A/root, complete NLL differs from residual NLL by a parameter-independent term and a positive scaling, so the population minimizers in residual parameters agree. The scaling still affects optimization; after A changes the root and analysis terms are no longer constant. Compare all witnesses using complete NLL, rather than comparing their recorded training-loss values directly. Fixed root *weights* do not imply a constant root score after analysis changes.

FIT NLL is an empirical likelihood diagnostic; it is not held-out generalization, KID or energy score. Even genuine complete-NLL improvement need not improve a finite KID estimate. Keep this witness test separate from generation gates and avoid diagnosing representational failure from their disagreement.

## Scope

The corrected claims are defensible as architecture checks, complete-cost accounting requirements and scoped optimization diagnostics. They do not establish a competent trained latent baseline, native superiority, universal model-family dominance, or confirmatory significance from the existing three-seed engineering screen. No source edits, fitting, data access, or jobs were performed for this review.
