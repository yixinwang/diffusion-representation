# B1-v2 optimization child

Status: frozen after the 200-iteration B1-v2 development result and before any
1,000-iteration score. This is an exploratory parameter-tuning child under the
user-requested empirical loop. It is not attributed to a Blei publication.

## Parent result and first failed execution layer

The parent source commit is `03b71878c62cd7987b1d7070af83800edaeb5993` and
its result directory is
`qalt/results/observed_b1_rgb_block_development_20260830`. The proposed `B4`
cells converged, but the same later-color `A8` and `I8` cells reached the fixed
200-iteration limit in every seed. Several `O4`, `D4`, `A4`, and `I4` cells
also reached that limit. In seed 2100, the last relative likelihood gains in
the nonconverged `A8/I8` cells were `1.31e-8` and `3.07e-8`, above the fixed
`1e-8` tolerance.

The parent estimate
`NLL(B4)-NLL(A8)=-0.0759455` lies outside the required two-sided equivalence
interval. The leading execution explanation is that the independently fitted
scalar comparisons stopped too early. The rival explanation is that the
remaining gains are far too small to explain a difference of this size and
the scalar comparison family is genuinely weaker. Exact `E4` scalarization
tied `B4` to `1.3643e-12`, which rules out a missing joint-density constant.

## One changed layer

Change only the common EM maximum from 200 to 1,000 iterations. Keep relative
tolerance `1e-8`, the initialization count, every model family, scale and
weight bounds, shape, location fits, site sample, five seeds `2100..2104`,
40,000 fitting images, 5,000 repair images, dequantization, scores, image-level
inference, margins, and all diagnostics unchanged. Apply the same maximum to
every fitted arm. This is parameter tuning within the registered algorithms,
not a new algorithmic intervention, so it uses the recorded tuning exception
rather than a new gain theorem.

The repair set has already been inspected. All results from this child are
exploratory and have no confirmation coverage. The official CIFAR test batch
remains unopened.

## Prediction and stop decision

The execution explanation predicts that every fitted cell converges and that
the scalar-arm NLL improves enough to move `B4-A8` and `B4-I8` toward the
`[-0.01,0.01]` interval. It is rejected if any required comparison remains
outside that interval, if any required arm remains nonconverged, or if the NLL
change is negligible relative to the parent difference. Do not increase the
iteration maximum again after seeing this child. Preserve the parent and child
results separately.

Even a pass would not permit full-flow scaling. Radial PIT, angular moments,
cross-band energy, a support-correct reversible local map, and untouched B1
confirmation remain separate requirements.
