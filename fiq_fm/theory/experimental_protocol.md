# Historical protocol and repaired-run requirements

The checked pre-audit result tables do not satisfy this protocol because their seed-level files are absent and their summary schemas do not match the current runners. No old table is promoted as verified evidence. Fresh runs must use new output directories and preserve every script-emitted artifact.

## Historical hypotheses to be re-registered before confirmation

1. On an exact quotient with correlated local fibers, gauge-fixed FIQ-FM should improve held-out distribution error over:
   - a diagonal-fiber ablation,
   - a parameter-matched full flow at the finite training budget,
   - KL-VAE latent flow with a diagonal decoder,
   - RAE latent flow with a block decoder.
2. The fitted block-minus-diagonal held-out NLL gain should agree with the analytic conditional Gaussian KL gap.
3. On real digits under misspecification, FIQ-FM should improve at least one preregistered distribution metric and one conditional-fidelity metric over both latent baselines without using test labels for training or model selection.
4. No image-scale asymptotic speed claim is promoted from these vector experiments.

## Data partitions

### Synthetic exact quotient

Each seed uses independent samples:

- 5,000 train,
- 1,500 validation,
- 2,500 test.

The distribution has ambient dimension 18 and active dimension 2. The true chart is hidden by an orthogonal matrix. The residual contains eight conditionally correlated two-dimensional blocks.

### Sklearn digits

For each seed, the 1,797 examples are stratified into:

- 60% train,
- 20% validation,
- 20% test.

Uniform dequantization noise is generated independently for each split. The coordinate-wise mean and standard deviation are estimated on the training split only. The split indices are persisted in each seed JSON, and a unit test checks disjointness.

## Model selection boundaries

The following use train and validation data only:

- flow-moment chart estimation,
- fixed-axis residual partitioning,
- early stopping,
- VAE/RAE encoder and decoder fitting,
- flow and fiber fitting,
- linear-probe regularization,
- evaluator early stopping.

The held-out test split is used only after every model for the seed is frozen.

## Fair comparison constraints

All methods receive the same training examples, augmentations/dequantization, conditioning labels, random-seed list, and final test sample count.

For each comparison:

- FIQ, VAE-LFM, and RAE-LFM have the same active dimension.
- Their latent vector fields have the same depth, width, and ODE solver/NFE.
- The full ambient flow width is chosen not to exceed the FIQ generation parameter budget.
- Inference timing includes latent integration, fiber/decoder sampling, and the chart.
- The VAE and RAE endpoint decoders are stochastic and are charged in parameter and timing totals.
- The diagonal FIQ ablation changes only the fiber covariance family.

## Metrics

### Distribution metrics

- sliced 2-Wasserstein,
- adaptive-bandwidth Gaussian-kernel MMD U-statistic, clipped at zero,
- energy distance,
- covariance error,
- normalized mean error.

### Digits feature and conditional metrics

A classifier/feature network is trained only on the training split and early-stopped on validation. It supplies:

- feature Fréchet distance,
- class-conditional feature Fréchet distance,
- feature-space precision and recall,
- requested-label accuracy.

### Representation diagnostics

- z-only conditional-mean reconstruction MSE,
- validation-selected linear-probe test accuracy,
- held-out fiber NLL,
- active-subspace angle on synthetic data only.

## Statistical reporting

Every requested seed is retained. Synthetic seeds correspond to independent datasets. Digits seeds are overlapping resamples of one finite dataset, so their intervals quantify split sensitivity and are not treated as independent-replication significance. Any confirmatory real-data analysis needs a sampling unit and multiplicity rule fixed before inspecting results.

## Promotion criteria

### Strong synthetic result

Promote only when all hold:

1. FIQ improves sliced-W2 over every baseline on all registered seeds or has a positive paired 95% bootstrap interval.
2. The block-vs-diagonal NLL advantage is positive and numerically agrees with the analytic KL gap.
3. Active-subspace recovery is accurate.
4. The comparison includes endpoint costs.

### Real-data result

Promote only the metrics whose paired intervals exclude zero against both latent baselines. A lower mean without such an interval is reported as descriptive, not established.

## Reproduction

```bash
python -m pip install -e .
pytest -q
python experiments/synthetic_exact/run.py --seeds 0 1 2 3 4 \
  --output results/synthetic_exact_confirmation
python experiments/sklearn_digits/run.py --seeds 0 1 2 3 4 \
  --output results/sklearn_digits_confirmation
```

The scripts persist configuration, all per-seed training/metric records, long-form metrics, paired summaries, and regenerated plots.
