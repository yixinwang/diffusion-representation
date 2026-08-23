# Archived pre-audit results

These tables preserve the result snapshot committed on 16 August 2026. They are **not currently promoted as verified evidence**: the referenced per-seed JSON files are absent, and the original digits reproduction command crashed during aggregation. The aggregation bug and method-dependent metric randomness were repaired on 23 August 2026. Fresh confirmation must write to a new result directory and reproduce every table from seed-level files before these claims are restored.

All values below are from five registered seeds. Lower is better unless stated otherwise. `±` is one standard error.

## Exact quotient: ambient D=18, active d=2

| Method | Sliced W2 | MMD2 | Energy | Covariance error |
|---|---:|---:|---:|---:|
| Gauge-fixed FIQ-FM, block fiber | **0.0400 ± 0.0016** | **0.00044 ± 0.00020** | **0.00245 ± 0.00063** | **0.0433 ± 0.0026** |
| FIQ diagonal-fiber ablation | 0.0642 ± 0.0029 | 0.00158 ± 0.00053 | 0.00769 ± 0.00189 | 0.1119 ± 0.0048 |
| Full FM, parameter matched | 0.0614 ± 0.0016 | 0.00282 ± 0.00055 | 0.01265 ± 0.00174 | 0.0596 ± 0.0061 |
| KL-VAE LFM, diagonal decoder | 0.0708 ± 0.0016 | 0.00150 ± 0.00031 | 0.00889 ± 0.00115 | 0.1203 ± 0.0047 |
| RAE LFM, block decoder | 0.0638 ± 0.0013 | 0.00094 ± 0.00037 | 0.00575 ± 0.00131 | 0.0995 ± 0.0080 |

Paired sliced-W2 gains of FIQ:

| Baseline | Absolute gain | Bootstrap 95% CI | One-sided paired t p-value |
|---|---:|---:|---:|
| Diagonal ablation | 0.0242 | [0.0185, 0.0298] | 0.0008 |
| Full FM | 0.0214 | [0.0153, 0.0286] | 0.0013 |
| KL-VAE LFM | 0.0307 | [0.0222, 0.0382] | 0.0008 |
| RAE LFM | 0.0237 | [0.0193, 0.0286] | 0.0004 |

Mechanism checks:

- mean active-subspace sine error: 0.020;
- analytic diagonal-fiber KL gap: 1.291 nats;
- fitted held-out diagonal-minus-block NLL: 1.292 nats;
- median CPU generation ratio, full/FIQ: 1.30x.

The full-flow result is a finite-budget empirical comparison, not a universal theorem that a quotient estimator always beats a well-trained ambient flow.

## Sklearn digits: ambient D=64, active d=16

| Method | Class feature Fréchet | Sliced W2 | Requested-label accuracy ↑ | Feature Fréchet |
|---|---:|---:|---:|---:|
| Gauge-fixed FIQ-FM, block fiber | **11.002 ± 0.646** | **0.307 ± 0.006** | **0.658 ± 0.015** | **5.239 ± 0.484** |
| FIQ diagonal-fiber ablation | 11.289 ± 0.800 | 0.318 ± 0.005 | 0.628 ± 0.013 | 5.327 ± 0.606 |
| Full FM, parameter matched | 33.820 ± 1.359 | 0.286 ± 0.002 | 0.342 ± 0.015 | 18.922 ± 0.822 |
| KL-VAE LFM, diagonal decoder | 11.990 ± 0.850 | 0.345 ± 0.006 | 0.607 ± 0.014 | 6.179 ± 0.678 |
| RAE LFM, block decoder | 11.131 ± 0.791 | 0.337 ± 0.005 | 0.629 ± 0.015 | 5.342 ± 0.636 |

Established paired improvements over latent baselines:

- sliced-W2 versus VAE: 0.0384, 95% CI [0.0374, 0.0394];
- sliced-W2 versus RAE: 0.0305, 95% CI [0.0273, 0.0335];
- requested-label accuracy versus VAE: +0.0506, 95% CI [0.0399, 0.0589], p=0.0012;
- requested-label accuracy versus RAE: +0.0294, 95% CI [0.0122, 0.0455], p=0.0153.

Representation diagnostics:

| Representation | z-only reconstruction MSE | Linear-probe accuracy ↑ |
|---|---:|---:|
| FIQ | **0.190 ± 0.005** | 0.961 |
| KL-VAE | 0.420 ± 0.003 | 0.954 |
| RAE | 0.373 ± 0.003 | **0.965** |

The class-conditional feature-Fréchet mean is best for FIQ, but its paired interval versus VAE/RAE includes zero. It is therefore not promoted as an established improvement. The full flow has lower raw sliced-W2 but poor conditional-label fidelity at this matched budget; no claim of full-flow quality parity on real images is made.

## Archived files

- `synthetic_exact_verified/config.json`
- `synthetic_exact_verified/metrics_long.csv`
- `synthetic_exact_verified/summary.json`
- `sklearn_digits_verified/config.json`
- `sklearn_digits_verified/metrics_long.csv`
- `sklearn_digits_verified/summary.json`

The archived CSV and JSON summaries alone are insufficient to reconstruct every reported statistic. They remain a historical snapshot rather than confirmation evidence.
