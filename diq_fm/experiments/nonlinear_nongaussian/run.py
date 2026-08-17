from __future__ import annotations

import argparse
import inspect
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import t, ttest_1samp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from diqfm import synthetic as synth


def paired_row(raw: pd.DataFrame, baseline: str, metric: str) -> dict[str, float | int | str]:
    gain = raw[f"{metric}_{baseline}"] - raw[f"{metric}_diq"]
    n = len(gain)
    standard_error = float(gain.std(ddof=1) / np.sqrt(n))
    critical = float(t.ppf(0.975, n - 1))
    test = ttest_1samp(gain, 0.0)
    return {
        "comparison": f"diq_vs_{baseline}",
        "metric": metric,
        "n": n,
        "gain_mean": float(gain.mean()),
        "ci95_low": float(gain.mean() - critical * standard_error),
        "ci95_high": float(gain.mean() + critical * standard_error),
        "p_two_sided": float(test.pvalue),
        "wins": int((gain > 0).sum()),
    }


def summarize(raw: pd.DataFrame, output_dir: Path) -> None:
    raw.to_csv(output_dir / "raw_metrics.csv", index=False)
    raw.drop(columns="seed").agg(["mean", "sem"]).T.to_csv(output_dir / "summary.csv")
    rows = []
    for baseline in ["linear", "full_gmm", "pca_latent"]:
        for metric in ["swd", "fid"]:
            rows.append(paired_row(raw, baseline, metric))
    for baseline in ["linear", "full_gmm"]:
        gain = raw[f"nll_{baseline}"] - raw["nll_diq"]
        n = len(gain)
        standard_error = float(gain.std(ddof=1) / np.sqrt(n))
        critical = float(t.ppf(0.975, n - 1))
        test = ttest_1samp(gain, 0.0)
        rows.append({
            "comparison": f"diq_vs_{baseline}",
            "metric": "nll",
            "n": n,
            "gain_mean": float(gain.mean()),
            "ci95_low": float(gain.mean() - critical * standard_error),
            "ci95_high": float(gain.mean() + critical * standard_error),
            "p_two_sided": float(test.pvalue),
            "wins": int((gain > 0).sum()),
        })
    gain = raw["state_mse_linear"] - raw["state_mse"]
    n = len(gain)
    standard_error = float(gain.std(ddof=1) / np.sqrt(n))
    critical = float(t.ppf(0.975, n - 1))
    test = ttest_1samp(gain, 0.0)
    rows.append({
        "comparison": "diq_vs_linear",
        "metric": "state_mse",
        "n": n,
        "gain_mean": float(gain.mean()),
        "ci95_low": float(gain.mean() - critical * standard_error),
        "ci95_high": float(gain.mean() + critical * standard_error),
        "p_two_sided": float(test.pvalue),
        "wins": int((gain > 0).sum()),
    })
    pd.DataFrame(rows).to_csv(output_dir / "paired_comparisons.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--n-train-per-context", type=int, default=40)
    parser.add_argument("--n-val-per-context", type=int, default=80)
    parser.add_argument("--n-test-per-context", type=int, default=800)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "results")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in range(args.seeds):
        parameters = inspect.signature(synth.run).parameters
        validation_name = "n_validation" if "n_validation" in parameters else "n_val"
        result = synth.run(
            seed=seed,
            n_train=args.n_train_per_context,
            n_test=args.n_test_per_context,
            steps=args.steps,
            **{validation_name: args.n_val_per_context},
        )
        metrics = result[0] if isinstance(result, tuple) else result
        rows.append({"seed": seed, **metrics})
        pd.DataFrame(rows).to_csv(args.output_dir / "partial.csv", index=False)
    raw = pd.DataFrame(rows)
    summarize(raw, args.output_dir)
    print(raw.to_string(index=False))


if __name__ == "__main__":
    main()
