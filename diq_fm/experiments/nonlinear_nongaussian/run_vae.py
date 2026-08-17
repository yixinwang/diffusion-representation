from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import t, ttest_1samp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from diqfm import vae


def paired_comparisons(vae_raw: pd.DataFrame, diq_raw: pd.DataFrame) -> pd.DataFrame:
    merged = diq_raw.merge(vae_raw, on="seed", suffixes=("_diq_run", "_vae_run"))
    metrics = {
        "swd": merged["swd_stochastic_decode"] - merged["swd_diq"],
        "fid": merged["fid_stochastic_decode"] - merged["fid_diq"],
        "state_r2": merged["state_r2_diq_run"] - merged["state_r2_vae_run"],
    }
    rows = []
    for metric, gain in metrics.items():
        n = len(gain)
        standard_error = float(gain.std(ddof=1) / np.sqrt(n))
        critical = float(t.ppf(0.975, n - 1))
        test = ttest_1samp(gain, 0.0)
        rows.append({
            "comparison": "diq_vs_vae",
            "metric": metric,
            "n": n,
            "gain_mean": float(gain.mean()),
            "ci95_low": float(gain.mean() - critical * standard_error),
            "ci95_high": float(gain.mean() + critical * standard_error),
            "p_two_sided": float(test.pvalue),
            "wins": int((gain > 0).sum()),
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "results")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in range(args.seeds):
        rows.append({"seed": seed, **vae.run(seed)})
        pd.DataFrame(rows).to_csv(args.output_dir / "vae_raw.csv", index=False)
    raw = pd.DataFrame(rows)
    raw.drop(columns="seed").agg(["mean", "sem"]).T.to_csv(args.output_dir / "vae_summary.csv")
    diq_path = args.output_dir / "raw_metrics.csv"
    if diq_path.exists():
        paired_comparisons(raw, pd.read_csv(diq_path)).to_csv(args.output_dir / "vae_paired.csv", index=False)
    print(raw.to_string(index=False))


if __name__ == "__main__":
    main()
