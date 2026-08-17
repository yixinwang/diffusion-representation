from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import torch
from scipy.stats import t, ttest_1samp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from diqfm import synthetic
from diqfm.rotated import make_orthogonal, rotate_data, train_rotated_chart


def align(train_representation, train_state, test_representation, test_state):
    design = np.c_[np.ones(len(train_representation)), train_representation]
    coefficient = np.linalg.lstsq(design, train_state, rcond=None)[0]
    prediction = np.c_[np.ones(len(test_representation)), test_representation] @ coefficient
    mse = float(np.mean((prediction - test_state) ** 2))
    r2 = float(1 - np.sum((prediction - test_state) ** 2) / np.sum((test_state - test_state.mean(0)) ** 2))
    return mse, r2


def run_seed(seed: int, steps: int):
    params = synthetic.make_params(seed=31415 + seed)
    mixing = make_orthogonal(700 + seed)
    train_base = synthetic.sample(180, params, 1000 + seed, paired=True)
    validation_base = synthetic.sample(80, params, 2000 + seed, paired=True)
    test_base = synthetic.sample(700, params, 3000 + seed, paired=False)
    train = rotate_data(train_base, mixing, paired=True)
    validation = rotate_data(validation_base, mixing, paired=True)
    test = rotate_data(test_base, mixing, paired=False)
    model = train_rotated_chart(train, validation, seed=seed, steps=steps)
    train_x = np.concatenate([train["xa"], train["xb"]])
    train_state = np.concatenate([train_base["za"], train_base["zb"]])
    test_state = test_base["z"]
    with torch.no_grad():
        learned_train = model.encode(torch.tensor(train_x, dtype=torch.float32))[:, :2].numpy()
        learned_test = model.encode(torch.tensor(test["x"], dtype=torch.float32))[:, :2].numpy()
        test_tensor = torch.tensor(test["x"][:512], dtype=torch.float32)
        cycle = float(((model.decode(model.encode(test_tensor)) - test_tensor) ** 2).mean().item())
    diq_mse, diq_r2 = align(learned_train, train_state, learned_test, test_state)
    difference = train["xa"] - train["xb"]
    _, vectors = np.linalg.eigh(difference.T @ difference / len(difference))
    linear_basis = vectors[:, :2]
    linear_mse, linear_r2 = align(train_x @ linear_basis, train_state, test["x"] @ linear_basis, test_state)
    mean = train_x.mean(0)
    centered = train_x - mean
    _, pca_vectors = np.linalg.eigh(centered.T @ centered / len(centered))
    pca_basis = pca_vectors[:, -2:]
    pca_mse, pca_r2 = align(centered @ pca_basis, train_state, (test["x"] - mean) @ pca_basis, test_state)
    with torch.no_grad():
        state_a = model.encode(torch.tensor(train["xa"], dtype=torch.float32))[:, :2].numpy()
        state_b = model.encode(torch.tensor(train["xb"], dtype=torch.float32))[:, :2].numpy()
    return {
        "seed": seed,
        "diq_aligned_state_mse": diq_mse,
        "diq_state_r2": diq_r2,
        "linear_pair_state_mse": linear_mse,
        "linear_pair_state_r2": linear_r2,
        "pca_state_mse": pca_mse,
        "pca_state_r2": pca_r2,
        "diq_pair_variance": float(0.5 * np.mean(np.sum((state_a - state_b) ** 2, axis=1))),
        "linear_pair_variance": float(0.5 * np.mean(np.sum(((train["xa"] @ linear_basis) - (train["xb"] @ linear_basis)) ** 2, axis=1))),
        "cycle_mse": cycle,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "results")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in range(args.seeds):
        print(f"seed {seed}", flush=True)
        rows.append(run_seed(seed, args.steps))
        pd.DataFrame(rows).to_csv(args.output_dir / "raw_metrics.csv", index=False)
    raw = pd.DataFrame(rows)
    raw.drop(columns="seed").agg(["mean", "sem"]).T.to_csv(args.output_dir / "summary.csv")
    comparisons = []
    for baseline in ["linear_pair_state_mse", "pca_state_mse"]:
        gain = raw[baseline] - raw["diq_aligned_state_mse"]
        n = len(gain)
        standard_error = float(gain.std(ddof=1) / np.sqrt(n))
        critical = float(t.ppf(0.975, n - 1))
        test = ttest_1samp(gain, 0.0)
        comparisons.append({
            "comparison": f"diq_vs_{baseline}",
            "gain_mean": float(gain.mean()),
            "ci95_low": float(gain.mean() - critical * standard_error),
            "ci95_high": float(gain.mean() + critical * standard_error),
            "p_two_sided": float(test.pvalue),
            "wins": int((gain > 0).sum()),
            "n": n,
        })
    pd.DataFrame(comparisons).to_csv(args.output_dir / "paired_comparisons.csv", index=False)
    print(raw.to_string(index=False))


if __name__ == "__main__":
    main()
