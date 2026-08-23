#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from fiqfm.rotation import (  # noqa: E402
    block_subspace_metrics,
    conditional_gaussian_nll,
    fit_covariance_regression,
    fit_feature_map,
    gaussian_w2_squared,
    haar_orthogonal,
    impose_covariance_family,
    learn_commutant_blocks,
    learn_pair_partition,
    offblock_energy,
    predicted_covariance_contrasts,
    predict_covariance,
    rotate_batch_covariance,
    rotate_covariance,
    rotate_residual,
    sample_rotation_unit,
    signed_permutation,
)


@dataclass(frozen=True)
class Config:
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    n_train: int = 12_000
    n_validation: int = 4_000
    n_test: int = 8_000
    n_random_features: int = 2
    contrast_rank: int = 4
    ridge: float = 1.0
    covariance_floor: float = 0.05
    covariance_ceiling: float = 20.0
    equivalence_margin_nats_per_dim: float = 0.02
    confirmation: bool = False


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def git_value(*arguments: str) -> str | None:
    try:
        return subprocess.run(
            ["git", *arguments],
            cwd=ROOT.parent,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def provenance() -> dict[str, object]:
    source_paths = [
        Path(__file__).resolve(),
        ROOT / "src/fiqfm/rotation.py",
        ROOT / "tests/test_rotation.py",
        ROOT / "theory/residual_rotation_protocol.md",
    ]
    hashes = {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in source_paths
    }
    return {
        "git_commit": git_value("rev-parse", "HEAD"),
        "tracked_worktree_dirty": bool(git_value("status", "--porcelain", "--untracked-files=no")),
        "source_sha256": hashes,
        "source_manifest_sha256": hashlib.sha256(
            json.dumps(hashes, sort_keys=True).encode()
        ).hexdigest(),
        "python": sys.version,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "command": sys.argv,
    }


def paired_interval(values: list[float], seed: int, alpha: float = 0.05) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(20_000, len(array)))
    means = array[indices].mean(axis=1)
    return {
        "mean": float(array.mean()),
        "low": float(np.quantile(means, alpha / 2.0)),
        "high": float(np.quantile(means, 1.0 - alpha / 2.0)),
        "values": array.tolist(),
    }


def transform_covariance(covariance: np.ndarray, axes: np.ndarray) -> np.ndarray:
    return rotate_batch_covariance(covariance, axes)


def evaluate_model(
    name: str,
    family: str,
    axes: np.ndarray,
    residual_observed: np.ndarray,
    target_observed: np.ndarray,
    predicted_observed: np.ndarray,
    truth_rotation: np.ndarray,
    validation_contrasts: np.ndarray,
    config: Config,
) -> dict[str, float | int | str]:
    aligned_residual = residual_observed @ axes
    aligned_target = transform_covariance(target_observed, axes)
    aligned_prediction = transform_covariance(predicted_observed, axes)
    model_covariance = impose_covariance_family(
        aligned_prediction,
        family,
        floor=config.covariance_floor,
        ceiling=config.covariance_ceiling,
    )
    nll = conditional_gaussian_nll(aligned_residual, model_covariance)
    truth_nll = conditional_gaussian_nll(aligned_residual, aligned_target)
    centered = aligned_target - np.eye(4)
    cross = aligned_target[:, :2, 2:]
    leakage = 2.0 * np.sum(cross**2) / max(float(np.sum(centered**2)), 1e-15)
    metrics = block_subspace_metrics(axes, truth_rotation)
    head_outputs = {"diagonal": 4, "block": 6, "full": 10}[family]
    return {
        "method": name,
        "family": family,
        "test_nll": float(nll.mean()),
        "bayes_nll": float(truth_nll.mean()),
        "excess_nll": float((nll - truth_nll).mean()),
        "conditional_w2_squared": float(
            gaussian_w2_squared(aligned_target, model_covariance).mean()
        ),
        "true_crossblock_leakage": float(leakage),
        "heldout_contrast_loss": offblock_energy(validation_contrasts, axes),
        "projector_error": metrics["projector_error"],
        "maximum_principal_sine": metrics["maximum_principal_sine"],
        "covariance_head_outputs": head_outputs,
    }


def run_arm(
    arm: str,
    rotation: np.ndarray,
    train,
    validation,
    test,
    feature_map,
    config: Config,
) -> dict[str, object]:
    train_residual = rotate_residual(train.residual, rotation)
    test_residual = rotate_residual(test.residual, rotation)
    train_features = feature_map.transform(train.active)
    validation_features = feature_map.transform(validation.active)
    test_features = feature_map.transform(test.active)

    fit_started = time.perf_counter()
    coefficients = fit_covariance_regression(
        train_features, train_residual, ridge=config.ridge
    )
    fit_seconds = time.perf_counter() - fit_started
    chart_contrasts, singular_values = predicted_covariance_contrasts(
        train_features, coefficients, rank=config.contrast_rank
    )
    audit_contrasts, audit_singular_values = predicted_covariance_contrasts(
        validation_features, coefficients, rank=config.contrast_rank
    )

    permutation_started = time.perf_counter()
    permutation_axes = learn_pair_partition(chart_contrasts)
    permutation_seconds = time.perf_counter() - permutation_started
    jbd_started = time.perf_counter()
    jbd = learn_commutant_blocks(chart_contrasts)
    jbd_seconds = time.perf_counter() - jbd_started

    predicted_test = predict_covariance(test_features, coefficients)
    target_observed = rotate_covariance(test.covariance, rotation)
    methods = [
        ("oracle_block", "block", rotation),
        ("permutation_block", "block", permutation_axes),
        ("jbd_block", "block", jbd.axes),
        ("oracle_diagonal", "diagonal", rotation),
        ("provisional_full", "full", np.eye(4)),
    ]
    rows = [
        evaluate_model(
            name,
            family,
            axes,
            test_residual,
            target_observed,
            predicted_test,
            rotation,
            audit_contrasts,
            config,
        )
        for name, family, axes in methods
    ]
    row_lookup = {row["method"]: row for row in rows}
    denominator = (
        float(row_lookup["permutation_block"]["test_nll"])
        - float(row_lookup["oracle_block"]["test_nll"])
    )
    closed = None
    if denominator > 1e-12:
        closed = (
            float(row_lookup["permutation_block"]["test_nll"])
            - float(row_lookup["jbd_block"]["test_nll"])
        ) / denominator
    for row in rows:
        row["arm"] = arm
    return {
        "arm": arm,
        "rows": rows,
        "diagnostics": {
            "contrast_singular_values": singular_values.tolist(),
            "audit_contrast_singular_values": audit_singular_values.tolist(),
            "commutant_eigenvalues": jbd.commutant_eigenvalues.tolist(),
            "commutant_eigengap": float(
                jbd.commutant_eigenvalues[1] - jbd.commutant_eigenvalues[0]
            ),
            "separator_eigenvalues": jbd.separator_eigenvalues.tolist(),
            "jbd_oracle_gap_fraction_closed": closed,
            "unconditional_covariance_eigenvalues": np.linalg.eigvalsh(
                np.cov(train_residual, rowvar=False)
            ).tolist(),
            "fit_seconds": fit_seconds,
            "permutation_seconds": permutation_seconds,
            "jbd_seconds": jbd_seconds,
        },
    }


def run_seed(seed: int, config: Config, output: Path) -> dict[str, object]:
    train = sample_rotation_unit(config.n_train, 100_000 * seed + 1)
    validation = sample_rotation_unit(config.n_validation, 100_000 * seed + 2)
    test = sample_rotation_unit(config.n_test, 100_000 * seed + 3)
    feature_map = fit_feature_map(
        train.active, n_random=config.n_random_features, seed=100_000 * seed + 4
    )
    rotations = {
        "signed_permutation": signed_permutation(4, 100_000 * seed + 5),
        "haar": haar_orthogonal(4, 100_000 * seed + 6),
    }
    arms = {}
    for name, rotation in rotations.items():
        print(f"[rotation] seed={seed} arm={name}", flush=True)
        arms[name] = run_arm(
            name, rotation, train, validation, test, feature_map, config
        )
    result = {
        "seed": seed,
        "data_seeds": {
            "train": 100_000 * seed + 1,
            "validation": 100_000 * seed + 2,
            "test": 100_000 * seed + 3,
            "features": 100_000 * seed + 4,
            "signed_permutation": 100_000 * seed + 5,
            "haar": 100_000 * seed + 6,
        },
        "active_excess_kurtosis": float(
            np.mean(
                ((test.active - test.active.mean(axis=0)) / test.active.std(axis=0)) ** 4
            )
            - 3.0
        ),
        "arms": arms,
    }
    write_json(output / f"seed_{seed}.json", result)
    return result


def summarize(results: list[dict[str, object]], config: Config) -> dict[str, object]:
    rows = [
        {"seed": result["seed"], **row}
        for result in results
        for arm in result["arms"].values()
        for row in arm["rows"]
    ]
    summary: dict[str, object] = {
        "mode": "confirmation" if config.confirmation else "development",
        "n_units": len(results),
        "methods": {},
        "paired": {},
    }
    metrics = [
        "test_nll",
        "excess_nll",
        "conditional_w2_squared",
        "true_crossblock_leakage",
        "heldout_contrast_loss",
        "projector_error",
        "maximum_principal_sine",
    ]
    for arm in ("signed_permutation", "haar"):
        summary["methods"][arm] = {}
        for method in (
            "oracle_block",
            "permutation_block",
            "jbd_block",
            "oracle_diagonal",
            "provisional_full",
        ):
            selected = [
                row for row in rows if row["arm"] == arm and row["method"] == method
            ]
            summary["methods"][arm][method] = {
                metric: paired_interval(
                    [float(row[metric]) for row in selected],
                    seed=7000 + len(method) + len(metric) + len(arm),
                )
                for metric in metrics
            }

    def differences(arm: str, left: str, right: str, metric: str = "test_nll") -> list[float]:
        by_seed = {
            (int(row["seed"]), str(row["method"])): float(row[metric])
            for row in rows
            if row["arm"] == arm
        }
        return [
            by_seed[(int(result["seed"]), left)]
            - by_seed[(int(result["seed"]), right)]
            for result in results
        ]

    comparisons = {
        "permutation_arm_permutation_minus_oracle": (
            "signed_permutation",
            "permutation_block",
            "oracle_block",
        ),
        "permutation_arm_jbd_minus_oracle": (
            "signed_permutation",
            "jbd_block",
            "oracle_block",
        ),
        "haar_permutation_minus_jbd": ("haar", "permutation_block", "jbd_block"),
        "haar_jbd_minus_oracle": ("haar", "jbd_block", "oracle_block"),
        "haar_full_minus_oracle": ("haar", "provisional_full", "oracle_block"),
        "haar_diagonal_minus_oracle": ("haar", "oracle_diagonal", "oracle_block"),
    }
    for name, (arm, left, right) in comparisons.items():
        summary["paired"][name] = paired_interval(
            differences(arm, left, right), seed=9000 + len(name)
        )
    summary["maximum_claim"] = (
        "Development diagnostics only; no residual-rotation claim is promoted."
        if not config.confirmation
        else "Confirmation gates must be evaluated exactly as preregistered."
    )
    return {"rows": rows, "summary": summary}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "results/residual_rotation_development")
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--confirmation", action="store_true")
    args = parser.parse_args()
    if args.confirmation and (args.smoke or args.seeds is not None):
        raise SystemExit("confirmation uses only the frozen seeds and sizes")
    if args.confirmation:
        config = Config(seeds=tuple(range(100, 130)), confirmation=True)
    elif args.smoke:
        config = Config(
            seeds=tuple(args.seeds or (0,)),
            n_train=2_000,
            n_validation=1_000,
            n_test=1_000,
        )
    else:
        config = Config(seeds=tuple(args.seeds or Config().seeds))
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    write_json(args.output / "config.json", {**asdict(config), "provenance": provenance()})
    results = [run_seed(seed, config, args.output) for seed in config.seeds]
    aggregate = summarize(results, config)
    write_csv(args.output / "metrics_long.csv", aggregate["rows"])
    aggregate["summary"]["runtime_seconds"] = time.perf_counter() - started
    write_json(args.output / "summary.json", aggregate["summary"])
    print(
        json.dumps(
            {
                name: values["mean"]
                for name, values in aggregate["summary"]["paired"].items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
