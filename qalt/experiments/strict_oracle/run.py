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

from qalt.core import (  # noqa: E402
    decoder,
    euler_scale,
    fiber_kl,
    fiber_w2_squared,
    inverse_decoder,
    pooled_variance,
    sample_active,
    separate_variances,
)


@dataclass(frozen=True)
class Config:
    active_dim: int = 4
    fiber_dim: int = 12
    target_scale: float = 2.0
    solver_steps: tuple[int, ...] = (4, 10, 20, 50)
    sample_sizes: tuple[int, ...] = (4, 8, 16, 32, 64, 128, 512)
    confirmation_units: int = 200
    confirmation_seed_start: int = 10_000
    transformer_width: int = 64
    tie_tolerance: float = 1e-12


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def source_manifest() -> dict[str, str]:
    paths = [
        Path(__file__).resolve(),
        ROOT / "src/qalt/core.py",
        ROOT / "tests/test_qalt_core.py",
        ROOT / "theory/experimental_protocol.md",
        ROOT / "theory/QALT_repaired.tex",
    ]
    return {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
    }


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
    source_hashes = source_manifest()
    return {
        "git_commit": git_value("rev-parse", "HEAD"),
        "tracked_worktree_dirty": bool(git_value("status", "--porcelain", "--untracked-files=no")),
        "source_sha256": source_hashes,
        "source_manifest_sha256": hashlib.sha256(
            json.dumps(source_hashes, sort_keys=True).encode()
        ).hexdigest(),
        "python": sys.version,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "command": sys.argv,
    }


def paired_interval(values: list[float], seed: int) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(20_000, len(array)))
    means = array[indices].mean(axis=1)
    return {
        "mean": float(array.mean()),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
    }


def operation_ratio(config: Config, steps: int) -> float:
    active = config.active_dim
    full = config.active_dim + config.fiber_dim
    width = config.transformer_width
    active_eval = active * active * width + active * width * width
    full_eval = full * full * width + full * width * width
    chart_and_fiber = 2 * full + config.fiber_dim * active
    return float((steps * active_eval + chart_and_fiber) / (steps * full_eval))


def solver_rows(config: Config) -> list[dict[str, float | int | str]]:
    target = np.full(config.fiber_dim, config.target_scale)
    rows: list[dict[str, float | int | str]] = []
    for steps in config.solver_steps:
        euler = euler_scale(target, steps)
        for method, scale in {
            "qalt_exact": target,
            "full_latent_euler": euler,
            "full_latent_exponential": target,
            "full_latent_structural_split": target,
        }.items():
            rows.append(
                {
                    "study": "known_scale_solver",
                    "steps": steps,
                    "method": method,
                    "kl": fiber_kl(target, scale),
                    "w2_squared": fiber_w2_squared(target, scale),
                    "qalt_full_operation_ratio": operation_ratio(config, steps),
                }
            )
    return rows


def estimation_rows(config: Config) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    shared_scale = np.full(config.fiber_dim, config.target_scale)
    misspecified_scale = np.sqrt(np.tile([1.0, 4.0], config.fiber_dim // 2))
    for sample_size in config.sample_sizes:
        for unit in range(config.confirmation_units):
            seed = config.confirmation_seed_start + 10_000 * sample_size + unit
            rng = np.random.default_rng(seed)
            for regime, target_scale in {
                "exact_sharing": shared_scale,
                "alternating_variances": misspecified_scale,
            }.items():
                samples = rng.normal(size=(sample_size, config.fiber_dim)) * target_scale
                pooled_scale = np.sqrt(pooled_variance(samples))
                separate_scale = np.sqrt(separate_variances(samples))
                methods = {
                    "qalt_pooled": pooled_scale,
                    "full_latent_coordinatewise": separate_scale,
                    "full_latent_pooled": pooled_scale,
                }
                for method, model_scale in methods.items():
                    rows.append(
                        {
                            "study": "variance_estimation",
                            "regime": regime,
                            "sample_size": sample_size,
                            "unit": unit,
                            "seed": seed,
                            "method": method,
                            "kl": fiber_kl(target_scale, model_scale),
                        }
                    )
    return rows


def nonlinear_sanity(config: Config) -> dict[str, float]:
    rng = np.random.default_rng(config.confirmation_seed_start - 1)
    z = sample_active(10_000, config.active_dim, rng)
    r = config.target_scale * rng.normal(size=(len(z), config.fiber_dim))
    x = decoder(z, r)
    recovered_z, recovered_r = inverse_decoder(x, config.active_dim)
    centered = z - z.mean(axis=0)
    standardized = centered / z.std(axis=0)
    return {
        "max_inverse_error": float(
            max(np.max(np.abs(recovered_z - z)), np.max(np.abs(recovered_r - r)))
        ),
        "mean_active_excess_kurtosis": float(np.mean(standardized**4) - 3.0),
    }


def summarize(
    config: Config,
    solvers: list[dict[str, float | int | str]],
    estimates: list[dict[str, float | int | str]],
) -> dict[str, object]:
    solver_lookup = {(row["steps"], row["method"]): row for row in solvers}
    exact_ties = []
    euler_gaps = []
    for steps in config.solver_steps:
        qalt = float(solver_lookup[(steps, "qalt_exact")]["kl"])
        for method in ("full_latent_exponential", "full_latent_structural_split"):
            exact_ties.append(abs(float(solver_lookup[(steps, method)]["kl"]) - qalt))
        euler_gaps.append(float(solver_lookup[(steps, "full_latent_euler")]["kl"]) - qalt)

    studies: dict[str, object] = {}
    for regime in ("exact_sharing", "alternating_variances"):
        studies[regime] = {}
        for sample_size in config.sample_sizes:
            subset = [
                row
                for row in estimates
                if row["regime"] == regime and row["sample_size"] == sample_size
            ]
            by_method = {
                method: {
                    int(row["unit"]): float(row["kl"])
                    for row in subset
                    if row["method"] == method
                }
                for method in (
                    "qalt_pooled",
                    "full_latent_coordinatewise",
                    "full_latent_pooled",
                )
            }
            units = sorted(by_method["qalt_pooled"])
            coordinate_gain = [
                by_method["full_latent_coordinatewise"][unit]
                - by_method["qalt_pooled"][unit]
                for unit in units
            ]
            pooled_difference = [
                by_method["full_latent_pooled"][unit] - by_method["qalt_pooled"][unit]
                for unit in units
            ]
            studies[regime][str(sample_size)] = {
                "coordinatewise_minus_qalt": paired_interval(
                    coordinate_gain, config.confirmation_seed_start + sample_size
                ),
                "full_pooled_minus_qalt_max_abs": float(np.max(np.abs(pooled_difference))),
            }

    largest = str(config.sample_sizes[-1])
    exact_study = studies["exact_sharing"]
    misspecified_study = studies["alternating_variances"]
    checks = {
        "exact_solver_controls_tie": max(exact_ties) <= config.tie_tolerance,
        "euler_is_strictly_worse": min(euler_gaps) > 0.0,
        "same_information_pooled_control_ties": all(
            exact_study[str(n)]["full_pooled_minus_qalt_max_abs"] <= config.tie_tolerance
            for n in config.sample_sizes
        ),
        "pooling_beats_coordinatewise_under_exact_sharing": all(
            exact_study[str(n)]["coordinatewise_minus_qalt"]["ci95_low"] > 0.0
            for n in config.sample_sizes
        ),
        "misspecified_pooling_loses_at_largest_n": (
            misspecified_study[largest]["coordinatewise_minus_qalt"]["ci95_high"] < 0.0
        ),
    }
    return {
        "solver_euler_kl_gaps": dict(zip(map(str, config.solver_steps), euler_gaps)),
        "estimation": studies,
        "checks": checks,
        "all_registered_checks_pass": all(checks.values()),
        "maximum_claim": (
            "Restricted oracle mechanisms reproduced. Optimized same-information controls tie "
            "QALT in quality; QALT may retain a compute advantage when measured overhead is below "
            "the saved repeated-token cost."
        ),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "results/strict_oracle")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    config = Config(confirmation_units=20) if args.quick else Config()
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    manifest = [
        config.confirmation_seed_start + 10_000 * n + unit
        for n in config.sample_sizes
        for unit in range(config.confirmation_units)
    ]
    manifest_hash = hashlib.sha256(json.dumps(manifest).encode()).hexdigest()
    configuration = {
        **asdict(config),
        "confirmation_seed_manifest_sha256": manifest_hash,
        "provenance": provenance(),
    }
    write_json(args.output / "config.json", configuration)

    solvers = solver_rows(config)
    estimates = estimation_rows(config)
    write_csv(args.output / "solver_metrics.csv", solvers)
    write_csv(args.output / "estimation_metrics.csv", estimates)
    summary = summarize(config, solvers, estimates)
    summary["nonlinear_non_gaussian_sanity"] = nonlinear_sanity(config)
    summary["runtime_seconds"] = time.perf_counter() - started
    write_json(args.output / "summary.json", summary)
    print(json.dumps({"checks": summary["checks"]}, indent=2))
    if not summary["all_registered_checks_pass"]:
        raise SystemExit("one or more registered checks failed")


if __name__ == "__main__":
    main()
