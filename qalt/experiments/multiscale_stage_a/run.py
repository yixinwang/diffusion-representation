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
from scipy import stats


ROOT = Path(__file__).resolve().parents[2]
REPOSITORY = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))

from qalt.multiscale import (  # noqa: E402
    MixtureFiber,
    band_slices,
    diagonal_scale,
    euler_scale,
    fit_mixture_fiber,
    gaussian_log_prob,
    haar_forward,
    haar_inverse,
    token_benchmark,
)


@dataclass(frozen=True)
class Config:
    seeds: tuple[int, ...] = (700, 701, 702, 703, 704)
    image_shape: tuple[int, ...] = (32, 32)
    video_shape: tuple[int, ...] = (16, 32, 32)
    image_sizes: tuple[int, int, int] = (64, 16, 32)
    video_sizes: tuple[int, int, int] = (16, 4, 8)
    solver_steps: int = 20
    equivalence_margin: float = 0.02
    excess_nll_maximum: float = 0.01
    timing_repeats: int = 9
    confirmation: bool = False


@dataclass(frozen=True)
class GeneratedData:
    observations: np.ndarray
    truth_fibers: tuple[MixtureFiber, ...]


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def git_value(*arguments: str) -> str | None:
    try:
        return subprocess.run(
            ["git", *arguments], cwd=REPOSITORY, check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def source_manifest() -> dict[str, str]:
    paths = [
        Path(__file__).resolve(),
        ROOT / "src/qalt/multiscale.py",
        ROOT / "tests/test_multiscale.py",
        REPOSITORY / "research/MULTISCALE_QALT_PROTOCOL.md",
    ]
    return {
        str(path.relative_to(REPOSITORY)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
    }


def provenance() -> dict[str, object]:
    hashes = source_manifest()
    return {
        "git_commit": git_value("rev-parse", "HEAD"),
        "tracked_worktree_dirty": bool(git_value("status", "--porcelain", "--untracked-files=no")),
        "source_sha256": hashes,
        "source_manifest_sha256": hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(),
        "python": sys.version,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "command": sys.argv,
    }


def truth_fiber(band: int) -> MixtureFiber:
    return MixtureFiber(
        logistic=np.array([-0.35 + 0.08 * band, 1.15, -0.65]),
        scales=np.array([0.32 + 0.015 * band, 1.25 + 0.04 * band]),
    )


def generate_data(n: int, shape: tuple[int, ...], seed: int) -> GeneratedData:
    """Generate an equal-dimensional nonlinear, non-Gaussian multiscale law."""
    rng = np.random.default_rng(seed)
    axes = tuple(range(1, len(shape) + 1))
    full_shape = (n, *shape)
    bands = band_slices(full_shape, axes)
    coefficients = np.empty(full_shape, dtype=float)
    coarse_shape = coefficients[bands[0]].shape
    component = rng.integers(0, 3, size=(n,) + (1,) * len(shape))
    location = np.array([-1.5, 0.2, 1.4])[component]
    base = rng.standard_t(5, size=coarse_shape)
    spatial = np.linspace(-1.0, 1.0, int(np.prod(coarse_shape[1:]))).reshape(coarse_shape[1:])
    coarse = location + 0.55 * base + 0.25 * np.sin(2.0 * spatial + location)
    coefficients[bands[0]] = coarse
    fibers = []
    for number, index in enumerate(bands[1:], start=1):
        fiber = truth_fiber(number)
        coefficients[index] = fiber.sample(coarse, rng)
        fibers.append(fiber)
    observations = haar_inverse(coefficients, axes)
    return GeneratedData(observations=observations, truth_fibers=tuple(fibers))


def coefficients_from_observations(observations: np.ndarray) -> tuple[np.ndarray, list[tuple[slice, ...]]]:
    axes = tuple(range(1, observations.ndim))
    coefficients = haar_forward(observations, axes)
    return coefficients, band_slices(coefficients.shape, axes)


def fit_fibers(observations: np.ndarray) -> tuple[MixtureFiber, ...]:
    coefficients, bands = coefficients_from_observations(observations)
    parent = coefficients[bands[0]]
    return tuple(fit_mixture_fiber(coefficients[index], parent) for index in bands[1:])


def model_nll(
    observations: np.ndarray,
    fibers: tuple[MixtureFiber, ...],
    method: str,
    diagonal_scales: tuple[float, ...] | None = None,
    steps: int = 20,
) -> float:
    coefficients, bands = coefficients_from_observations(observations)
    parent = coefficients[bands[0]]
    total = 0.0
    for number, (index, fiber) in enumerate(zip(bands[1:], fibers)):
        detail = coefficients[index]
        if method in {"qalt", "exact_split", "full_token_exact", "oracle"}:
            log_prob = fiber.log_prob(detail, parent)
        elif method == "diagonal_vae":
            if diagonal_scales is None:
                raise ValueError("diagonal scales are required")
            log_prob = gaussian_log_prob(detail, diagonal_scales[number]).reshape(-1)
        elif method == "coarse_only":
            log_prob = gaussian_log_prob(detail, 1.0).reshape(-1)
        elif method == "full_token_euler":
            approximate = MixtureFiber(fiber.logistic, euler_scale(fiber.scales, steps))
            log_prob = approximate.log_prob(detail, parent)
        else:
            raise ValueError(f"unknown method: {method}")
        total -= float(np.sum(log_prob))
    return total / observations.size


def run_modality(config: Config, seed: int, modality: str) -> dict[str, object]:
    if modality == "image":
        shape, sizes = config.image_shape, config.image_sizes
    elif modality == "video":
        shape, sizes = config.video_shape, config.video_sizes
    else:
        raise ValueError("unknown modality")
    train = generate_data(sizes[0], shape, 100_000 * seed + 1)
    validation = generate_data(sizes[1], shape, 100_000 * seed + 2)
    test = generate_data(sizes[2], shape, 100_000 * seed + 3)
    fitted = fit_fibers(train.observations)
    train_coefficients, bands = coefficients_from_observations(train.observations)
    diagonal = tuple(diagonal_scale(train_coefficients[index]) for index in bands[1:])
    methods = {
        "oracle": test.truth_fibers,
        "qalt": fitted,
        "exact_split": fitted,
        "full_token_exact": fitted,
        "diagonal_vae": fitted,
        "coarse_only": fitted,
        "full_token_euler": fitted,
    }
    nll = {
        method: model_nll(
            test.observations,
            fibers,
            method,
            diagonal_scales=diagonal,
            steps=config.solver_steps,
        )
        for method, fibers in methods.items()
    }
    validation_nll = model_nll(validation.observations, fitted, "qalt")
    total_tokens = int(np.prod(shape))
    active_tokens = total_tokens // (2 ** len(shape))
    benchmark = token_benchmark(shape, config.solver_steps, config.timing_repeats)
    expected_ratio = (config.solver_steps * active_tokens + total_tokens - active_tokens) / (
        config.solver_steps * total_tokens
    )
    return {
        "modality": modality,
        "seed": seed,
        "sizes": {"train": sizes[0], "validation": sizes[1], "test": sizes[2]},
        "nll_per_dimension": nll,
        "validation_qalt_nll_per_dimension": validation_nll,
        "benchmark": benchmark,
        "expected_update_ratio": expected_ratio,
        "empirical_update_ratio": benchmark["qalt_token_updates"] / benchmark["full_token_updates"],
        "max_haar_roundtrip_error": float(
            np.max(
                np.abs(
                    haar_inverse(
                        haar_forward(test.observations[:1], tuple(range(1, len(shape) + 1))),
                        tuple(range(1, len(shape) + 1)),
                    )
                    - test.observations[:1]
                )
            )
        ),
    }


def student_interval(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    critical = float(stats.t.ppf(0.975, len(array) - 1))
    half = critical * array.std(ddof=1) / np.sqrt(len(array))
    return {"mean": float(array.mean()), "low": float(array.mean() - half), "high": float(array.mean() + half)}


def one_sided_test(values: list[float], null: float, alternative: str) -> dict[str, float | str]:
    array = np.asarray(values, dtype=float)
    standard_error = float(array.std(ddof=1) / np.sqrt(len(array)))
    mean = float(array.mean())
    if standard_error <= np.finfo(float).eps:
        passes = mean > null if alternative == "greater" else mean < null
        raw_p = 0.0 if passes else 1.0
    else:
        statistic = (mean - null) / standard_error
        raw_p = float(stats.t.sf(statistic, len(array) - 1) if alternative == "greater" else stats.t.cdf(statistic, len(array) - 1))
    return {"mean": mean, "null": null, "alternative": alternative, "standard_error": standard_error, "raw_p": raw_p}


def holm_adjust(raw: dict[str, float]) -> dict[str, float]:
    ordered = sorted(raw.items(), key=lambda item: (item[1], item[0]))
    adjusted = {}
    running = 0.0
    for index, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, (len(ordered) - index) * value))
        adjusted[name] = running
    return adjusted


def summarize(config: Config, units: list[dict[str, object]]) -> dict[str, object]:
    modalities: dict[str, object] = {}
    all_checks = []
    confirmation_components: dict[str, dict[str, float | str]] = {}
    for modality in ("image", "video"):
        selected = [unit for unit in units if unit["modality"] == modality]
        def differences(left: str, right: str) -> list[float]:
            return [
                float(unit["nll_per_dimension"][left]) - float(unit["nll_per_dimension"][right])
                for unit in selected
            ]
        qalt_excess = student_interval(differences("qalt", "oracle"))
        exact_difference = student_interval(differences("exact_split", "qalt"))
        diagonal_gap = student_interval(differences("diagonal_vae", "qalt"))
        coarse_gap = student_interval(differences("coarse_only", "qalt"))
        euler_gap = student_interval(differences("full_token_euler", "qalt"))
        latency = student_interval([float(unit["benchmark"]["latency_ratio"]) for unit in selected])
        memory = student_interval([float(unit["benchmark"]["memory_ratio"]) for unit in selected])
        update_error = max(
            abs(float(unit["empirical_update_ratio"]) - float(unit["expected_update_ratio"]))
            / float(unit["expected_update_ratio"])
            for unit in selected
        )
        checks = {
            "exact_split_equivalent": exact_difference["low"] > -config.equivalence_margin and exact_difference["high"] < config.equivalence_margin,
            "qalt_excess_nll_below_ceiling": qalt_excess["high"] < config.excess_nll_maximum,
            "diagonal_gap_positive": diagonal_gap["low"] > 0.0,
            "coarse_only_gap_positive": coarse_gap["low"] > 0.0,
            "latency_strictly_lower": latency["high"] < 1.0,
            "memory_strictly_lower": memory["high"] < 1.0,
            "update_accounting_within_ten_percent": update_error < 0.10,
            "full_exact_ties": max(abs(value) for value in differences("full_token_exact", "qalt")) < 1e-12,
            "finite_euler_gap_positive": euler_gap["low"] > 0.0,
            "haar_roundtrip": max(float(unit["max_haar_roundtrip_error"]) for unit in selected) < 1e-10,
        }
        modalities[modality] = {
            "qalt_minus_oracle_nll": qalt_excess,
            "exact_split_minus_qalt_nll": exact_difference,
            "diagonal_minus_qalt_nll": diagonal_gap,
            "coarse_only_minus_qalt_nll": coarse_gap,
            "euler_minus_qalt_nll": euler_gap,
            "latency_ratio": latency,
            "memory_ratio": memory,
            "maximum_relative_update_accounting_error": update_error,
            "checks": checks,
        }
        all_checks.extend(checks.values())
        if config.confirmation:
            registered = {
                "exact_lower": (differences("exact_split", "qalt"), -config.equivalence_margin, "greater"),
                "exact_upper": (differences("exact_split", "qalt"), config.equivalence_margin, "less"),
                "qalt_excess": (differences("qalt", "oracle"), config.excess_nll_maximum, "less"),
                "diagonal_gap": (differences("diagonal_vae", "qalt"), 0.0, "greater"),
                "coarse_gap": (differences("coarse_only", "qalt"), 0.0, "greater"),
                "euler_gap": (differences("full_token_euler", "qalt"), 0.0, "greater"),
                "latency_ratio": ([float(unit["benchmark"]["latency_ratio"]) for unit in selected], 1.0, "less"),
                "memory_ratio": ([float(unit["benchmark"]["memory_ratio"]) for unit in selected], 1.0, "less"),
            }
            for name, (values, null, alternative) in registered.items():
                confirmation_components[f"{modality}_{name}"] = one_sided_test(values, null, alternative)
    inference = None
    if config.confirmation:
        adjusted = holm_adjust({name: float(value["raw_p"]) for name, value in confirmation_components.items()})
        for name, value in confirmation_components.items():
            value["holm_p"] = adjusted[name]
            value["passes"] = adjusted[name] < 0.05
        hard_checks = all(
            modality["checks"][name]
            for modality in modalities.values()
            for name in ("update_accounting_within_ten_percent", "full_exact_ties", "haar_roundtrip")
        )
        inference = {
            "family": "16 preregistered one-sided paired Student tests with Holm correction",
            "components": confirmation_components,
            "hard_checks_pass": hard_checks,
            "all_registered_gates_pass": hard_checks and all(bool(value["passes"]) for value in confirmation_components.values()),
        }
    return {
        "mode": "confirmation" if config.confirmation else "development",
        "modalities": modalities,
        "confirmation_inference": inference,
        "all_registered_development_gates_pass": all(all_checks) if not config.confirmation else None,
        "all_registered_confirmation_gates_pass": None if not config.confirmation else inference["all_registered_gates_pass"],
        "maximum_claim": "Exact same-information controls tie; any strict quality gap is restricted to lossy, diagonal, or finite-Euler controls.",
    }


def write_csv(path: Path, units: list[dict[str, object]]) -> None:
    rows = []
    for unit in units:
        for method, nll in unit["nll_per_dimension"].items():
            rows.append({"seed": unit["seed"], "modality": unit["modality"], "method": method, "nll_per_dimension": nll})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--confirmation", action="store_true")
    args = parser.parse_args()
    if args.confirmation and args.seeds is not None:
        raise SystemExit("confirmation uses only frozen seeds 800..829")
    if args.confirmation:
        config = Config(seeds=tuple(range(800, 830)), confirmation=True)
    else:
        config = Config(seeds=tuple(args.seeds)) if args.seeds is not None else Config()
    if not config.confirmation and any(seed >= 800 for seed in config.seeds):
        raise SystemExit("confirmation seeds are sealed until a separate freeze commit")
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    write_json(args.output / "config.json", {**asdict(config), "provenance": provenance()})
    units = []
    for seed in config.seeds:
        for modality in ("image", "video"):
            print(f"[multiscale-qalt] seed={seed} modality={modality}", flush=True)
            unit = run_modality(config, seed, modality)
            units.append(unit)
            write_json(args.output / f"seed_{seed}_{modality}.json", unit)
    write_csv(args.output / "metrics.csv", units)
    summary = summarize(config, units)
    summary["runtime_seconds"] = time.perf_counter() - started
    write_json(args.output / "summary.json", summary)
    print(json.dumps(summary["modalities"], indent=2), flush=True)
    passed = summary["all_registered_confirmation_gates_pass"] if config.confirmation else summary["all_registered_development_gates_pass"]
    if not passed:
        raise SystemExit("one or more registered gates failed")


if __name__ == "__main__":
    main()
