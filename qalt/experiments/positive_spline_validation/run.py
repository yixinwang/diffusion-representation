"""Frozen CPU development validation of the convex C1 spline estimator.

Simulator inversion and true-density evaluation are isolated from fitting.
Independent full arrays are the statistical units. No real data are loaded.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import platform
import resource
import signal
import subprocess
import sys
import time

import numpy as np
import scipy
from scipy.spatial.distance import cdist

from qalt.positive_spline_flow import PositiveSplineFlow

ROOT = Path(__file__).resolve().parents[3]
SOURCE_FILES = (
    "qalt/src/qalt/positive_density_spline.py",
    "qalt/src/qalt/positive_spline_flow.py",
    "qalt/tests/test_positive_density_spline.py",
    "qalt/tests/test_positive_spline_flow.py",
    "qalt/experiments/positive_spline_validation/run.py",
    "qalt/experiments/positive_spline_validation/PROTOCOL.md",
    "qalt/experiments/positive_spline_validation/run.slurm",
)
RHO = 0.65
BINS = 4
MAX_ITERATIONS = 500
GAP_TOLERANCE = 1e-4
LOWER, UPPER = 0.175, 3.3
EVALUATION_COUNT = 4096
GENERATION_COUNT = 256
CHECK_TOLERANCE = 1e-8
MAX_RUNTIME = 600


class RuntimeLimit(Exception):
    pass


def _alarm(signum, frame):
    raise RuntimeLimit("registered run-time limit reached")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def array_hash(values):
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256(str((array.shape, array.dtype.str)).encode())
    digest.update(array.view(np.uint8))
    return digest.hexdigest()


def atomic_json(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def source_guard(expected_commit):
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if commit != expected_commit:
        raise ValueError("HEAD differs from the submitted source commit")
    hashes = {}
    for relative in SOURCE_FILES:
        expected = subprocess.check_output(["git", "rev-parse", f"HEAD:{relative}"], cwd=ROOT, text=True).strip()
        actual = subprocess.check_output(["git", "hash-object", relative], cwd=ROOT, text=True).strip()
        if actual != expected:
            raise ValueError(f"source differs from committed version: {relative}")
        hashes[relative] = sha256(ROOT / relative)
    return commit, hashes


def _stream(seed, dimension, sample_count, world, purpose):
    # All identities are explicit; neither the order of cells nor fitting RNG
    # consumption changes any simulation or Gaussian source stream.
    world_id = {"local": 0, "distant": 1}[world]
    return np.random.default_rng(np.random.SeedSequence([seed, dimension, sample_count, world_id, purpose]))


def simulate_observations(count, dimension, rng, world):
    """Return observed coordinates only, with uniform marginals.

    Conditional F(t|c)=t+rho*cos(2pi*c)*sin(2pi*t)/(2pi).
    Vectorized bisection belongs only to this simulator. No teacher coefficients,
    uniforms, or inverses are supplied to PositiveSplineFlow.fit.
    """
    source = rng.uniform(size=(count, dimension))
    values = source.copy()
    children = np.arange(1, dimension, 2) if world == "local" else np.array([dimension - 1])
    parents = children - 1 if world == "local" else np.array([0])
    target = source[:, children]
    amplitude = RHO * np.cos(2 * np.pi * source[:, parents])
    low, high = np.zeros_like(target), np.ones_like(target)
    for _ in range(54):
        midpoint = low + (high - low) / 2
        cdf = midpoint + amplitude * np.sin(2 * np.pi * midpoint) / (2 * np.pi)
        go_right = cdf < target
        low = np.where(go_right, midpoint, low)
        high = np.where(go_right, high, midpoint)
    values[:, children] = low + (high - low) / 2
    if np.any((values <= 0) | (values >= 1)):
        raise FloatingPointError("simulator reached a floating-point endpoint")
    return values


def true_log_density(values, world):
    """Evaluation oracle only; never called by a fitting operation."""
    transformed = np.cos(2 * np.pi * values)
    if world == "local":
        return np.log1p(RHO * transformed[:, 0::2] * transformed[:, 1::2]).sum(axis=1)
    return np.log1p(RHO * transformed[:, 0] * transformed[:, -1])


def graph(dimension, world):
    if world == "local":
        return tuple(() if j % 2 == 0 else (j - 1,) for j in range(dimension)), tuple(j % 2 for j in range(dimension))
    return ((),) + tuple((j - 1,) for j in range(1, dimension)), (0,) + (1,) * (dimension - 1)


def mean_mcse(values):
    values = np.asarray(values, dtype=np.float64)
    return {"mean": float(np.mean(values)), "mcse": float(np.std(values, ddof=1) / math.sqrt(len(values))), "independent_array_count": len(values)}


def moments(values):
    cosine = np.cos(2 * np.pi * values)
    return {
        "marginal_cosine": [mean_mcse(cosine[:, j]) for j in range(values.shape[1])],
        "adjacent_pair_cosine_product": [mean_mcse(cosine[:, j] * cosine[:, j + 1]) for j in range(0, values.shape[1], 2)],
        "first_last_cosine_product": mean_mcse(cosine[:, 0] * cosine[:, -1]),
    }


def energy_score(real, generated):
    """Unbiased U-statistic energy score, Euclidean distance / sqrt(D).

    Each generated array and each evaluation array is an independent source
    draw. Exact-copy comparisons reuse all source/evaluation pairs. This point
    estimate is exploratory and supplies no significance or coverage claim.
    """
    n = len(generated)
    scale = math.sqrt(real.shape[1])
    cross = cdist(generated, real) / scale
    within = cdist(generated, generated) / scale
    np.fill_diagonal(within, 0)
    return float(cross.mean() - within.sum() / (2 * n * (n - 1)))


def evaluate_cell(cell, smoke=False):
    world, dimension, count, seed = cell
    eval_count = 64 if smoke else EVALUATION_COUNT
    source_count = 16 if smoke else GENERATION_COUNT
    max_iterations = 2 if smoke else MAX_ITERATIONS
    started = time.perf_counter()
    observations = simulate_observations(count, dimension, _stream(seed, dimension, count, world, 1), world)
    parents, groups = graph(dimension, world)
    fit_start = time.perf_counter()
    model, fitting = PositiveSplineFlow.fit(observations, parents, groups, bins=BINS,
        max_iterations=max_iterations, gap_tolerance=GAP_TOLERANCE, lower=LOWER, upper=UPPER)
    fit_seconds = time.perf_counter() - fit_start
    # Development observations are created only after this cell's model freezes.
    evaluation = simulate_observations(eval_count, dimension, _stream(seed, dimension, count, world, 2), world)
    source = _stream(seed, dimension, count, world, 3).normal(size=(source_count, dimension))
    score_start = time.perf_counter()
    fitted_log_density = model.log_prob(evaluation)
    oracle_log_density = true_log_density(evaluation, world)
    log_ratio = oracle_log_density - fitted_log_density
    score_seconds = time.perf_counter() - score_start
    generation_start = time.perf_counter()
    generated, decode_logdet = model.decode(source)
    generation_seconds = time.perf_counter() - generation_start
    recovered, encode_logdet = model.encode(generated)
    generated_log_density = model.log_prob(generated)
    normal_log_density = -.5 * (dimension * math.log(2 * math.pi) + np.sum(source**2, axis=1))
    checks = {
        "source_roundtrip_max": float(np.max(np.abs(recovered - source))),
        "logdet_cancellation_max": float(np.max(np.abs(decode_logdet + encode_logdet))),
        "normalized_density_identity_max": float(np.max(np.abs(generated_log_density + decode_logdet - normal_log_density))),
        "absolute_tolerance": CHECK_TOLERANCE,
    }
    checks["passed"] = all(checks[k] <= CHECK_TOLERANCE for k in (
        "source_roundtrip_max", "logdet_cancellation_max", "normalized_density_identity_max"))
    copied = PositiveSplineFlow(model.parents, model.groups, model.models)
    copy_generated, copy_logdet = copied.decode(source)
    copy_log_density = copied.log_prob(evaluation)
    copy_equal = bool(np.array_equal(copy_generated, generated) and np.array_equal(copy_logdet, decode_logdet)
                      and np.array_equal(copy_log_density, fitted_log_density))
    if not copy_equal:
        raise AssertionError("identical stochastic decoder copy failed exact equality")
    if not np.all(np.isfinite(log_ratio)):
        raise FloatingPointError("nonfinite evaluation log density")
    status = "optimization_gap_not_met" if not fitting.converged else "optimization_gap_met"
    if not checks["passed"]:
        status = "numerical_validation_failed"
    return {
        "world": world, "dimension": dimension, "train_array_count": count, "seed": seed,
        "status": status, "claim_status": "development_only_no_promotion",
        "fit": {**asdict(fitting), "converged": fitting.converged,
                "per_coordinate_optimization_gap": fitting.per_coordinate_optimization_gap,
                "pooled_responses_are_independent": False},
        "evaluation": {"joint_kl": mean_mcse(log_ratio), "per_coordinate_kl": mean_mcse(log_ratio / dimension),
            "negative_kl_estimates_are_not_clipped": True,
            "energy_score": energy_score(evaluation, generated),
            "energy_convention": "E||X-Y||/sqrt(D) - 0.5 E||X-Xprime||/sqrt(D), unbiased self U-statistic",
            "observed_moments": moments(evaluation), "generated_moments": moments(generated),
            "mcse_scope": "independent full arrays conditional on this fitted model; no retraining or adaptive coverage"},
        "numerical_checks": checks,
        "same_information_stochastic_decoder_copy": {"exact_equality": copy_equal,
            "gaussian_source_dimension": dimension, "training_cost": "inherits the full original fit",
            "sampling_cost": "same complete computation and coefficients"},
        "timings": {"fit_seconds": fit_seconds, "density_score_seconds": score_seconds,
            "generation_seconds": generation_seconds, "total_seconds": time.perf_counter() - started,
            "interpretation": "single CPU measurements, no efficiency superiority claim"},
        "stream_ids": {"training": 1, "development": 2, "gaussian_source": 3},
        "hashes": {"training_arrays": array_hash(observations), "development_arrays": array_hash(evaluation),
            "gaussian_source": array_hash(source), "generated_arrays": array_hash(generated)},
        "model": {"parents": model.parents, "groups": model.groups,
            "heads": [{"bins": m.bins, "context_dimension": m.context_dimension, "lower": m.lower,
                       "upper": m.upper, "coefficients": m.coefficients.tolist()} for m in model.models]},
    }


def finish(output, manifest, results, remaining, termination, started, error=None):
    summary = {"manifest": manifest, "termination": termination, "completed_cell_count": len(results),
        "remaining_cells": remaining, "results": results, "runtime_seconds": time.perf_counter() - started,
        "optimization_gap_met_cells": sum(r["fit"]["converged"] for r in results),
        "numerically_valid_cells": sum(r["numerical_checks"]["passed"] for r in results),
        "quality_advantage_established": False, "cost_advantage_established": False,
        "population_confirmation_established": False, "error": error}
    atomic_json(output / "summary.json", summary)
    payload = sorted(p for p in output.glob("*.json") if p.name != "COMPLETE.json")
    hashes = {p.name: sha256(p) for p in payload}
    (output / "SHA256SUMS").write_text("".join(f"{h}  {name}\n" for name, h in hashes.items()))
    atomic_json(output / "COMPLETE.json", {"status": termination, "completed_cell_count": len(results),
        "payload_hashes": hashes, "sha256sums_hash": sha256(output / "SHA256SUMS"),
        "all_registered_cells_completed": not remaining and termination == "completed"})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--smoke", action="store_true", help="API smoke only: one tiny cell, two optimizer steps")
    parser.add_argument("--max-runtime-seconds", type=int, default=MAX_RUNTIME)
    args = parser.parse_args()
    started = time.perf_counter()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite an existing result directory")
    if not 1 <= args.max_runtime_seconds <= MAX_RUNTIME:
        raise ValueError("runtime limit must lie between one and 600 seconds")
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        if os.environ.get(variable) != "1":
            raise ValueError(f"{variable}=1 is required for the registered CPU run")
    commit, hashes = source_guard(args.source_commit)
    cells = [("local", 4, 32, 8101)] if args.smoke else list(itertools.product(
        ("local", "distant"), (8, 32), (256, 1024, 4096), (8101, 8102, 8103)))
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {"status": "api_smoke_only" if args.smoke else "frozen_synthetic_development",
        "source_commit": commit, "source_hashes": hashes, "cell_count": len(cells), "cells": cells,
        "rho": RHO, "bins": BINS, "max_iterations": 2 if args.smoke else MAX_ITERATIONS,
        "gap_tolerance": GAP_TOLERANCE, "coefficient_lower": LOWER, "coefficient_upper": UPPER,
        "evaluation_array_count": 64 if args.smoke else EVALUATION_COUNT,
        "gaussian_source_count": 16 if args.smoke else GENERATION_COUNT,
        "runtime_limit_seconds": args.max_runtime_seconds, "real_data_accessed": False,
        "independent_units": "complete arrays; shared-group sites are dependent within an array",
        "method_identity": "convex positive-density C1 estimator, distinct from the neural spline pilot",
        "python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__,
        "platform": platform.platform(), "hostname": platform.node(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "command": [sys.executable, *sys.argv]}
    atomic_json(args.output / "manifest.json", manifest)
    results, remaining = [], list(cells)
    signal.signal(signal.SIGALRM, _alarm)
    signal.setitimer(signal.ITIMER_REAL, max(.01, args.max_runtime_seconds - (time.perf_counter() - started)))
    termination, error = "completed", None
    try:
        while remaining:
            cell = remaining[0]
            result = evaluate_cell(cell, smoke=args.smoke)
            name = f"cell_{cell[0]}_d{cell[1]}_n{cell[2]}_seed{cell[3]}.json"
            atomic_json(args.output / name, result)
            results.append(result)
            remaining.pop(0)
            atomic_json(args.output / "progress.json", {"completed": len(results), "remaining": remaining,
                "last_cell": name, "last_status": result["status"]})
            print(json.dumps({"cell": cell, "status": result["status"],
                "per_coordinate_gap": result["fit"]["per_coordinate_optimization_gap"]}), flush=True)
            if not result["numerical_checks"]["passed"]:
                termination = "failed_numerical_validation"
                error = f"numerical validation failed in preserved cell {name}"
                break
    except RuntimeLimit as exc:
        termination, error = "runtime_limit_partial", str(exc)
    except Exception as exc:
        termination, error = "failed_partial", f"{type(exc).__name__}: {exc}"
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        manifest["peak_rss_bytes"] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * (1 if sys.platform == "darwin" else 1024)
        finish(args.output, manifest, results, remaining, termination, started, error)
    if termination in ("failed_partial", "failed_numerical_validation"):
        raise RuntimeError(error)


if __name__ == "__main__":
    main()
