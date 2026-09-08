"""Frozen observation-only conditional-CDF development study; no real-data access."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import subprocess
import time

import numpy as np
import scipy
from scipy.special import ndtr

from qalt.conditional_cdf import ConditionalCDFFlow

THETA = 0.65
CONFIG = {"theta": THETA, "dimensions": [8, 32], "train_sizes": [256, 1024, 4096],
          "seeds": [7101, 7102, 7103], "worlds": ["tree", "distant"],
          "evaluation_size": 4096, "sample_size": 256,
          "bins_rule": "ceil(n**0.25)", "bisection_steps": 52,
          "wall_budget_seconds": 570}
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def digest_array(array):
    array = np.ascontiguousarray(array)
    return hashlib.sha256(str(array.shape).encode() + array.dtype.str.encode()
                          + array.tobytes()).hexdigest()


def true_cdf(value, context, theta=THETA):
    return value + theta * np.cos(2 * np.pi * context) * np.sin(2 * np.pi * value) / (2 * np.pi)


def true_inverse(uniform, context, theta=THETA):
    """Strictly monotone known-world CDF inversion, used only by data generation."""
    uniform, context = np.broadcast_arrays(uniform, context)
    lo, hi = np.zeros_like(uniform), np.ones_like(uniform)
    for _ in range(CONFIG["bisection_steps"]):
        mid = (lo + hi) / 2
        below = true_cdf(mid, context, theta) < uniform
        lo, hi = np.where(below, mid, lo), np.where(below, hi, mid)
    return (lo + hi) / 2


def world_sample(rng, count, dimension, world):
    # Full vectors are independent statistical units. World parameters are
    # confined to the generator/evaluator and never passed to the fitting API.
    values = ndtr(rng.standard_normal((count, dimension)))
    if world == "tree":
        for j in range(1, dimension):
            values[:, j] = true_inverse(values[:, j], values[:, (j - 1) // 2])
    elif world == "distant":
        values[:, -1] = true_inverse(values[:, -1], values[:, 0])
    else:
        raise ValueError("unknown world")
    return values


def true_log_prob(values, world):
    if world == "tree":
        parents = (np.arange(1, values.shape[1]) - 1) // 2
        return np.log1p(THETA * np.cos(2 * np.pi * values[:, 1:])
                        * np.cos(2 * np.pi * values[:, parents])).sum(axis=1)
    if world == "distant":
        return np.log1p(THETA * np.cos(2 * np.pi * values[:, 0])
                        * np.cos(2 * np.pi * values[:, -1]))
    raise ValueError("unknown world")


def graph(dimension, world):
    parents = np.arange(dimension, dtype=np.int64) - 1
    if world == "tree":
        parents[1:] = (np.arange(1, dimension) - 1) // 2
    return parents


def latent_copy_sample(flow, coarse_noise, decoder_noise):
    """Identical fitted stochastic decoder; all D prior coordinates are counted."""
    copied = ConditionalCDFFlow(flow.parents, flow.probabilities, bins=flow.bins)
    return copied.decode(np.concatenate([coarse_noise, decoder_noise], axis=1))


def write_json(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def evaluate_cell(world, dimension, n, seed):
    world_id = CONFIG["worlds"].index(world)
    streams = np.random.SeedSequence([seed, world_id, dimension, n]).spawn(3)
    train = world_sample(np.random.default_rng(streams[0]), n, dimension, world)
    evaluation = world_sample(np.random.default_rng(streams[1]), CONFIG["evaluation_size"], dimension, world)
    parents, bins = graph(dimension, world), int(math.ceil(n ** 0.25))
    start = time.perf_counter()
    fitted = ConditionalCDFFlow.fit(train, parents, bins=bins)
    fit_seconds = time.perf_counter() - start
    # This is the first use of the true density after fitting is complete.
    losses = (true_log_prob(evaluation, world) - fitted.log_prob(evaluation)) / dimension
    gaussian = np.random.default_rng(streams[2]).standard_normal((CONFIG["sample_size"], dimension))
    decode_times = []
    for _ in range(3):
        start = time.perf_counter()
        samples, inverse_logdet = fitted.decode(gaussian)
        decode_times.append(time.perf_counter() - start)
    recovered, forward_logdet = fitted.encode(samples)
    encoded_eval, eval_logdet = fitted.encode(evaluation[:CONFIG["sample_size"]])
    eval_recovered, eval_inverse_logdet = fitted.decode(encoded_eval)
    copied, copied_logdet = latent_copy_sample(fitted, gaussian[:, :1], gaussian[:, 1:])
    base_logprob = -.5 * (dimension * np.log(2 * np.pi) + (gaussian ** 2).sum(axis=1))
    record = {
        "world": world, "dimension": dimension, "train_size": n, "seed": seed, "bins": bins,
        "evaluation_size": len(evaluation), "prior_dimension": dimension,
        "kl_per_coordinate_mc": float(losses.mean()),
        "kl_per_coordinate_mc_se": float(losses.std(ddof=1) / np.sqrt(len(losses))),
        "fit_seconds": fit_seconds, "sample_seconds_median": float(np.median(decode_times)),
        "sample_seconds_repetition_1": decode_times[0], "sample_seconds_repetition_2": decode_times[1],
        "sample_seconds_repetition_3": decode_times[2],
        "max_gaussian_roundtrip": float(np.max(np.abs(gaussian - recovered))),
        "max_observation_roundtrip": float(np.max(np.abs(eval_recovered - evaluation[:len(eval_recovered)]))),
        "max_logdet_cancellation": float(max(np.max(np.abs(inverse_logdet + forward_logdet)),
                                              np.max(np.abs(eval_logdet + eval_inverse_logdet)))),
        "max_density_identity_error": float(np.max(np.abs(fitted.log_prob(samples) + inverse_logdet - base_logprob))),
        "latent_copy_samples_equal": bool(np.array_equal(samples, copied)),
        "latent_copy_logdet_equal": bool(np.array_equal(inverse_logdet, copied_logdet)),
        "latent_copy_max_sample_difference": float(np.max(np.abs(samples - copied))),
        "theorem_omitted_context_floor_joint": THETA ** 2 / (8 * (1 + abs(THETA))) if world == "distant" else 0.,
        "theorem_omitted_context_floor_per_coordinate": THETA ** 2 / (8 * (1 + abs(THETA)) * dimension) if world == "distant" else 0.,
        "train_sha256": digest_array(train), "evaluation_sha256": digest_array(evaluation),
        "prior_sha256": digest_array(gaussian), "sample_sha256": digest_array(samples),
        "parents_sha256": digest_array(parents),
        "parameters_sha256": hashlib.sha256("".join(digest_array(p) for p in fitted.probabilities).encode()).hexdigest(),
        "table_probability_entries": sum(p.size for p in fitted.probabilities),
    }
    record["numerical_checks_pass"] = bool(record["max_gaussian_roundtrip"] <= 1e-9
        and record["max_observation_roundtrip"] <= 1e-10
        and record["max_logdet_cancellation"] <= 1e-8
        and record["max_density_identity_error"] <= 1e-8
        and record["latent_copy_samples_equal"] and record["latent_copy_logdet_equal"])
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if commit != args.source_commit:
        raise SystemExit("source commit differs from requested revision")
    paths = ["qalt/src/qalt/conditional_cdf.py", "qalt/experiments/conditional_cdf_validation/run.py",
             "qalt/experiments/conditional_cdf_validation/PROTOCOL.md", "qalt/experiments/conditional_cdf_validation/run.slurm",
             "qalt/tests/test_conditional_cdf_validation.py", "qalt/tests/test_conditional_cdf.py",
             "research/transport_iteration_20260908/conditional_learnability.md"]
    if subprocess.check_output(["git", "status", "--porcelain", "--", *paths], cwd=ROOT, text=True).strip():
        raise SystemExit("model/protocol/runner must be committed and unchanged")
    args.output.mkdir(parents=True, exist_ok=False)
    provenance = {"real_data_accessed": False, "test_data_accessed": False, "quality_advantage_established": False, "cost_advantage_established": False, "config": CONFIG, "source_commit": commit, "source_hashes": {
        p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in paths},
        "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
        "platform": platform.platform(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "thread_environment": {k: os.environ.get(k) for k in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]}}
    write_json(args.output / "provenance.json", provenance)
    start, records = time.perf_counter(), []
    try:
        for world in CONFIG["worlds"]:
            for dimension in CONFIG["dimensions"]:
                for n in CONFIG["train_sizes"]:
                    for seed in CONFIG["seeds"]:
                        if time.perf_counter() - start > CONFIG["wall_budget_seconds"]:
                            raise TimeoutError("frozen wall budget reached; no partial success claim")
                        record = evaluate_cell(world, dimension, n, seed)
                        write_json(args.output / f"{world}_d{dimension}_n{n}_s{seed}.json", record)
                        records.append(record)
                        if not record["numerical_checks_pass"]:
                            raise RuntimeError("numerical identity check failed; preserve result and stop")
        summary = {"cells_completed": len(records), "wall_seconds": time.perf_counter() - start,
                   "peak_rss_native": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                   "peak_rss_units": "bytes" if platform.system() == "Darwin" else "KiB",
                   "all_numerical_checks_pass": all(r["numerical_checks_pass"] for r in records),
                   "interpretation": "Development-only MC density errors; no rate fit or latent superiority claim."}
        write_json(args.output / "summary.json", summary)
        hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(args.output.glob("*.json"))}
        write_json(args.output / "COMPLETE.json", {"source_commit": commit, "result_hashes": hashes})
    except BaseException as exc:
        write_json(args.output / "FAILED.json", {"source_commit": commit, "cells_completed": len(records),
                  "exception_type": type(exc).__name__, "message": str(exc), "wall_seconds": time.perf_counter() - start})
        raise


if __name__ == "__main__":
    main()
