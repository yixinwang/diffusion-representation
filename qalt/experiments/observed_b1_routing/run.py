from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from qalt.data_integrity import load_cifar_training_batches, stable_json_hash, stratified_cifar_split
from qalt.multiscale import haar_forward, haar_inverse
from qalt.observed_routing import (
    DETAIL_COUNT,
    empirical_bernstein_upper,
    fit_conditional_model,
    image_haar,
    mixture_log_bounds,
    paired_dequantize,
    per_image_channel_log_prob,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA = Path("/ocean/datasets/community/cifar/cifar-10/cifar-10-batches-py")
DEVELOPMENT_SEEDS = tuple(range(1100, 1105))


def sha256_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).view(np.uint8)).hexdigest()


def bootstrap_interval(values: np.ndarray, rng: np.random.Generator, draws: int = 2000) -> tuple[float, float]:
    sample = np.asarray(values, dtype=float)
    means = np.empty(draws)
    for start in range(0, draws, 100):
        stop = min(start + 100, draws)
        indices = rng.integers(0, len(sample), size=(stop - start, len(sample)))
        means[start:stop] = np.mean(sample[indices], axis=1)
    return tuple(float(value) for value in np.quantile(means, (0.025, 0.975)))


def model_parameters(model) -> int:
    location = sum(beta.size for beta in model.coefficients)
    mixture = sum(component.weights.size + component.scales.size for channel in model.mixtures for component in channel)
    return int(location + mixture + len(model.boundaries))


def build_coefficients(images: np.ndarray, record_ids: np.ndarray, seed: int, chunk: int = 512) -> tuple[np.ndarray, np.ndarray, str]:
    coarse_parts = []
    detail_parts = []
    digest = hashlib.sha256()
    for start in range(0, len(images), chunk):
        stop = min(start + chunk, len(images))
        values = paired_dequantize(images[start:stop], record_ids[start:stop], seed)
        digest.update(values.tobytes())
        coarse, detail = image_haar(values)
        coarse_parts.append(coarse)
        detail_parts.append(detail)
        print(f"coefficients {stop}/{len(images)}", flush=True)
    return np.concatenate(coarse_parts), np.concatenate(detail_parts), digest.hexdigest()


def run(seed: int, output: Path, data_root: Path, source_commit: str) -> dict:
    if seed not in DEVELOPMENT_SEEDS:
        raise ValueError(f"seed {seed} is outside frozen development seeds {DEVELOPMENT_SEEDS}")
    images, labels = load_cifar_training_batches(data_root)
    split = stratified_cifar_split(labels)
    fit_indices = np.asarray(split["fit"], dtype=np.int64)
    validation_indices = np.asarray(split["validation"], dtype=np.int64)
    fit_coarse, fit_detail, fit_dequant_hash = build_coefficients(images[fit_indices], fit_indices, seed)
    validation_coarse, validation_detail, validation_dequant_hash = build_coefficients(
        images[validation_indices], validation_indices, seed
    )
    rng = np.random.default_rng(seed)
    total_sites = len(fit_coarse) * 16 * 16
    sample = np.sort(rng.choice(total_sites, size=min(250_000, total_sites), replace=False))

    print("fit O4", flush=True)
    o4, sample_hash = fit_conditional_model(fit_coarse, fit_detail, 4, False, True, rng, sample=sample)
    print("fit I4", flush=True)
    i4, i_sample_hash = fit_conditional_model(fit_coarse, fit_detail, 4, True, True, rng, sample=sample, boundaries=o4.boundaries)
    print("fit controls", flush=True)
    o1, _ = fit_conditional_model(
        fit_coarse, fit_detail, 1, False, True, rng, coefficients=o4.coefficients, boundaries=o4.boundaries, sample=sample
    )
    o4_unconditional, _ = fit_conditional_model(
        fit_coarse, fit_detail, 4, False, False, rng, coefficients=o4.coefficients, boundaries=o4.boundaries, sample=sample
    )
    o8, _ = fit_conditional_model(
        fit_coarse, fit_detail, 8, False, True, rng, coefficients=o4.coefficients, boundaries=o4.boundaries, sample=sample
    )
    i8, _ = fit_conditional_model(
        fit_coarse, fit_detail, 8, True, True, rng, coefficients=i4.coefficients, boundaries=o4.boundaries, sample=sample
    )
    if sample_hash != i_sample_hash:
        raise AssertionError("mixture sample identities differ")

    models = {"o1": o1, "o4": o4, "o4_unconditional": o4_unconditional, "o8": o8, "i4": i4, "i8": i8}
    scores = {}
    for name, model in models.items():
        print(f"score {name}", flush=True)
        scores[name] = per_image_channel_log_prob(model, validation_coarse, validation_detail)
    if not np.array_equal(scores["o4"], per_image_channel_log_prob(o4, validation_coarse, validation_detail)):
        raise AssertionError("exact-copy score does not tie")

    comparisons = {}
    for name, difference in {
        "o1_minus_o4": (scores["o4"] - scores["o1"]).sum(axis=1) / DETAIL_COUNT,
        "o4_unconditional_minus_o4": (scores["o4"] - scores["o4_unconditional"]).sum(axis=1) / DETAIL_COUNT,
        "o4_minus_o8": (scores["o8"] - scores["o4"]).sum(axis=1) / DETAIL_COUNT,
        "i4_minus_i8": (scores["i8"] - scores["i4"]).sum(axis=1) / DETAIL_COUNT,
    }.items():
        interval = bootstrap_interval(difference, rng)
        comparisons[name] = {"mean": float(np.mean(difference)), "bootstrap_95": list(interval)}

    channel_regret = (scores["i4"] - scores["o4"]) / DETAIL_COUNT
    log_lower, log_upper = mixture_log_bounds()
    log_width = log_upper - log_lower
    routes = []
    for mask in range(1 << 9):
        one_shot = [channel for channel in range(9) if mask & (1 << channel)]
        regret = channel_regret[:, one_shot].sum(axis=1) if one_shot else np.zeros(len(validation_indices))
        if one_shot:
            lower = -log_width * len(one_shot) / 9.0
            upper = log_width * len(one_shot) / 9.0
            bound = empirical_bernstein_upper(regret, lower, upper, 1 << 9, 0.05)
        else:
            lower = upper = bound = 0.0
        depth = 9 - len(one_shot) + int(bool(one_shot))
        routes.append(
            {
                "mask": mask,
                "one_shot_channels": one_shot,
                "mean_regret": float(np.mean(regret)),
                "upper_certificate": float(bound),
                "analytic_interval": [float(lower), float(upper)],
                "depth": depth,
                "eligible": bool(bound <= 0.01),
            }
        )
    selected = min((route for route in routes if route["eligible"]), key=lambda route: (route["depth"], route["mask"]))

    probe = paired_dequantize(images[fit_indices[:4]], fit_indices[:4], seed)
    transformed = haar_forward(probe, axes=(2, 3))
    roundtrip = float(np.max(np.abs(haar_inverse(transformed, axes=(2, 3)) - probe)))
    summary = {
        "seed": seed,
        "protocol_origin_commit": "b8f585d",
        "source_commit": source_commit,
        "split_hash": stable_json_hash(split),
        "fit_size": len(fit_indices),
        "validation_size": len(validation_indices),
        "fit_input_hash": sha256_array(images[fit_indices]),
        "validation_input_hash": sha256_array(images[validation_indices]),
        "fit_dequant_hash": fit_dequant_hash,
        "validation_dequant_hash": validation_dequant_hash,
        "mixture_sample_hash": sample_hash,
        "model_parameters": {name: model_parameters(model) for name, model in models.items()},
        "comparisons": comparisons,
        "channel_o4_regret_vs_i4": [float(value) for value in np.mean(channel_regret, axis=0)],
        "selected_route": selected,
        "gates": {
            "o4_beats_o1": comparisons["o1_minus_o4"]["bootstrap_95"][0] > 0.01,
            "conditioning_helps": comparisons["o4_unconditional_minus_o4"]["bootstrap_95"][0] > 0.0,
            "o4_equivalent_o8": comparisons["o4_minus_o8"]["bootstrap_95"][0] > -0.01
            and comparisons["o4_minus_o8"]["bootstrap_95"][1] < 0.01,
            "certified_depth_reduction": selected["depth"] < 9,
            "roundtrip": roundtrip < 1e-6,
            "finite": all(np.all(np.isfinite(value)) for value in scores.values()),
        },
        "roundtrip_max_error": roundtrip,
        "test_access": False,
    }
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / "validation_cluster_scores.npz", **scores)
    (output / "routes.json").write_text(json.dumps(routes, indent=2) + "\n")
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output / "config.json").write_text(
        json.dumps({"seed": seed, "data_root": str(data_root), "source_commit": source_commit}, indent=2) + "\n"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    summary = run(args.seed, args.output, args.data_root, args.source_commit)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"FAILED: {error}", file=sys.stderr, flush=True)
        raise
