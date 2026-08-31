from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time
from typing import Callable

import numpy as np

from qalt.data_integrity import (
    adaptive_cifar_repair_split,
    load_cifar_training_batches,
    sha256_file,
    stable_json_hash,
)
from qalt.observed_block import (
    ARM_NAMES,
    diagnostic_sufficient_statistics,
    detail_to_blocks,
    fit_observed_block_models,
    image_haar_inverse,
    per_image_band_log_scores,
)
from qalt.observed_routing import image_haar, paired_dequantize
from qalt.rgb_block import recycle_component_normal


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA = Path("/ocean/datasets/community/cifar/cifar-10/cifar-10-batches-py")
DEVELOPMENT_SEEDS = tuple(range(2100, 2105))
PROTOCOL_ORIGIN_COMMIT = "56ae6e0"
PROTOCOL_HASH = "7bb11e4174c485b85a5f7f18c54acdc756494e4cf3eb93a73a042248a6fb9a7c"
OPTIMIZATION_CHILD_PROTOCOL_HASH = "9906bb5d395e91b61065a8d5cf59b3ceb63b86c6a5090a0685411b53bc83c8fa"
DEFAULT_MAX_ITERATIONS = 200
OPTIMIZATION_CHILD_MAX_ITERATIONS = 1_000
SEED_SCHEMA_VERSION = "observed_b1_rgb_block_seed_v1"


def sha256_array(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


def build_coefficients(
    images: np.ndarray,
    record_ids: np.ndarray,
    seed: int,
    chunk_images: int = 512,
    progress: Callable[[str], None] = print,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Dequantize by record key and return coarse/explicit-block arrays."""
    if chunk_images < 1:
        raise ValueError("chunk_images must be positive")
    coarse_parts = []
    block_parts = []
    digest = hashlib.sha256()
    for start in range(0, len(images), chunk_images):
        stop = min(start + chunk_images, len(images))
        values = paired_dequantize(images[start:stop], record_ids[start:stop], seed)
        digest.update(values.tobytes())
        coarse, detail = image_haar(values)
        coarse_parts.append(coarse)
        block_parts.append(detail_to_blocks(detail))
        progress(f"coefficients {stop}/{len(images)}")
    return np.concatenate(coarse_parts), np.concatenate(block_parts), digest.hexdigest()


def _current_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        text=True,
    ).strip()


def _jsonable_convergence(models) -> dict[str, object]:
    return {
        name: [[bool(value) for value in cell] for cell in model.converged]
        for name, model in models.arms().items()
    }


def _diagnostic_arrays(diagnostics) -> dict[str, np.ndarray]:
    names = (
        "pit_grid",
        "counts",
        "pit_leq_counts",
        "angular_counts",
        "angular_second_sums",
        "angular_fourth_sums",
        "responsibility_sums",
        "shape_eigenvalues",
        "scale_lower_hits",
        "scale_upper_hits",
        "weight_floor_hits",
        "shape_lower_hits",
        "shape_upper_hits",
        "normalized_energy_sum",
        "normalized_energy_outer_sum",
        "heatmap_b4_minus_i8_sum",
        "per_image_counts",
        "per_image_pit_leq_counts",
        "per_image_angular_counts",
        "per_image_angular_second_sums",
        "per_image_angular_fourth_sums",
        "per_image_responsibility_sums",
        "per_image_normalized_energy_count",
        "per_image_normalized_energy_sum",
        "per_image_normalized_energy_outer_sum",
        "per_image_heatmap_b4_minus_i8",
    )
    output = {name: np.asarray(getattr(diagnostics, name)) for name in names}
    for prefix, mapping in (
        ("per_image_clipped_location", diagnostics.per_image_clipped_location_counts),
        ("per_image_finite_score", diagnostics.per_image_finite_score_counts),
    ):
        for name, values in mapping.items():
            output[f"{prefix}_{name}"] = np.asarray(values)
    return output


def _same_prior_check(models, seed: int, draws: int = 100_000) -> dict[str, object]:
    mixture = models.b4.mixtures[0][0]
    base = np.random.default_rng(seed + 90_000).normal(size=(draws, 3))
    component, remapped = recycle_component_normal(base[:, 0], mixture.weights)
    frequencies = np.bincount(component, minlength=len(mixture.weights)) / draws
    return {
        "draws": draws,
        "component_frequencies": frequencies.tolist(),
        "maximum_frequency_error": float(np.max(np.abs(frequencies - mixture.weights))),
        "remapped_mean": float(np.mean(remapped)),
        "remapped_variance": float(np.var(remapped)),
        "joint_sample_finite": bool(np.all(np.isfinite(mixture.sample(base)))),
        "base_dimension": 3,
    }


def run(
    seed: int,
    output: Path,
    data_root: Path,
    source_commit: str,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
) -> dict[str, object]:
    started = time.perf_counter()
    if seed not in DEVELOPMENT_SEEDS:
        raise ValueError(f"seed {seed} is outside frozen development seeds {DEVELOPMENT_SEEDS}")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    if max_iterations not in {DEFAULT_MAX_ITERATIONS, OPTIMIZATION_CHILD_MAX_ITERATIONS}:
        raise ValueError(
            "max_iterations must select the frozen parent (200) or optimization child (1000)"
        )
    actual_commit = _current_commit()
    if source_commit != actual_commit:
        raise ValueError(f"declared source commit {source_commit} != checked-out {actual_commit}")
    protocol_path = PROJECT_ROOT / "qalt" / "theory" / "OBSERVED_B1_RGB_BLOCK_PROTOCOL.md"
    if sha256_file(protocol_path) != PROTOCOL_HASH:
        raise ValueError("frozen RGB-block protocol hash mismatch")
    if max_iterations == OPTIMIZATION_CHILD_MAX_ITERATIONS:
        child_protocol_path = (
            PROJECT_ROOT
            / "qalt"
            / "theory"
            / "OBSERVED_B1_RGB_BLOCK_OPTIMIZATION_CHILD_PROTOCOL.md"
        )
        if sha256_file(child_protocol_path) != OPTIMIZATION_CHILD_PROTOCOL_HASH:
            raise ValueError("frozen optimization-child protocol hash mismatch")
        execution_protocol_hash = OPTIMIZATION_CHILD_PROTOCOL_HASH
        study_status = "exploratory_optimization_child_no_coverage"
    else:
        execution_protocol_hash = PROTOCOL_HASH
        study_status = "adaptive_development_no_coverage"

    opened_files: list[str] = []
    print("load five allowlisted CIFAR training batches", flush=True)
    images, labels = load_cifar_training_batches(data_root, opened_files=opened_files)
    split = adaptive_cifar_repair_split(labels)
    fit_ids = np.asarray(split["fit"], dtype=np.int64)
    holdout_ids = np.asarray(split["repair_holdout"], dtype=np.int64)
    discovery_ids = np.asarray(split["excluded_discovery"], dtype=np.int64)
    fit_images = images[fit_ids].copy()
    holdout_images = images[holdout_ids].copy()
    holdout_labels = labels[holdout_ids].copy()
    del images
    if np.intersect1d(discovery_ids, fit_ids).size or np.intersect1d(discovery_ids, holdout_ids).size:
        raise AssertionError("discovery indices entered downstream v2 arrays")

    print("build fitting coefficients", flush=True)
    fit_coarse, fit_blocks, fit_dequant_hash = build_coefficients(
        fit_images,
        fit_ids,
        seed,
        progress=lambda message: print(f"fit {message}", flush=True),
    )
    print("build repair-holdout coefficients", flush=True)
    holdout_coarse, holdout_blocks, holdout_dequant_hash = build_coefficients(
        holdout_images,
        holdout_ids,
        seed,
        progress=lambda message: print(f"holdout {message}", flush=True),
    )
    print("fit registered arms", flush=True)
    models = fit_observed_block_models(
        fit_coarse,
        fit_blocks,
        rng=np.random.default_rng(seed),
        maximum_sites=250_000,
        max_iterations=max_iterations,
        progress=lambda message: print(f"model {message}", flush=True),
    )
    print("score repair holdout", flush=True)
    scores = per_image_band_log_scores(models, holdout_coarse, holdout_blocks, chunk_images=256)
    b4_e4_error = float(np.max(np.abs(scores["b4"] - scores["e4"])))
    if b4_e4_error > 1e-8:
        raise AssertionError(f"B4/E4 samplewise tie failed: {b4_e4_error}")
    print("compute registered diagnostics", flush=True)
    diagnostics = diagnostic_sufficient_statistics(models, holdout_coarse, holdout_blocks)
    if diagnostics.p4_product_max_abs > 1e-8:
        raise AssertionError(f"P4 fitted-marginal tie failed: {diagnostics.p4_product_max_abs}")
    if diagnostics.z4_component_marginal_max_abs > 1e-12:
        raise AssertionError(
            f"Z4 component-marginal tie failed: {diagnostics.z4_component_marginal_max_abs}"
        )

    probe = paired_dequantize(fit_images[:4], fit_ids[:4], seed)
    probe_coarse, probe_detail = image_haar(probe)
    reconstructed = image_haar_inverse(probe_coarse, detail_to_blocks(probe_detail))
    roundtrip = float(np.max(np.abs(reconstructed - probe)))
    test_access = any(Path(path).name == "test_batch" for path in opened_files)
    if test_access:
        raise AssertionError("official CIFAR test batch entered the opened-file ledger")
    second_deviation, fourth_deviation = diagnostics.angular_max_deviations()
    detail_spatial_shape = [int(value) for value in holdout_blocks.shape[1:3]]
    sites_per_band = int(np.prod(detail_spatial_shape))
    if detail_spatial_shape != [16, 16] or sites_per_band != 256:
        raise AssertionError("registered CIFAR detail grid must be exactly 16x16")
    summary: dict[str, object] = {
        "schema_version": SEED_SCHEMA_VERSION,
        "status": study_status,
        "seed": seed,
        "sites_per_band": sites_per_band,
        "detail_spatial_shape": detail_spatial_shape,
        "protocol_origin_commit": PROTOCOL_ORIGIN_COMMIT,
        "protocol_hash": PROTOCOL_HASH,
        "execution_protocol_hash": execution_protocol_hash,
        "max_iterations": max_iterations,
        "source_commit": source_commit,
        "split_hash": stable_json_hash(split),
        "fit_size": int(len(fit_ids)),
        "holdout_size": int(len(holdout_ids)),
        "excluded_discovery_size": int(len(discovery_ids)),
        "fit_input_hash": sha256_array(fit_images),
        "holdout_input_hash": sha256_array(holdout_images),
        "fit_record_ids_hash": sha256_array(fit_ids.astype("<i8")),
        "holdout_record_ids_hash": sha256_array(holdout_ids.astype("<i8")),
        "holdout_labels_hash": sha256_array(holdout_labels.astype("<i8")),
        "fit_dequant_hash": fit_dequant_hash,
        "holdout_dequant_hash": holdout_dequant_hash,
        "site_sample_hash": models.sample_hash,
        "opened_files": opened_files,
        "official_test_deserialized": test_access,
        "downstream_discovery_intersections": {"fit": 0, "holdout": 0},
        "parameter_counts": models.parameter_counts(),
        "converged": _jsonable_convergence(models),
        "fit_traces": models.fit_trace_export(),
        "diagnostics": {
            "pit_max_deviation": diagnostics.pit_max_deviation(),
            "angular_second_max_deviation": second_deviation,
            "angular_fourth_max_deviation": fourth_deviation,
            "cross_band_energy_correlation": diagnostics.cross_band_energy_correlation().tolist(),
            "clipped_location_counts": dict(diagnostics.clipped_location_counts),
            "location_prediction_count": diagnostics.location_prediction_count,
            "finite_score_counts": dict(diagnostics.finite_score_counts),
            "score_count_per_arm": diagnostics.score_count_per_arm,
            "b4_e4_max_abs": max(b4_e4_error, diagnostics.b4_e4_max_abs),
            "p4_product_max_abs": diagnostics.p4_product_max_abs,
            "z4_component_marginal_max_abs": diagnostics.z4_component_marginal_max_abs,
            "radial_pit_tolerance": 0.02,
            "angular_tolerance": 0.03,
            "cross_band_energy_tolerance": 0.05,
        },
        "same_prior_check": _same_prior_check(models, seed),
        "roundtrip_max_error": roundtrip,
        "runtime_seconds": time.perf_counter() - started,
        "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "environment": {
            "python": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "hostname": platform.node(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
            "command": sys.argv,
        },
        "source_hashes": {
            "runner": sha256_file(Path(__file__)),
            "aggregate": sha256_file(Path(__file__).with_name("aggregate.py")),
            "observed_block": sha256_file(PROJECT_ROOT / "qalt" / "src" / "qalt" / "observed_block.py"),
            "rgb_block": sha256_file(PROJECT_ROOT / "qalt" / "src" / "qalt" / "rgb_block.py"),
            "statistics": sha256_file(PROJECT_ROOT / "qalt" / "src" / "qalt" / "observed_block_statistics.py"),
            "data_integrity": sha256_file(PROJECT_ROOT / "qalt" / "src" / "qalt" / "data_integrity.py"),
            "observed_routing": sha256_file(PROJECT_ROOT / "qalt" / "src" / "qalt" / "observed_routing.py"),
        },
    }
    if roundtrip >= 1e-6 or not all(np.all(np.isfinite(value)) for value in scores.values()):
        raise AssertionError("round-trip or finite-score hard check failed")

    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(
        output / "scores.npz",
        record_ids=holdout_ids,
        labels=holdout_labels,
        **{name: scores[name] for name in ARM_NAMES},
    )
    np.savez_compressed(output / "diagnostics.npz", **_diagnostic_arrays(diagnostics))
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n")
    (output / "config.json").write_text(
        json.dumps(
            {
                "seed": seed,
                "data_root": str(data_root),
                "source_commit": source_commit,
                "protocol_hash": PROTOCOL_HASH,
                "execution_protocol_hash": execution_protocol_hash,
                "max_iterations": max_iterations,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--max-iterations", type=int, default=DEFAULT_MAX_ITERATIONS)
    args = parser.parse_args()
    summary = run(
        args.seed,
        args.output,
        args.data_root,
        args.source_commit,
        max_iterations=args.max_iterations,
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"FAILED: {error}", file=sys.stderr, flush=True)
        raise
