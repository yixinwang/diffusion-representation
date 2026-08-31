"""Run the frozen seed-2100 observed B4 reversible radial child."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time
import traceback
from typing import Callable

import numpy as np
import scipy
from scipy import stats
from scipy.special import gammainc

from qalt.data_integrity import (
    adaptive_cifar_repair_split,
    cifar_training_batch_paths,
    load_cifar_training_batches,
    sha256_file,
    stable_json_hash,
)
from qalt.observed_block import (
    BlockConditionalModel,
    detail_to_blocks,
    fit_observed_b4_model,
    image_haar_inverse,
)
from qalt.observed_block_statistics import (
    holm_adjust,
    stratified_welch_summary,
    superiority_pvalue,
)
from qalt.observed_routing import coarse_energy_strata, image_haar, paired_dequantize
from qalt.radial_b4_flow import (
    SearchResult,
    cubic_from_base,
    cubic_log_ratio,
    cubic_to_base,
    fit_cubic_parameter,
    fit_student_parameter,
    gaussianize_gsm,
    invert_gaussianized_gsm,
    student_from_base,
    student_log_ratio,
    student_to_base,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA = Path("/ocean/datasets/community/cifar/cifar-10/cifar-10-batches-py")
PROTOCOL_PATH = PROJECT_ROOT / "qalt" / "theory" / "OBSERVED_B4_RADIAL_CHILD_PROTOCOL.md"
PROTOCOL_HASH = "b74b9e0187e4c68a2413fe4864797fec2b3ae1b35237785a00776f763bc1edbd"
REGISTERED_SEED = 2100
MAXIMUM_SITES = 250_000
MAX_ITERATIONS = 200
SITES_PER_IMAGE = 256
DETAILS_PER_IMAGE = 2_304
PIT_GRID = np.linspace(0.05, 0.95, 19)
ALPHA = 0.05
PRACTICAL_MARGIN = 0.01
ARMS = ("b4", "cubic", "student")
FLOAT64_ROUNDTRIP_TOLERANCE = 1e-10
HAAR_ROUNDTRIP_TOLERANCE = 1e-6
RADIAL_INVARIANCE_TOLERANCE = 1e-12
RUN_CONTEXT: dict[str, object] = {"phase": "not_started", "opened_files": []}
REGISTERED_SOURCE_PATHS = {
    "protocol": PROTOCOL_PATH,
    "runner": Path(__file__).resolve(),
    "radial_b4_flow": PROJECT_ROOT / "qalt" / "src" / "qalt" / "radial_b4_flow.py",
    "cubic_radial_flow": PROJECT_ROOT / "qalt" / "src" / "qalt" / "cubic_radial_flow.py",
    "observed_block": PROJECT_ROOT / "qalt" / "src" / "qalt" / "observed_block.py",
    "observed_block_statistics": PROJECT_ROOT / "qalt" / "src" / "qalt" / "observed_block_statistics.py",
    "observed_routing": PROJECT_ROOT / "qalt" / "src" / "qalt" / "observed_routing.py",
    "rgb_block": PROJECT_ROOT / "qalt" / "src" / "qalt" / "rgb_block.py",
    "data_integrity": PROJECT_ROOT / "qalt" / "src" / "qalt" / "data_integrity.py",
    "flow_tests": PROJECT_ROOT / "qalt" / "tests" / "test_radial_b4_flow.py",
    "runner_tests": PROJECT_ROOT / "qalt" / "tests" / "test_observed_b4_radial_child_runner.py",
    "b4_fitter_tests": PROJECT_ROOT / "qalt" / "tests" / "test_observed_block.py",
    "readme": PROJECT_ROOT / "qalt" / "experiments" / "observed_b4_radial_child" / "README.md",
    "slurm": PROJECT_ROOT / "qalt" / "experiments" / "observed_b4_radial_child" / "run_seed.slurm",
}


def _sha256_array(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


def _update_array_hash(digest: hashlib._Hash, array: np.ndarray) -> None:
    values = np.ascontiguousarray(np.asarray(array, dtype="<f8"))
    digest.update(str(values.shape).encode("ascii"))
    digest.update(values.view(np.uint8))


def _b4_parameter_hash(model: BlockConditionalModel) -> str:
    """Hash the complete fitted B4 numerical parameter sequence."""

    digest = hashlib.sha256()
    _update_array_hash(digest, model.boundaries)
    digest.update(repr(float(model.location.ridge)).encode("ascii"))
    digest.update(json.dumps(model.location.parent_masks).encode("ascii"))
    for coefficient in model.location.coefficients:
        _update_array_hash(digest, coefficient)
    for band in model.mixtures:
        for mixture in band:
            _update_array_hash(digest, mixture.weights)
            _update_array_hash(digest, mixture.scales)
            _update_array_hash(digest, mixture.shape)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _current_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
    ).strip()


def _source_hashes() -> dict[str, str]:
    return {name: sha256_file(path) for name, path in REGISTERED_SOURCE_PATHS.items()}


def _assert_registered_sources_committed() -> None:
    for path in REGISTERED_SOURCE_PATHS.values():
        relative = path.relative_to(PROJECT_ROOT).as_posix()
        try:
            subprocess.check_output(
                ["git", "ls-files", "--error-unmatch", relative],
                cwd=PROJECT_ROOT,
                stderr=subprocess.STDOUT,
                text=True,
            )
            committed = subprocess.check_output(
                ["git", "rev-parse", f"HEAD:{relative}"],
                cwd=PROJECT_ROOT,
                text=True,
            ).strip()
            working = subprocess.check_output(
                ["git", "hash-object", relative],
                cwd=PROJECT_ROOT,
                text=True,
            ).strip()
        except subprocess.CalledProcessError as error:
            raise ValueError(f"registered source is not committed: {relative}") from error
        if committed != working:
            raise ValueError(f"registered source differs from HEAD: {relative}")


def _run_focused_tests() -> dict[str, object]:
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "qalt/tests/test_radial_b4_flow.py",
        "qalt/tests/test_observed_b4_radial_child_runner.py",
        "qalt/tests/test_observed_block.py::test_b4_only_fitter_is_bitwise_identical_to_full_fitter",
    ]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(PROJECT_ROOT / "qalt" / "src")
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "passed": completed.returncode == 0,
    }


def _build_coefficients(
    images: np.ndarray,
    record_ids: np.ndarray,
    seed: int,
    progress: Callable[[str], None] = print,
) -> tuple[np.ndarray, np.ndarray, str]:
    coarse_parts = []
    block_parts = []
    digest = hashlib.sha256()
    for start in range(0, len(images), 512):
        stop = min(start + 512, len(images))
        values = paired_dequantize(images[start:stop], record_ids[start:stop], seed)
        digest.update(values.tobytes())
        coarse, detail = image_haar(values)
        coarse_parts.append(coarse)
        block_parts.append(detail_to_blocks(detail))
        progress(f"coefficients {stop}/{len(images)}")
    return np.concatenate(coarse_parts), np.concatenate(block_parts), digest.hexdigest()


def _b4_coordinates(
    model: BlockConditionalModel,
    coarse: np.ndarray,
    blocks: np.ndarray,
    sample: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    block_values = np.asarray(blocks, dtype=np.float64)
    if block_values.ndim != 5 or block_values.shape[-2:] != (3, 3):
        raise ValueError("blocks must have shape [image,row,column,band,color]")
    targets = block_values.reshape(-1, 3, 3)
    if sample is None:
        means = model.location.predict_flat(coarse, block_values).reshape(-1, 3, 3)
        strata, _ = coarse_energy_strata(coarse, model.boundaries)
    else:
        chosen = np.asarray(sample, dtype=np.int64)
        targets = targets[chosen]
        means = model.location.predict_flat(coarse, block_values, chosen).reshape(-1, 3, 3)
        strata, _ = coarse_energy_strata(coarse, model.boundaries, chosen)
    residual = targets - means
    gaussian = np.empty_like(residual)
    log_det = np.empty((len(residual), 3))
    b4_log_prob = np.empty((len(residual), 3))
    for band in range(3):
        for stratum, mixture in enumerate(model.mixtures[band]):
            selected = strata == stratum
            gaussian[selected, band], log_det[selected, band] = gaussianize_gsm(
                mixture, residual[selected, band]
            )
            b4_log_prob[selected, band] = mixture.log_prob(residual[selected, band])
    return (
        gaussian.reshape(len(gaussian), 9),
        np.sum(log_det, axis=1),
        np.sum(b4_log_prob, axis=1),
        residual,
        strata,
    )


def _b4_inverse_residual(
    model: BlockConditionalModel,
    gaussian: np.ndarray,
    strata: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(gaussian, dtype=np.float64).reshape(-1, 3, 3)
    labels = np.asarray(strata, dtype=np.int64)
    if labels.shape != (len(values),):
        raise ValueError("strata must align with Gaussianized rows")
    residual = np.empty_like(values)
    inverse_log_det = np.empty((len(values), 3))
    for band in range(3):
        for stratum, mixture in enumerate(model.mixtures[band]):
            selected = labels == stratum
            residual[selected, band], inverse_log_det[selected, band] = invert_gaussianized_gsm(
                mixture, values[selected, band]
            )
    return residual, np.sum(inverse_log_det, axis=1)


def _latent_image_statistics(latent: np.ndarray) -> dict[str, np.ndarray]:
    values = np.asarray(latent, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (SITES_PER_IMAGE, 9):
        raise ValueError("latents must have shape [image,256,9]")
    radius_squared = np.sum(values * values, axis=2)
    nonzero = radius_squared > 0.0
    if not np.all(nonzero):
        raise ValueError("registered continuous latents must have nonzero radius")
    pit = gammainc(4.5, 0.5 * radius_squared)
    direction = values / np.sqrt(radius_squared[:, :, None])
    direction_squared = direction * direction
    energy = np.sum(values.reshape(len(values), SITES_PER_IMAGE, 3, 3) ** 2, axis=3)
    shares = energy / radius_squared[:, :, None]
    return {
        "pit_fraction": np.mean(pit[:, :, None] <= PIT_GRID[None, None, :], axis=1),
        "direction_second": np.einsum("nsi,nsj->nij", direction, direction) / SITES_PER_IMAGE,
        "direction_fourth": np.einsum(
            "nsi,nsj->nij", direction_squared, direction_squared
        )
        / SITES_PER_IMAGE,
        "share_first": np.mean(shares, axis=1),
        "share_second": np.einsum("nsi,nsj->nij", shares, shares) / SITES_PER_IMAGE,
        "energy_sum": np.sum(energy, axis=1),
        "energy_outer_sum": np.einsum("nsi,nsj->nij", energy, energy),
        "count": np.full(len(values), SITES_PER_IMAGE, dtype=np.int64),
    }


def _balanced_matrix_summary(values: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sample = np.asarray(values, dtype=np.float64)
    classes = np.asarray(labels, dtype=np.int64)
    if sample.shape[0] != len(classes) or not np.array_equal(np.unique(classes), np.arange(10)):
        raise ValueError("matrix rows must align with CIFAR classes 0,...,9")
    flat = sample.reshape(len(sample), -1)
    means = []
    variance = np.zeros(flat.shape[1])
    for class_id in range(10):
        selected = flat[classes == class_id]
        if len(selected) != 500:
            raise ValueError("the repair screen requires 500 images per class")
        means.append(np.mean(selected, axis=0))
        variance += np.var(selected, axis=0, ddof=1) / len(selected) / 100.0
    return np.mean(means, axis=0).reshape(sample.shape[1:]), np.sqrt(variance).reshape(sample.shape[1:])


def _energy_correlation_summary(
    statistics: dict[str, np.ndarray], labels: np.ndarray
) -> dict[str, object]:
    counts = np.asarray(statistics["count"], dtype=np.float64)
    sums = np.asarray(statistics["energy_sum"], dtype=np.float64)
    outer = np.asarray(statistics["energy_outer_sum"], dtype=np.float64)
    total_count = float(np.sum(counts))
    mean = np.sum(sums, axis=0) / total_count
    second = np.sum(outer, axis=0) / total_count
    covariance = second - np.outer(mean, mean)
    standard = np.sqrt(np.diag(covariance))
    correlation = covariance / np.outer(standard, standard)
    standard_error = np.zeros((3, 3))
    upper_absolute = np.zeros((3, 3))
    critical = float(stats.norm.ppf(1.0 - ALPHA / (2.0 * 3.0)))
    for row in range(3):
        for column in range(row + 1, 3):
            centered_cross = (
                outer[:, row, column]
                - mean[row] * sums[:, column]
                - mean[column] * sums[:, row]
                + counts * mean[row] * mean[column]
            )
            centered_row = (
                outer[:, row, row]
                - 2.0 * mean[row] * sums[:, row]
                + counts * mean[row] ** 2
            )
            centered_column = (
                outer[:, column, column]
                - 2.0 * mean[column] * sums[:, column]
                + counts * mean[column] ** 2
            )
            rho = correlation[row, column]
            image_influence = (
                centered_cross / (standard[row] * standard[column])
                - 0.5
                * rho
                * (
                    centered_row / covariance[row, row]
                    + centered_column / covariance[column, column]
                )
            ) / counts
            _, error = _balanced_matrix_summary(image_influence[:, None], labels)
            standard_error[row, column] = standard_error[column, row] = float(error[0])
            limit = abs(rho) + critical * float(error[0])
            upper_absolute[row, column] = upper_absolute[column, row] = limit
    off = np.triu_indices(3, k=1)
    return {
        "estimate": correlation.tolist(),
        "cluster_standard_error": standard_error.tolist(),
        "simultaneous_upper_absolute": upper_absolute.tolist(),
        "maximum_absolute": float(np.max(np.abs(correlation[off]))),
        "maximum_simultaneous_upper_absolute": float(np.max(upper_absolute[off])),
    }


def _diagnostic_summary(
    statistics: dict[str, np.ndarray], labels: np.ndarray
) -> dict[str, object]:
    pit_mean, pit_error = _balanced_matrix_summary(statistics["pit_fraction"], labels)
    pit_deviation = np.abs(pit_mean - PIT_GRID)
    pit_critical = float(stats.norm.ppf(1.0 - ALPHA / (2.0 * len(PIT_GRID))))
    pit_upper = pit_deviation + pit_critical * pit_error

    second_mean, second_error = _balanced_matrix_summary(statistics["direction_second"], labels)
    fourth_mean, fourth_error = _balanced_matrix_summary(statistics["direction_fourth"], labels)
    second_target = np.eye(9) / 9.0
    fourth_target = np.full((9, 9), 1.0 / 99.0)
    np.fill_diagonal(fourth_target, 1.0 / 33.0)
    upper = np.triu_indices(9)
    angular_deviation = np.concatenate(
        [np.abs(second_mean - second_target)[upper], np.abs(fourth_mean - fourth_target)[upper]]
    )
    angular_error = np.concatenate([second_error[upper], fourth_error[upper]])
    angular_critical = float(stats.norm.ppf(1.0 - ALPHA / (2.0 * len(angular_deviation))))
    angular_upper = angular_deviation + angular_critical * angular_error

    share_mean, share_error = _balanced_matrix_summary(statistics["share_first"], labels)
    share_second_mean, share_second_error = _balanced_matrix_summary(
        statistics["share_second"], labels
    )
    share_second_target = np.full((3, 3), 1.0 / 11.0)
    np.fill_diagonal(share_second_target, 5.0 / 33.0)
    share_upper_indices = np.triu_indices(3)
    share_deviation = np.concatenate(
        [np.abs(share_mean - 1.0 / 3.0), np.abs(share_second_mean - share_second_target)[share_upper_indices]]
    )
    share_standard_error = np.concatenate(
        [share_error, share_second_error[share_upper_indices]]
    )
    share_critical = float(stats.norm.ppf(1.0 - ALPHA / (2.0 * len(share_deviation))))
    share_upper = share_deviation + share_critical * share_standard_error

    energy = _energy_correlation_summary(statistics, labels)
    return {
        "radial_pit": {
            "grid": PIT_GRID.tolist(),
            "estimate": pit_mean.tolist(),
            "cluster_standard_error": pit_error.tolist(),
            "family_critical_value": pit_critical,
            "simultaneous_upper": pit_upper.tolist(),
            "maximum_deviation": float(np.max(pit_deviation)),
            "maximum_simultaneous_upper": float(np.max(pit_upper)),
        },
        "angular": {
            "second": second_mean.tolist(),
            "second_cluster_standard_error": second_error.tolist(),
            "fourth": fourth_mean.tolist(),
            "fourth_cluster_standard_error": fourth_error.tolist(),
            "family_critical_value": angular_critical,
            "simultaneous_upper_flat": angular_upper.tolist(),
            "maximum_deviation": float(np.max(angular_deviation)),
            "maximum_simultaneous_upper": float(np.max(angular_upper)),
        },
        "band_energy_share": {
            "first": share_mean.tolist(),
            "first_cluster_standard_error": share_error.tolist(),
            "second": share_second_mean.tolist(),
            "second_cluster_standard_error": share_second_error.tolist(),
            "family_critical_value": share_critical,
            "simultaneous_upper_flat": share_upper.tolist(),
            "maximum_deviation": float(np.max(share_deviation)),
            "maximum_simultaneous_upper": float(np.max(share_upper)),
        },
        "latent_band_energy_correlation": energy,
    }


def _radial_invariance_summary(
    statistics: dict[str, dict[str, np.ndarray]],
) -> dict[str, object]:
    fields = ("direction_second", "direction_fourth", "share_first", "share_second")
    comparisons: dict[str, dict[str, float]] = {}
    maximum = 0.0
    for arm in ("cubic", "student"):
        differences = {
            field: float(
                np.max(np.abs(statistics[arm][field] - statistics["b4"][field]))
            )
            for field in fields
        }
        comparisons[f"b4_vs_{arm}"] = differences
        maximum = max(maximum, *differences.values())
    return {
        "comparisons": comparisons,
        "maximum_absolute_difference": maximum,
        "tolerance": RADIAL_INVARIANCE_TOLERANCE,
        "passed": maximum <= RADIAL_INVARIANCE_TOLERANCE,
    }


def _sitewise_radial_invariance(
    reference: np.ndarray,
    candidate: np.ndarray,
) -> dict[str, float]:
    left = np.asarray(reference, dtype=np.float64).reshape(-1, 9)
    right = np.asarray(candidate, dtype=np.float64).reshape(-1, 9)
    if left.shape != right.shape:
        raise ValueError("radial invariance arrays must have the same shape")
    left_radius_squared = np.sum(left * left, axis=1)
    right_radius_squared = np.sum(right * right, axis=1)
    if np.any(left_radius_squared <= 0.0) or np.any(right_radius_squared <= 0.0):
        raise ValueError("registered radial invariance rows must have nonzero radius")
    left_direction = left / np.sqrt(left_radius_squared[:, None])
    right_direction = right / np.sqrt(right_radius_squared[:, None])
    left_energy = np.sum(left.reshape(-1, 3, 3) ** 2, axis=2)
    right_energy = np.sum(right.reshape(-1, 3, 3) ** 2, axis=2)
    left_share = left_energy / left_radius_squared[:, None]
    right_share = right_energy / right_radius_squared[:, None]
    return {
        "direction_max_absolute_difference": float(
            np.max(np.abs(left_direction - right_direction))
        ),
        "band_share_max_absolute_difference": float(
            np.max(np.abs(left_share - right_share))
        ),
    }


def _json_safe_welch_summary(summary: dict[str, float]) -> dict[str, object]:
    output: dict[str, object] = dict(summary)
    degrees = float(summary["degrees_of_freedom"])
    output["degrees_of_freedom_is_infinite"] = not math.isfinite(degrees)
    output["degrees_of_freedom"] = degrees if math.isfinite(degrees) else None
    return output


def _quality_summary(
    scores: dict[str, np.ndarray], labels: np.ndarray
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    contrasts = {
        "b4_minus_cubic": (scores["cubic"] - scores["b4"]) / DETAILS_PER_IMAGE,
        "student_minus_cubic": (scores["cubic"] - scores["student"]) / DETAILS_PER_IMAGE,
        "b4_minus_student": (scores["student"] - scores["b4"]) / DETAILS_PER_IMAGE,
    }
    p_b4, b4_summary = superiority_pvalue(
        contrasts["b4_minus_cubic"], labels, PRACTICAL_MARGIN
    )
    p_student, student_summary = superiority_pvalue(
        contrasts["student_minus_cubic"], labels, -PRACTICAL_MARGIN
    )
    adjusted = holm_adjust(
        {"b4_minus_cubic": p_b4, "student_minus_cubic": p_student},
        expected_count=2,
    )
    descriptive = stratified_welch_summary(contrasts["b4_minus_student"], labels)
    summary = {
        "b4_minus_cubic": {
            **_json_safe_welch_summary(b4_summary),
            "margin": PRACTICAL_MARGIN,
            "raw_pvalue": p_b4,
            "holm_adjusted_pvalue": adjusted["b4_minus_cubic"],
            "passed": adjusted["b4_minus_cubic"] < ALPHA,
        },
        "student_minus_cubic": {
            **_json_safe_welch_summary(student_summary),
            "margin": -PRACTICAL_MARGIN,
            "raw_pvalue": p_student,
            "holm_adjusted_pvalue": adjusted["student_minus_cubic"],
            "passed": adjusted["student_minus_cubic"] < ALPHA,
        },
        "b4_minus_student_descriptive": _json_safe_welch_summary(descriptive),
    }
    summary["all_primary_pass"] = bool(
        summary["b4_minus_cubic"]["passed"]
        and summary["student_minus_cubic"]["passed"]
    )
    return summary, contrasts


def _search_export(result: SearchResult) -> dict[str, object]:
    return {
        "parameter": result.parameter,
        "objective": result.objective,
        "evaluations": result.evaluations,
        "grid_parameters": list(result.grid_parameters),
        "grid_objectives": list(result.grid_objectives),
        "refined_candidates": [list(value) for value in result.refined_candidates],
    }


def _time_callable(function: Callable[[], float], warmups: int = 2, repetitions: int = 9) -> dict[str, object]:
    for _ in range(warmups):
        value = function()
        if not math.isfinite(float(value)):
            raise FloatingPointError("benchmark warmup produced a nonfinite checksum")
    elapsed = []
    checksums = []
    for _ in range(repetitions):
        started = time.perf_counter()
        value = function()
        elapsed.append(time.perf_counter() - started)
        checksums.append(float(value))
    return {
        "warmups": warmups,
        "repetitions": repetitions,
        "seconds": elapsed,
        "median_seconds": float(np.median(elapsed)),
        "q25_seconds": float(np.quantile(elapsed, 0.25)),
        "q75_seconds": float(np.quantile(elapsed, 0.75)),
        "checksum_range": [min(checksums), max(checksums)],
    }


def _benchmark(
    model: BlockConditionalModel,
    residual: np.ndarray,
    gaussian: np.ndarray,
    strata: np.ndarray,
    cubic_a: float,
    student_tau: float,
) -> dict[str, object]:
    rows = np.asarray(residual)[:10_000]
    repaired = np.asarray(gaussian)[:10_000]
    labels = np.asarray(strata)[:10_000]
    peak_before = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

    def b4_full_score() -> float:
        return float(np.sum(_coordinates_from_residual(model, rows, labels)[2]))

    def cubic_full_score() -> float:
        encoded, _, b4_log_prob, _, _ = _coordinates_from_residual(model, rows, labels)
        return float(np.sum(b4_log_prob + cubic_log_ratio(encoded, cubic_a)))

    def student_full_score() -> float:
        encoded, _, b4_log_prob, _, _ = _coordinates_from_residual(model, rows, labels)
        return float(np.sum(b4_log_prob + student_log_ratio(encoded, student_tau)))

    def b4_roundtrip() -> float:
        encoded, _, _, _, _ = _coordinates_from_residual(model, rows, labels)
        decoded, _ = _b4_inverse_residual(model, encoded, labels)
        return float(np.sum(decoded[::100]))

    def cubic_full_roundtrip() -> float:
        encoded, _, _, _, _ = _coordinates_from_residual(model, rows, labels)
        base, _ = cubic_to_base(encoded, cubic_a)
        restored, _ = cubic_from_base(base, cubic_a)
        decoded, _ = _b4_inverse_residual(model, restored, labels)
        return float(np.sum(decoded[::100]))

    def student_full_roundtrip() -> float:
        encoded, _, _, _, _ = _coordinates_from_residual(model, rows, labels)
        base, _ = student_to_base(encoded, student_tau)
        restored, _ = student_from_base(base, student_tau)
        decoded, _ = _b4_inverse_residual(model, restored, labels)
        return float(np.sum(decoded[::100]))

    timings = {
        "b4_full_score": _time_callable(b4_full_score),
        "cubic_full_score": _time_callable(cubic_full_score),
        "student_full_score": _time_callable(student_full_score),
        "b4_repaired_roundtrip": _time_callable(b4_roundtrip),
        "cubic_full_roundtrip": _time_callable(cubic_full_roundtrip),
        "student_full_roundtrip": _time_callable(student_full_roundtrip),
        "cubic_radial_roundtrip": _time_callable(
            lambda: float(np.sum(cubic_from_base(cubic_to_base(repaired, cubic_a)[0], cubic_a)[0][::100]))
        ),
        "student_radial_roundtrip": _time_callable(
            lambda: float(
                np.sum(student_from_base(student_to_base(repaired, student_tau)[0], student_tau)[0][::100])
            )
        ),
        "cubic_score": _time_callable(lambda: float(np.sum(cubic_log_ratio(repaired, cubic_a)))),
        "student_score": _time_callable(lambda: float(np.sum(student_log_ratio(repaired, student_tau)))),
    }
    for measurement in timings.values():
        measurement["vectors_per_second"] = len(rows) / measurement["median_seconds"]
    b4_time = timings["b4_repaired_roundtrip"]["median_seconds"]
    cubic_time = timings["cubic_full_roundtrip"]["median_seconds"]
    cubic_radial_time = timings["cubic_radial_roundtrip"]["median_seconds"]
    student_radial_time = timings["student_radial_roundtrip"]["median_seconds"]
    ratios = {
        "cubic_full_over_b4_repaired": cubic_time / b4_time,
        "cubic_radial_over_student_radial": cubic_radial_time / student_radial_time,
    }
    return {
        "vector_count": len(rows),
        "timings": timings,
        "peak_rss_kib_before": peak_before,
        "peak_rss_kib_after": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "ratios": ratios,
        "full_overhead_pass": ratios["cubic_full_over_b4_repaired"] <= 1.10,
        "radial_speed_pass": ratios["cubic_radial_over_student_radial"] <= 0.50,
        "all_checks_pass": bool(
            ratios["cubic_full_over_b4_repaired"] <= 1.10
            and ratios["cubic_radial_over_student_radial"] <= 0.50
        ),
    }


def _coordinates_from_residual(
    model: BlockConditionalModel,
    residual: np.ndarray,
    strata: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows = np.asarray(residual, dtype=np.float64).reshape(-1, 3, 3)
    labels = np.asarray(strata, dtype=np.int64)
    gaussian = np.empty_like(rows)
    log_det = np.empty((len(rows), 3))
    b4_log_prob = np.empty((len(rows), 3))
    for band in range(3):
        for stratum, mixture in enumerate(model.mixtures[band]):
            selected = labels == stratum
            gaussian[selected, band], log_det[selected, band] = gaussianize_gsm(
                mixture, rows[selected, band]
            )
            b4_log_prob[selected, band] = mixture.log_prob(rows[selected, band])
    return gaussian.reshape(len(rows), 9), np.sum(log_det, axis=1), np.sum(
        b4_log_prob, axis=1
    ), rows, labels


def run(
    seed: int,
    output: Path,
    data_root: Path,
    expected_source_commit: str,
) -> dict[str, object]:
    global RUN_CONTEXT
    started = time.perf_counter()
    RUN_CONTEXT = {"phase": "argument_validation", "opened_files": []}
    if seed != REGISTERED_SEED:
        raise ValueError(f"this screen requires seed {REGISTERED_SEED}")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    if os.environ.get("OMP_NUM_THREADS") != "1":
        raise ValueError("this screen requires OMP_NUM_THREADS=1")
    actual_protocol_hash = sha256_file(PROTOCOL_PATH)
    if actual_protocol_hash != PROTOCOL_HASH:
        raise ValueError("frozen observed radial-child protocol hash mismatch")
    _assert_registered_sources_committed()
    source_commit = _current_commit()
    if expected_source_commit != source_commit:
        raise ValueError(
            f"submitted source commit {expected_source_commit!r} does not equal HEAD {source_commit}"
        )
    source_hashes = _source_hashes()
    focused_tests = _run_focused_tests()
    if not focused_tests["passed"]:
        raise RuntimeError("focused radial-child tests failed")

    RUN_CONTEXT["phase"] = "data_load"
    opened_files: list[str] = []
    RUN_CONTEXT["opened_files"] = opened_files
    print("load five allowlisted CIFAR training batches", flush=True)
    images, labels = load_cifar_training_batches(data_root, opened_files=opened_files)
    expected_opened_files = [
        str(path.resolve()) for path in cifar_training_batch_paths(data_root)
    ]
    data_integrity_pass = opened_files == expected_opened_files
    if not data_integrity_pass:
        raise AssertionError("opened-file ledger differs from the exclusive training allowlist")
    split = adaptive_cifar_repair_split(labels)
    fit_ids = np.asarray(split["fit"], dtype=np.int64)
    holdout_ids = np.asarray(split["repair_holdout"], dtype=np.int64)
    discovery_ids = np.asarray(split["excluded_discovery"], dtype=np.int64)
    fit_images = images[fit_ids].copy()
    holdout_images = images[holdout_ids].copy()
    holdout_labels = labels[holdout_ids].copy()
    del images, labels
    if np.intersect1d(discovery_ids, fit_ids).size or np.intersect1d(discovery_ids, holdout_ids).size:
        raise AssertionError("excluded discovery records entered downstream arrays")
    official_test_deserialized = any(
        Path(path).name == "test_batch" for path in opened_files
    )
    if official_test_deserialized:
        raise AssertionError("official CIFAR test_batch entered the opened-file ledger")

    print("build fitting coefficients", flush=True)
    fit_coarse, fit_blocks, fit_dequant_hash = _build_coefficients(
        fit_images, fit_ids, seed, lambda message: print(f"fit {message}", flush=True)
    )
    print("build repair-holdout coefficients", flush=True)
    holdout_coarse, holdout_blocks, holdout_dequant_hash = _build_coefficients(
        holdout_images,
        holdout_ids,
        seed,
        lambda message: print(f"holdout {message}", flush=True),
    )
    if holdout_blocks.shape[1:3] != (16, 16):
        raise AssertionError("registered CIFAR detail grid must be 16x16")

    RUN_CONTEXT["phase"] = "b4_fit"
    print("fit frozen B4 parent only", flush=True)
    models = fit_observed_b4_model(
        fit_coarse,
        fit_blocks,
        rng=np.random.default_rng(seed),
        maximum_sites=MAXIMUM_SITES,
        max_iterations=MAX_ITERATIONS,
        progress=lambda message: print(f"model {message}", flush=True),
    )
    if len(models.sample) != MAXIMUM_SITES:
        raise AssertionError(
            f"registered common site sample has {len(models.sample)} rather than {MAXIMUM_SITES} rows"
        )
    if not all(all(cell.converged for cell in band) for band in models.b4.fit_diagnostics):
        raise AssertionError("B4 parent did not converge in every cell")
    if any(models.b4.location.parent_masks):
        raise AssertionError("registered B4 inverse requires a coarse-only location")
    b4_parameter_hash = _b4_parameter_hash(models.b4)
    b4_parameter_count = models.b4.parameter_count()
    b4_fit_trace = models.b4.fit_trace_export()

    print("Gaussianize common fitting sites", flush=True)
    fit_gaussian, fit_log_det, fit_b4_log_prob, fit_residual, fit_strata = _b4_coordinates(
        models.b4, fit_coarse, fit_blocks, models.sample
    )
    standard_fit = -0.5 * (
        9 * math.log(2.0 * math.pi) + np.sum(fit_gaussian * fit_gaussian, axis=1)
    )
    fit_density_parity = float(np.max(np.abs(fit_b4_log_prob - standard_fit - fit_log_det)))
    probe_count = min(4_096, len(fit_gaussian))
    recovered_residual, inverse_log_det = _b4_inverse_residual(
        models.b4, fit_gaussian[:probe_count], fit_strata[:probe_count]
    )
    gaussianizer_roundtrip = float(
        np.max(np.abs(recovered_residual - fit_residual[:probe_count]))
    )
    gaussianizer_log_det = float(
        np.max(np.abs(inverse_log_det + fit_log_det[:probe_count]))
    )
    parity_pass = bool(
        fit_density_parity <= 1e-10
        and gaussianizer_roundtrip <= 1e-10
        and gaussianizer_log_det <= 1e-10
    )
    if not parity_pass:
        raise AssertionError("B4 reversible-coordinate parity failed")

    print("fit global cubic and Student radial parameters", flush=True)
    radial_fit_started = time.perf_counter()
    cubic_fit = fit_cubic_parameter(fit_gaussian)
    student_fit = fit_student_parameter(fit_gaussian)
    radial_fit_seconds = time.perf_counter() - radial_fit_started
    cubic_a = cubic_fit.parameter
    student_tau = student_fit.parameter
    cubic_boundary_pass = 1e-8 < cubic_a < 0.1 - 1e-8
    student_boundary_pass = not (
        student_tau > 0.0 and abs(1.0 / student_tau - 2.1) <= 1e-8
    )

    scores = {name: np.empty(len(holdout_ids)) for name in ARMS}
    diagnostics_parts: dict[str, dict[str, list[np.ndarray]]] = {
        arm: defaultdict(list) for arm in ARMS
    }
    benchmark_probe: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
    a0_tie = 0.0
    holdout_density_parity = 0.0
    reversibility_maxima: defaultdict[str, float] = defaultdict(float)
    sitewise_invariance_maxima = {
        arm: {
            "direction_max_absolute_difference": 0.0,
            "band_share_max_absolute_difference": 0.0,
        }
        for arm in ("cubic", "student")
    }
    stratum_mismatch_count = 0
    RUN_CONTEXT["phase"] = "repair_holdout_evaluation"
    print("score and diagnose repair holdout", flush=True)
    for start in range(0, len(holdout_ids), 128):
        stop = min(start + 128, len(holdout_ids))
        gaussian, b4_forward_log_det, b4_site, residual, strata = _b4_coordinates(
            models.b4,
            holdout_coarse[start:stop],
            holdout_blocks[start:stop],
        )
        standard = -0.5 * (
            9 * math.log(2.0 * math.pi) + np.sum(gaussian * gaussian, axis=1)
        )
        holdout_density_parity = max(
            holdout_density_parity,
            float(np.max(np.abs(b4_site - standard - b4_forward_log_det))),
        )
        cubic_ratio = cubic_log_ratio(gaussian, cubic_a)
        student_ratio = student_log_ratio(gaussian, student_tau)
        a0_tie = max(a0_tie, float(np.max(np.abs(cubic_log_ratio(gaussian, 0.0)))))
        image_count = stop - start
        b4_image = np.sum(b4_site.reshape(image_count, SITES_PER_IMAGE), axis=1)
        scores["b4"][start:stop] = b4_image
        scores["cubic"][start:stop] = b4_image + np.sum(
            cubic_ratio.reshape(image_count, SITES_PER_IMAGE), axis=1
        )
        scores["student"][start:stop] = b4_image + np.sum(
            student_ratio.reshape(image_count, SITES_PER_IMAGE), axis=1
        )
        cubic_base, cubic_to_log_det = cubic_to_base(gaussian, cubic_a)
        cubic_restored, cubic_from_log_det = cubic_from_base(cubic_base, cubic_a)
        student_base, student_to_log_det = student_to_base(gaussian, student_tau)
        student_restored, student_from_log_det = student_from_base(
            student_base, student_tau
        )
        for arm, candidate in (("cubic", cubic_base), ("student", student_base)):
            differences = _sitewise_radial_invariance(gaussian, candidate)
            for name, value in differences.items():
                sitewise_invariance_maxima[arm][name] = max(
                    sitewise_invariance_maxima[arm][name], value
                )
        reversibility_maxima["cubic_latent_roundtrip_max_error"] = max(
            reversibility_maxima["cubic_latent_roundtrip_max_error"],
            float(np.max(np.abs(cubic_restored - gaussian))),
        )
        reversibility_maxima["cubic_radial_log_det_cancellation_max_error"] = max(
            reversibility_maxima["cubic_radial_log_det_cancellation_max_error"],
            float(np.max(np.abs(cubic_to_log_det + cubic_from_log_det))),
        )
        reversibility_maxima["student_latent_roundtrip_max_error"] = max(
            reversibility_maxima["student_latent_roundtrip_max_error"],
            float(np.max(np.abs(student_restored - gaussian))),
        )
        reversibility_maxima["student_radial_log_det_cancellation_max_error"] = max(
            reversibility_maxima["student_radial_log_det_cancellation_max_error"],
            float(np.max(np.abs(student_to_log_det + student_from_log_det))),
        )

        original_blocks = holdout_blocks[start:stop]
        original_flat = original_blocks.reshape(-1, 3, 3)
        forward_location = original_flat - residual
        inverse_location = models.b4.location.predict_flat(
            holdout_coarse[start:stop], np.zeros_like(original_blocks)
        ).reshape(-1, 3, 3)
        reference_pixels = paired_dequantize(
            holdout_images[start:stop], holdout_ids[start:stop], seed
        )
        strata_after, _ = coarse_energy_strata(
            holdout_coarse[start:stop], models.b4.boundaries
        )
        stratum_mismatch_count += int(np.count_nonzero(strata_after != strata))
        reversibility_maxima["location_subtract_add_max_error"] = max(
            reversibility_maxima["location_subtract_add_max_error"],
            float(np.max(np.abs(inverse_location + residual - original_flat))),
        )
        reversibility_maxima["independent_location_recompute_max_error"] = max(
            reversibility_maxima["independent_location_recompute_max_error"],
            float(np.max(np.abs(inverse_location - forward_location))),
        )
        for arm, restored, radial_to_log_det, radial_from_log_det in (
            ("cubic", cubic_restored, cubic_to_log_det, cubic_from_log_det),
            ("student", student_restored, student_to_log_det, student_from_log_det),
        ):
            recovered_residual, b4_inverse_log_det = _b4_inverse_residual(
                models.b4, restored, strata_after
            )
            recovered_blocks = (inverse_location + recovered_residual).reshape(
                original_blocks.shape
            )
            reconstructed_pixels = image_haar_inverse(
                holdout_coarse[start:stop], recovered_blocks
            )
            reversibility_maxima[f"{arm}_residual_roundtrip_max_error"] = max(
                reversibility_maxima[f"{arm}_residual_roundtrip_max_error"],
                float(np.max(np.abs(recovered_residual - residual))),
            )
            reversibility_maxima[f"{arm}_full_log_det_cancellation_max_error"] = max(
                reversibility_maxima[f"{arm}_full_log_det_cancellation_max_error"],
                float(
                    np.max(
                        np.abs(
                            b4_forward_log_det
                            + radial_to_log_det
                            + radial_from_log_det
                            + b4_inverse_log_det
                        )
                    )
                ),
            )
            reversibility_maxima[f"{arm}_conditional_block_roundtrip_max_error"] = max(
                reversibility_maxima[f"{arm}_conditional_block_roundtrip_max_error"],
                float(np.max(np.abs(recovered_blocks - original_blocks))),
            )
            reversibility_maxima[f"{arm}_haar_roundtrip_max_error"] = max(
                reversibility_maxima[f"{arm}_haar_roundtrip_max_error"],
                float(np.max(np.abs(reconstructed_pixels - reference_pixels))),
            )
        latent = {
            "b4": gaussian,
            "cubic": cubic_base,
            "student": student_base,
        }
        for arm, values in latent.items():
            part = _latent_image_statistics(
                values.reshape(image_count, SITES_PER_IMAGE, 9)
            )
            for name, array in part.items():
                diagnostics_parts[arm][name].append(array)
        if benchmark_probe is None:
            benchmark_probe = (residual.copy(), gaussian.copy(), strata.copy())

    if any(not np.all(np.isfinite(values)) for values in scores.values()):
        raise AssertionError("nonfinite repair-holdout score")
    diagnostics_arrays = {
        arm: {name: np.concatenate(parts) for name, parts in arm_parts.items()}
        for arm, arm_parts in diagnostics_parts.items()
    }
    diagnostic_summaries = {
        arm: _diagnostic_summary(values, holdout_labels)
        for arm, values in diagnostics_arrays.items()
    }
    radial_invariance = _radial_invariance_summary(diagnostics_arrays)
    sitewise_invariance_maximum = max(
        value
        for arm_values in sitewise_invariance_maxima.values()
        for value in arm_values.values()
    )
    radial_invariance["aggregate_moment_passed"] = radial_invariance["passed"]
    radial_invariance["sitewise"] = sitewise_invariance_maxima
    radial_invariance["sitewise_maximum_absolute_difference"] = (
        sitewise_invariance_maximum
    )
    radial_invariance["passed"] = bool(
        radial_invariance["aggregate_moment_passed"]
        and sitewise_invariance_maximum <= RADIAL_INVARIANCE_TOLERANCE
    )
    quality, contrasts = _quality_summary(scores, holdout_labels)
    cubic_diagnostic = diagnostic_summaries["cubic"]
    diagnostic_passes = {
        "radial_pit": bool(
            cubic_diagnostic["radial_pit"]["maximum_deviation"] <= 0.02
            and cubic_diagnostic["radial_pit"]["maximum_simultaneous_upper"] <= 0.03
        ),
        "angular": bool(
            cubic_diagnostic["angular"]["maximum_deviation"] <= 0.03
            and cubic_diagnostic["angular"]["maximum_simultaneous_upper"] <= 0.04
        ),
        "band_energy_share": bool(
            cubic_diagnostic["band_energy_share"]["maximum_deviation"] <= 0.03
            and cubic_diagnostic["band_energy_share"]["maximum_simultaneous_upper"] <= 0.04
        ),
        "latent_band_energy_correlation": bool(
            cubic_diagnostic["latent_band_energy_correlation"]["maximum_absolute"] <= 0.05
            and cubic_diagnostic["latent_band_energy_correlation"][
                "maximum_simultaneous_upper_absolute"
            ]
            <= 0.07
        ),
        "radial_invariance": bool(radial_invariance["passed"]),
    }

    probe = paired_dequantize(fit_images[:4], fit_ids[:4], seed)
    probe_coarse, probe_detail = image_haar(probe)
    reconstructed = image_haar_inverse(probe_coarse, detail_to_blocks(probe_detail))
    haar_roundtrip = float(np.max(np.abs(reconstructed - probe)))
    if benchmark_probe is None:
        raise AssertionError("benchmark probe was not created")
    benchmark = _benchmark(
        models.b4,
        benchmark_probe[0],
        benchmark_probe[1],
        benchmark_probe[2],
        cubic_a,
        student_tau,
    )
    float64_reversibility_values = [
        value
        for name, value in reversibility_maxima.items()
        if not name.endswith("haar_roundtrip_max_error")
    ]
    reversibility_pass = bool(
        a0_tie <= FLOAT64_ROUNDTRIP_TOLERANCE
        and gaussianizer_roundtrip <= FLOAT64_ROUNDTRIP_TOLERANCE
        and gaussianizer_log_det <= FLOAT64_ROUNDTRIP_TOLERANCE
        and holdout_density_parity <= FLOAT64_ROUNDTRIP_TOLERANCE
        and max(float64_reversibility_values, default=0.0)
        <= FLOAT64_ROUNDTRIP_TOLERANCE
        and stratum_mismatch_count == 0
        and haar_roundtrip <= HAAR_ROUNDTRIP_TOLERANCE
        and reversibility_maxima["cubic_haar_roundtrip_max_error"]
        <= HAAR_ROUNDTRIP_TOLERANCE
        and reversibility_maxima["student_haar_roundtrip_max_error"]
        <= HAAR_ROUNDTRIP_TOLERANCE
    )

    ordered_layers = [
        ("source_data_integrity", data_integrity_pass),
        ("b4_coordinate_parity", parity_pass),
        ("fit_boundary", cubic_boundary_pass and student_boundary_pass),
        ("normalized_density_reversibility", reversibility_pass),
        ("paired_nll", bool(quality["all_primary_pass"])),
        ("radial_calibration", diagnostic_passes["radial_pit"]),
        (
            "direction_share_falsifiers",
            diagnostic_passes["angular"]
            and diagnostic_passes["band_energy_share"]
            and diagnostic_passes["radial_invariance"],
        ),
        ("latent_energy_dependence", diagnostic_passes["latent_band_energy_correlation"]),
        ("local_cpu", bool(benchmark["all_checks_pass"])),
    ]
    first_failed_layer = next((name for name, passed in ordered_layers if not passed), None)
    all_checks_pass = first_failed_layer is None

    summary: dict[str, object] = {
        "schema_version": "observed_b4_radial_child_seed_v2",
        "status": "adaptive_repair_screen_no_confirmation_coverage",
        "seed": seed,
        "source_commit": source_commit,
        "protocol_hash": actual_protocol_hash,
        "source_hashes": source_hashes,
        "focused_tests": focused_tests,
        "split_hash": stable_json_hash(split),
        "fit_size": int(len(fit_ids)),
        "holdout_size": int(len(holdout_ids)),
        "excluded_discovery_size": int(len(discovery_ids)),
        "fit_input_hash": _sha256_array(fit_images),
        "holdout_input_hash": _sha256_array(holdout_images),
        "fit_record_ids_hash": _sha256_array(fit_ids.astype("<i8")),
        "holdout_record_ids_hash": _sha256_array(holdout_ids.astype("<i8")),
        "holdout_labels_hash": _sha256_array(holdout_labels.astype("<i8")),
        "fit_dequant_hash": fit_dequant_hash,
        "holdout_dequant_hash": holdout_dequant_hash,
        "site_sample_hash": models.sample_hash,
        "site_sample_count": int(len(models.sample)),
        "opened_files": opened_files,
        "expected_opened_files": expected_opened_files,
        "opened_file_ledger_exact_match": data_integrity_pass,
        "official_test_deserialized": official_test_deserialized,
        "downstream_discovery_intersections": {"fit": 0, "holdout": 0},
        "b4_parameter_hash": b4_parameter_hash,
        "b4_parameter_count": b4_parameter_count,
        "b4_fit_trace": b4_fit_trace,
        "b4_converged": [[cell.converged for cell in band] for band in models.b4.fit_diagnostics],
        "b4_coordinate_parity": {
            "fit_density_max_error": fit_density_parity,
            "roundtrip_max_error": gaussianizer_roundtrip,
            "log_det_max_error": gaussianizer_log_det,
            "passed": parity_pass,
        },
        "radial_fit": {
            "cubic": _search_export(cubic_fit),
            "student": _search_export(student_fit),
            "student_degrees_of_freedom": (
                None if student_tau == 0.0 else 1.0 / student_tau
            ),
            "student_is_gaussian_endpoint": student_tau == 0.0,
            "seconds": radial_fit_seconds,
            "cubic_boundary_pass": cubic_boundary_pass,
            "student_boundary_pass": student_boundary_pass,
        },
        "quality": quality,
        "diagnostics": diagnostic_summaries,
        "radial_invariance": radial_invariance,
        "diagnostic_passes": diagnostic_passes,
        "reversibility": {
            "a0_b4_score_tie_max_error": a0_tie,
            "fit_b4_haar_roundtrip_max_error": haar_roundtrip,
            "holdout_b4_density_parity_max_error": holdout_density_parity,
            "stratum_mismatch_count": stratum_mismatch_count,
            "float64_tolerance": FLOAT64_ROUNDTRIP_TOLERANCE,
            "haar_float32_tolerance": HAAR_ROUNDTRIP_TOLERANCE,
            "maxima": dict(sorted(reversibility_maxima.items())),
            "passed": reversibility_pass,
        },
        "benchmark": benchmark,
        "ordered_layer_decisions": [
            {"layer": name, "passed": bool(passed)} for name, passed in ordered_layers
        ],
        "first_failed_layer": first_failed_layer,
        "all_checks_pass": all_checks_pass,
        "runtime_seconds": time.perf_counter() - started,
        "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "environment": {
            "python": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "hostname": platform.node(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
            "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
            "command": [sys.executable, *sys.argv],
        },
    }

    RUN_CONTEXT["phase"] = "result_write"
    output.mkdir(parents=True, exist_ok=False)
    _fsync_directory(output.parent)
    np.savez_compressed(
        output / "scores.npz",
        record_ids=holdout_ids,
        labels=holdout_labels,
        **scores,
        **{f"contrast_{name}": values for name, values in contrasts.items()},
    )
    diagnostic_output = {}
    for arm, values in diagnostics_arrays.items():
        for name, array in values.items():
            diagnostic_output[f"{arm}_{name}"] = array
    np.savez_compressed(output / "diagnostics.npz", **diagnostic_output)
    _fsync_file(output / "scores.npz")
    _fsync_file(output / "diagnostics.npz")
    _write_json(output / "summary.json", summary)
    _write_json(
        output / "config.json",
        {
            "seed": seed,
            "data_root": str(data_root),
            "source_commit": source_commit,
            "protocol_hash": actual_protocol_hash,
            "maximum_sites": MAXIMUM_SITES,
            "max_iterations": MAX_ITERATIONS,
            "float64_roundtrip_tolerance": FLOAT64_ROUNDTRIP_TOLERANCE,
            "haar_roundtrip_tolerance": HAAR_ROUNDTRIP_TOLERANCE,
            "radial_invariance_tolerance": RADIAL_INVARIANCE_TOLERANCE,
        },
    )
    payload_names = ("config.json", "diagnostics.npz", "scores.npz", "summary.json")
    payload_hashes = {
        name: sha256_file(output / name) for name in payload_names
    }
    checksum_text = "".join(
        f"{digest}  {name}\n" for name, digest in sorted(payload_hashes.items())
    )
    checksum_path = output / "SHA256SUMS"
    checksum_path.write_text(checksum_text, encoding="utf-8")
    _fsync_file(checksum_path)
    _fsync_directory(output)
    _write_json(
        output / "COMPLETE.json",
        {
            "status": "complete",
            "source_commit": source_commit,
            "protocol_hash": actual_protocol_hash,
            "payload_hashes": payload_hashes,
            "sha256sums_hash": sha256_file(checksum_path),
            "all_checks_pass": all_checks_pass,
            "first_failed_layer": first_failed_layer,
        },
    )
    _fsync_directory(output)
    RUN_CONTEXT["phase"] = "complete"
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--source-commit", required=True)
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    output_preexisting = arguments.output.exists()
    try:
        summary = run(
            arguments.seed,
            arguments.output,
            arguments.data_root,
            arguments.source_commit,
        )
    except Exception as error:
        if not output_preexisting:
            arguments.output.mkdir(parents=True, exist_ok=True)
            try:
                current_commit = _current_commit()
            except Exception:
                current_commit = None
            try:
                protocol_hash = sha256_file(PROTOCOL_PATH)
            except Exception:
                protocol_hash = None
            try:
                source_hashes = _source_hashes()
            except Exception:
                source_hashes = None
            _write_json(
                arguments.output / "failure.json",
                {
                    "status": "failed_before_complete_result",
                    "phase": RUN_CONTEXT.get("phase"),
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                    "submitted_source_commit": arguments.source_commit,
                    "current_source_commit": current_commit,
                    "protocol_hash": protocol_hash,
                    "source_hashes": source_hashes,
                    "opened_files": list(RUN_CONTEXT.get("opened_files", [])),
                    "environment": {
                        "hostname": platform.node(),
                        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
                        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                        "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
                        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
                    },
                    "command": [sys.executable, *sys.argv],
                },
            )
            _fsync_directory(arguments.output)
            _fsync_directory(arguments.output.parent)
        raise
    print(
        json.dumps(
            {
                "all_checks_pass": summary["all_checks_pass"],
                "first_failed_layer": summary["first_failed_layer"],
                "cubic_a": summary["radial_fit"]["cubic"]["parameter"],
                "student_degrees_of_freedom": summary["radial_fit"][
                    "student_degrees_of_freedom"
                ],
                "quality": summary["quality"],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
