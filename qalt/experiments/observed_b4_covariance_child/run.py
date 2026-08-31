"""Run the frozen seed-2100 observed B4 covariance-only child."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import platform
import resource
import subprocess
import sys
import time
import traceback
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

import numpy as np
import scipy
from scipy import stats

from qalt.covariance_b4_flow import (
    MAX_CONDITION_NUMBER,
    MIN_EIGENVALUE,
    CovarianceB4Flow,
    fit_covariance_b4_flow,
)
from qalt.data_integrity import (
    adaptive_cifar_repair_split,
    cifar_training_batch_paths,
    load_cifar_training_batches,
    sha256_file,
    stable_json_hash,
)
from qalt.observed_block import fit_observed_b4_model, image_haar_inverse
from qalt.observed_block_statistics import holm_adjust, superiority_pvalue
from qalt.observed_routing import coarse_energy_strata, paired_dequantize
from qalt.radial_b4_flow import student_log_ratio, student_to_base


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PARENT_RUNNER_PATH = (
    PROJECT_ROOT / "qalt" / "experiments" / "observed_b4_radial_child" / "run.py"
)
PARENT_RESULT = (
    PROJECT_ROOT
    / "qalt"
    / "results"
    / "observed_b4_radial_child_20260830"
    / "seed_2100"
)
PROTOCOL_PATH = (
    PROJECT_ROOT / "qalt" / "theory" / "OBSERVED_B4_COVARIANCE_CHILD_PROTOCOL.md"
)
PROTOCOL_HASH = "9657fef19cd6e414f1fc19bd048b6a6934b4fddd168fc197941e385a6a2875a0"
DEFAULT_DATA = Path("/ocean/datasets/community/cifar/cifar-10/cifar-10-batches-py")


def _load_parent_runner():
    specification = importlib.util.spec_from_file_location(
        "observed_b4_radial_parent_runner", PARENT_RUNNER_PATH
    )
    if specification is None or specification.loader is None:
        raise ImportError(f"cannot load parent runner from {PARENT_RUNNER_PATH}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


PARENT = _load_parent_runner()

REGISTERED_SEED = 2100
MAXIMUM_SITES = 250_000
MAX_ITERATIONS = 200
SITES_PER_IMAGE = 256
DETAILS_PER_IMAGE = 2_304
ALPHA = 0.05
PRACTICAL_MARGIN = 0.01
FLOAT64_TOLERANCE = 1e-10
HAAR_TOLERANCE = 1e-6
ARMS = ("b4", "diagonal", "block3", "full", "student")
COVARIANCE_ARMS = ("diagonal", "block3", "full")
FROZEN_STUDENT_TAU = 0.4722205736227421

EXPECTED_PARENT_COMMIT = "a05c4ebb57d99e7e1420e17b36e761d5f7b41a5d"
EXPECTED_PARENT_SUMMARY_HASH = (
    "9f83e2e8a2cdad2ef7f1409a651e51694ee22be45513bfcf96ae5de644ccc2fa"
)
EXPECTED_PARENT_COMPLETE_HASH = (
    "827fd24188d45cea7a72c7eae0af2d09691d8f2cc87137f868eb33f194efc69f"
)
EXPECTED_PARENT_SCORES_FILE_HASH = (
    "092c8036d298ae56f2b8723427e84a77abe1e084a5f0655708954b3cfd9102d8"
)
EXPECTED_SAMPLE_HASH = "ff127f2f793dcb022df1fb27a0c3e6a2659d3ac00dc02375018502d06e6e3327"
EXPECTED_B4_PARAMETER_HASH = (
    "33f130a399324d3f26aca9e8de57eea1fd521f43ed611c819feb162c10361564"
)
EXPECTED_FIT_INPUT_HASH = (
    "87e33e5bc0465b68d0cd4a7469bad8f5cd51bc2c8209f70d5fdc5ecbb0fc9192"
)
EXPECTED_HOLDOUT_INPUT_HASH = (
    "3c1dbca3401fe130455a0a7b3eca9b7de8caa7e9f76f718bcd6c80f5fa80d4ad"
)
EXPECTED_B4_SCORE_HASH = "7f510b5f314e1b2731046a156cd2b5b5e62807718612c94b1d16924e1f8664e9"
EXPECTED_STUDENT_SCORE_HASH = (
    "75ea4f0cb1de6bfdb8abb1cbf8015688c5f24eb535dd154b50a859939ddffbba"
)

RUN_CONTEXT: dict[str, object] = {
    "phase": "not_started",
    "opened_files": [],
    "first_failed_layer": "source_data_parent_reproduction",
}

REGISTERED_SOURCE_PATHS = {
    "protocol": PROTOCOL_PATH,
    "runner": Path(__file__).resolve(),
    "covariance_b4_flow": PROJECT_ROOT / "qalt" / "src" / "qalt" / "covariance_b4_flow.py",
    "covariance_flow_tests": PROJECT_ROOT / "qalt" / "tests" / "test_covariance_b4_flow.py",
    "runner_tests": PROJECT_ROOT
    / "qalt"
    / "tests"
    / "test_observed_b4_covariance_child_runner.py",
    "readme": Path(__file__).resolve().with_name("README.md"),
    "slurm": Path(__file__).resolve().with_name("run_seed.slurm"),
    "parent_runner": PARENT_RUNNER_PATH,
    **{
        f"parent_{name}": path
        for name, path in PARENT.REGISTERED_SOURCE_PATHS.items()
    },
}


def _sha256_array(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


def _current_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
    ).strip()


def _source_hashes() -> dict[str, str]:
    return {name: sha256_file(path) for name, path in REGISTERED_SOURCE_PATHS.items()}


def _assert_registered_sources_committed() -> None:
    for path in dict.fromkeys(REGISTERED_SOURCE_PATHS.values()):
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
                ["git", "hash-object", relative], cwd=PROJECT_ROOT, text=True
            ).strip()
        except subprocess.CalledProcessError as error:
            raise ValueError(f"registered source is not committed: {relative}") from error
        if committed != working:
            raise ValueError(f"registered source differs from HEAD: {relative}")


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


def _run_focused_tests() -> dict[str, object]:
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "qalt/tests/test_covariance_b4_flow.py",
        "qalt/tests/test_observed_b4_covariance_child_runner.py",
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


def _verify_parent_result() -> dict[str, object]:
    summary_path = PARENT_RESULT / "summary.json"
    complete_path = PARENT_RESULT / "COMPLETE.json"
    scores_path = PARENT_RESULT / "scores.npz"
    if sha256_file(summary_path) != EXPECTED_PARENT_SUMMARY_HASH:
        raise ValueError("parent summary hash mismatch")
    if sha256_file(complete_path) != EXPECTED_PARENT_COMPLETE_HASH:
        raise ValueError("parent COMPLETE hash mismatch")
    if sha256_file(scores_path) != EXPECTED_PARENT_SCORES_FILE_HASH:
        raise ValueError("parent scores file hash mismatch")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    if summary["source_commit"] != EXPECTED_PARENT_COMMIT:
        raise ValueError("parent source commit mismatch")
    if summary["first_failed_layer"] != "fit_boundary":
        raise ValueError("parent first failed layer mismatch")
    if not summary["reversibility"]["passed"]:
        raise ValueError("parent all-holdout B4 inverse evidence did not pass")
    if complete["payload_hashes"]["scores.npz"] != EXPECTED_PARENT_SCORES_FILE_HASH:
        raise ValueError("parent completion manifest does not bind scores")
    for name, path in PARENT.REGISTERED_SOURCE_PATHS.items():
        expected = summary["source_hashes"].get(name)
        if expected is None or sha256_file(path) != expected:
            raise ValueError(f"current parent dependency differs from parent result: {name}")
    with np.load(scores_path) as archive:
        record_ids = archive["record_ids"].copy()
        labels = archive["labels"].copy()
        b4_scores = archive["b4"].copy()
        student_scores = archive["student"].copy()
    if _sha256_array(b4_scores) != EXPECTED_B4_SCORE_HASH:
        raise ValueError("parent B4 score-array hash mismatch")
    if _sha256_array(student_scores) != EXPECTED_STUDENT_SCORE_HASH:
        raise ValueError("parent Student score-array hash mismatch")
    return {
        "summary_hash": EXPECTED_PARENT_SUMMARY_HASH,
        "complete_hash": EXPECTED_PARENT_COMPLETE_HASH,
        "scores_file_hash": EXPECTED_PARENT_SCORES_FILE_HASH,
        "source_commit": summary["source_commit"],
        "first_failed_layer": summary["first_failed_layer"],
        "inherited_all_holdout_b4_inverse": summary["reversibility"],
        "record_ids": record_ids,
        "labels": labels,
        "b4_scores": b4_scores,
        "student_scores": student_scores,
    }


def _json_safe_welch(summary: dict[str, float]) -> dict[str, object]:
    output: dict[str, object] = dict(summary)
    degrees = float(summary["degrees_of_freedom"])
    critical = float(stats.t.ppf(1.0 - ALPHA / 2.0, degrees))
    half_width = critical * float(summary["standard_error"])
    mean = float(summary["mean"])
    if not all(math.isfinite(value) for value in (critical, half_width, mean)):
        raise FloatingPointError("Welch effect interval is nonfinite")
    output["balanced_class_welch_effect_interval_95"] = [
        mean - half_width,
        mean + half_width,
    ]
    output["effect_interval_confidence_level"] = 0.95
    output["effect_interval_method"] = "two_sided_balanced_class_welch_t"
    output["effect_interval_critical_value"] = critical
    output["degrees_of_freedom_is_infinite"] = not math.isfinite(degrees)
    output["degrees_of_freedom"] = degrees if math.isfinite(degrees) else None
    return output


def _fit_covariance_arms(training: np.ndarray) -> dict[str, CovarianceB4Flow]:
    RUN_CONTEXT["phase"] = "covariance_fit"
    RUN_CONTEXT["first_failed_layer"] = "covariance_positivity"
    return {
        structure: fit_covariance_b4_flow(training, structure=structure)
        for structure in COVARIANCE_ARMS
    }


def _quality_summary(
    scores: dict[str, np.ndarray], labels: np.ndarray
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    contrasts = {
        "b4_minus_full": (scores["full"] - scores["b4"]) / DETAILS_PER_IMAGE,
        "block_minus_full": (scores["full"] - scores["block3"]) / DETAILS_PER_IMAGE,
        "student_minus_full": (scores["full"] - scores["student"]) / DETAILS_PER_IMAGE,
    }
    margins = {
        "b4_minus_full": PRACTICAL_MARGIN,
        "block_minus_full": PRACTICAL_MARGIN,
        "student_minus_full": -PRACTICAL_MARGIN,
    }
    raw: dict[str, float] = {}
    summaries: dict[str, dict[str, float]] = {}
    for name, values in contrasts.items():
        raw[name], summaries[name] = superiority_pvalue(values, labels, margins[name])
    adjusted = holm_adjust(raw, expected_count=3)
    exported = {
        name: {
            **_json_safe_welch(summaries[name]),
            "margin": margins[name],
            "raw_pvalue": raw[name],
            "holm_adjusted_pvalue": adjusted[name],
            "passed": adjusted[name] < ALPHA,
        }
        for name in contrasts
    }
    exported["all_primary_pass"] = bool(
        all(exported[name]["passed"] for name in contrasts)
    )
    return exported, contrasts


def _coordinate_image_statistics(latent: np.ndarray) -> dict[str, np.ndarray]:
    values = np.asarray(latent, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (SITES_PER_IMAGE, 9):
        raise ValueError("latents must have shape [image,256,9]")
    if not np.all(np.isfinite(values)):
        raise ValueError("latents must be finite")
    return {
        "coordinate_mean": np.mean(values, axis=1),
        "coordinate_second": np.einsum("nsi,nsj->nij", values, values)
        / SITES_PER_IMAGE,
    }


def _coordinate_summary(
    statistics: dict[str, np.ndarray], labels: np.ndarray
) -> dict[str, object]:
    mean, mean_error = PARENT._balanced_matrix_summary(
        statistics["coordinate_mean"], labels
    )
    second, second_error = PARENT._balanced_matrix_summary(
        statistics["coordinate_second"], labels
    )
    covariance = second - np.outer(mean, mean)
    influence = (
        statistics["coordinate_second"]
        - statistics["coordinate_mean"][:, :, None] * mean[None, None, :]
        - mean[None, :, None] * statistics["coordinate_mean"][:, None, :]
    )
    _, covariance_error = PARENT._balanced_matrix_summary(influence, labels)

    mean_deviation = np.abs(mean)
    mean_critical = float(stats.norm.ppf(1.0 - ALPHA / (2.0 * 9.0)))
    mean_upper = mean_deviation + mean_critical * mean_error

    target = np.eye(9)
    upper_indices = np.triu_indices(9)
    second_deviation = np.abs(second - target)
    second_critical = float(
        stats.norm.ppf(1.0 - ALPHA / (2.0 * len(upper_indices[0])))
    )
    second_upper = (
        second_deviation[upper_indices]
        + second_critical * second_error[upper_indices]
    )
    standard = np.sqrt(np.diag(covariance))
    correlation = covariance / np.outer(standard, standard)
    off = np.triu_indices(9, k=1)
    return {
        "mean": mean.tolist(),
        "mean_cluster_standard_error": mean_error.tolist(),
        "mean_family_critical_value": mean_critical,
        "mean_simultaneous_upper": mean_upper.tolist(),
        "mean_maximum_absolute": float(np.max(mean_deviation)),
        "mean_maximum_simultaneous_upper": float(np.max(mean_upper)),
        "uncentered_second_moment": second.tolist(),
        "uncentered_second_moment_cluster_standard_error": second_error.tolist(),
        "uncentered_second_moment_family_critical_value": second_critical,
        "uncentered_second_moment_simultaneous_upper_flat": second_upper.tolist(),
        "uncentered_second_moment_maximum_absolute_deviation": float(
            np.max(second_deviation[upper_indices])
        ),
        "uncentered_second_moment_maximum_simultaneous_upper": float(
            np.max(second_upper)
        ),
        "derived_covariance": covariance.tolist(),
        "derived_covariance_cluster_standard_error": covariance_error.tolist(),
        "derived_correlation": correlation.tolist(),
        "derived_correlation_maximum_absolute_off_diagonal": float(
            np.max(np.abs(correlation[off]))
        ),
    }


def _stratum_coordinate_image_statistics(
    latent: np.ndarray, strata: np.ndarray
) -> dict[str, np.ndarray]:
    values = np.asarray(latent, dtype=np.float64)
    assignments = np.asarray(strata, dtype=np.int64)
    if values.ndim != 3 or values.shape[1:] != (SITES_PER_IMAGE, 9):
        raise ValueError("latents must have shape [image,256,9]")
    if assignments.shape != values.shape[:2]:
        raise ValueError("strata must have shape [image,256]")
    if not np.all(np.isfinite(values)):
        raise ValueError("latents must be finite")
    if np.any((assignments < 0) | (assignments >= 4)):
        raise ValueError("strata must be in {0,1,2,3}")
    counts = np.zeros((len(values), 4), dtype=np.int64)
    coordinate_sum = np.zeros((len(values), 4, 9), dtype=np.float64)
    coordinate_second_sum = np.zeros((len(values), 4, 9, 9), dtype=np.float64)
    for stratum in range(4):
        selected = assignments == stratum
        counts[:, stratum] = np.sum(selected, axis=1)
        coordinate_sum[:, stratum] = np.einsum(
            "nsi,ns->ni", values, selected, optimize=True
        )
        coordinate_second_sum[:, stratum] = np.einsum(
            "nsi,nsj,ns->nij", values, values, selected, optimize=True
        )
    if not np.array_equal(
        np.sum(counts, axis=1), np.full(len(values), SITES_PER_IMAGE)
    ):
        raise AssertionError("stratum counts do not partition image sites")
    return {
        "stratum_count": counts,
        "stratum_coordinate_sum": coordinate_sum,
        "stratum_coordinate_second_sum": coordinate_second_sum,
    }


def _balanced_cluster_ratio_summary(
    numerator: np.ndarray, counts: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(numerator, dtype=np.float64)
    denominator = np.asarray(counts, dtype=np.float64)
    classes = np.asarray(labels, dtype=np.int64)
    if values.shape[:2] != denominator.shape or denominator.shape != (len(classes), 4):
        raise ValueError("stratum numerators and counts must align by image")
    if not np.array_equal(np.unique(classes), np.arange(10)):
        raise ValueError("image labels must contain CIFAR classes 0,...,9")
    estimate = np.zeros(values.shape[1:], dtype=np.float64)
    variance = np.zeros(values.shape[1:], dtype=np.float64)
    class_site_counts = np.zeros((4, 10), dtype=np.int64)
    trailing = (1,) * (values.ndim - 2)
    for class_id in range(10):
        selected = classes == class_id
        if np.count_nonzero(selected) != 500:
            raise ValueError("the repair screen requires 500 images per class")
        class_values = values[selected]
        class_counts = denominator[selected]
        site_counts = np.sum(class_counts, axis=0)
        if np.any(site_counts == 0):
            raise ValueError("each class and stratum must contain holdout sites")
        class_site_counts[:, class_id] = site_counts.astype(np.int64)
        ratio = np.sum(class_values, axis=0) / site_counts.reshape((4,) + trailing)
        estimate += ratio / 10.0
        mean_cluster_counts = site_counts / len(class_values)
        influence = (
            class_values
            - class_counts.reshape(class_counts.shape + trailing) * ratio[None]
        ) / mean_cluster_counts.reshape((4,) + trailing)
        variance += np.var(influence, axis=0, ddof=1) / len(class_values) / 100.0
    return estimate, np.sqrt(variance), class_site_counts


def _stratum_coordinate_summary(
    statistics: dict[str, np.ndarray], labels: np.ndarray
) -> dict[str, object]:
    counts = statistics["stratum_count"]
    mean, mean_error, class_site_counts = _balanced_cluster_ratio_summary(
        statistics["stratum_coordinate_sum"], counts, labels
    )
    second, second_error, second_class_site_counts = _balanced_cluster_ratio_summary(
        statistics["stratum_coordinate_second_sum"], counts, labels
    )
    if not np.array_equal(class_site_counts, second_class_site_counts):
        raise AssertionError("stratum diagnostic site counts disagree")

    mean_deviation = np.abs(mean)
    mean_critical = float(stats.norm.ppf(1.0 - ALPHA / (2.0 * 4.0 * 9.0)))
    mean_upper = mean_deviation + mean_critical * mean_error

    target = np.eye(9)
    upper_indices = np.triu_indices(9)
    second_deviation = np.abs(second - target[None])
    second_family_size = 4 * len(upper_indices[0])
    second_critical = float(
        stats.norm.ppf(1.0 - ALPHA / (2.0 * second_family_size))
    )
    second_upper = np.stack(
        [
            second_deviation[stratum][upper_indices]
            + second_critical * second_error[stratum][upper_indices]
            for stratum in range(4)
        ]
    )
    return {
        "site_counts": np.sum(counts, axis=0).astype(np.int64).tolist(),
        "class_site_counts_by_stratum": class_site_counts.tolist(),
        "coordinate_mean": mean.tolist(),
        "coordinate_mean_cluster_standard_error": mean_error.tolist(),
        "coordinate_mean_family_critical_value": mean_critical,
        "coordinate_mean_simultaneous_upper": mean_upper.tolist(),
        "coordinate_mean_maximum_absolute": float(np.max(mean_deviation)),
        "coordinate_mean_maximum_simultaneous_upper": float(np.max(mean_upper)),
        "uncentered_second_moment": second.tolist(),
        "uncentered_second_moment_cluster_standard_error": second_error.tolist(),
        "uncentered_second_moment_family_critical_value": second_critical,
        "uncentered_second_moment_simultaneous_upper_flat_by_stratum": (
            second_upper.tolist()
        ),
        "uncentered_second_moment_maximum_absolute_deviation": float(
            max(
                np.max(second_deviation[stratum][upper_indices])
                for stratum in range(4)
            )
        ),
        "uncentered_second_moment_maximum_simultaneous_upper": float(
            np.max(second_upper)
        ),
    }


def _global_covariance_oracle(gaussian: np.ndarray) -> dict[str, object]:
    values = np.asarray(gaussian, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 9 or not np.all(np.isfinite(values)):
        raise ValueError("oracle coordinates must be a finite [site,9] matrix")
    second = values.T @ values / len(values)
    sign, log_determinant = np.linalg.slogdet(second)
    if sign <= 0 or not math.isfinite(float(log_determinant)):
        raise FloatingPointError("holdout oracle second moment is not positive definite")
    gain = float((np.trace(second) - 9.0 - log_determinant) / 18.0)
    return {
        "role": "rejection_only_not_a_fitted_or_selected_arm",
        "uncentered_second_moment": second.tolist(),
        "eigenvalues": np.linalg.eigvalsh(second).tolist(),
        "log_determinant": float(log_determinant),
        "maximum_gain_nat_per_detail": gain,
        "practical_b4_endpoint_impossible_for_global_covariance_class": bool(
            gain <= PRACTICAL_MARGIN
        ),
    }


def _flow_export(flow: CovarianceB4Flow, training: np.ndarray) -> dict[str, object]:
    return {
        "structure": flow.structure,
        "parameter_count": flow.parameter_count,
        "covariance": flow.covariance.tolist(),
        "eigenvalues": flow.eigenvalues.tolist(),
        "minimum_eigenvalue": float(flow.eigenvalues[0]),
        "condition_number": float(flow.eigenvalues[-1] / flow.eigenvalues[0]),
        "log_det_whitening": flow.log_det_whitening,
        "training_mean": np.mean(training, axis=0).tolist(),
        "training_mean_log_ratio": float(np.mean(flow.log_ratio(training))),
        "parameter_hash": _sha256_array(flow.covariance),
    }


def _time_callable(
    function: Callable[[], float], warmups: int = 2, repetitions: int = 9
) -> dict[str, object]:
    for _ in range(warmups):
        value = float(function())
        if not math.isfinite(value):
            raise FloatingPointError("benchmark warmup produced a nonfinite checksum")
    elapsed = []
    checksums = []
    for _ in range(repetitions):
        started = time.perf_counter()
        checksums.append(float(function()))
        elapsed.append(time.perf_counter() - started)
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
    model,
    residual: np.ndarray,
    gaussian: np.ndarray,
    strata: np.ndarray,
    full: CovarianceB4Flow,
) -> dict[str, object]:
    rows = np.asarray(residual)[:10_000]
    repaired = np.asarray(gaussian)[:10_000]
    labels = np.asarray(strata)[:10_000]
    peak_before = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

    def b4_score() -> float:
        return float(np.sum(PARENT._coordinates_from_residual(model, rows, labels)[2]))

    def full_score() -> float:
        encoded, _, b4_log_prob, _, _ = PARENT._coordinates_from_residual(
            model, rows, labels
        )
        return float(np.sum(b4_log_prob + full.log_ratio(encoded)))

    def b4_roundtrip() -> float:
        encoded, _, _, _, _ = PARENT._coordinates_from_residual(model, rows, labels)
        decoded, _ = PARENT._b4_inverse_residual(model, encoded, labels)
        return float(np.sum(decoded[::100]))

    def full_roundtrip() -> float:
        encoded, _, _, _, _ = PARENT._coordinates_from_residual(model, rows, labels)
        base, _ = full.forward(encoded)
        restored, _ = full.inverse(base)
        decoded, _ = PARENT._b4_inverse_residual(model, restored, labels)
        return float(np.sum(decoded[::100]))

    timings = {
        "b4_full_score": _time_callable(b4_score),
        "full_covariance_score": _time_callable(full_score),
        "b4_repaired_roundtrip": _time_callable(b4_roundtrip),
        "full_covariance_roundtrip": _time_callable(full_roundtrip),
        "covariance_layer_score": _time_callable(
            lambda: float(np.sum(full.log_ratio(repaired)))
        ),
        "covariance_layer_roundtrip": _time_callable(
            lambda: float(np.sum(full.inverse(full.forward(repaired)[0])[0][::100]))
        ),
    }
    for measurement in timings.values():
        measurement["vectors_per_second"] = len(rows) / measurement["median_seconds"]
    ratios = {
        "full_score_over_b4": timings["full_covariance_score"]["median_seconds"]
        / timings["b4_full_score"]["median_seconds"],
        "full_roundtrip_over_b4": timings["full_covariance_roundtrip"][
            "median_seconds"
        ]
        / timings["b4_repaired_roundtrip"]["median_seconds"],
    }
    return {
        "vector_count": len(rows),
        "timings": timings,
        "ratios": ratios,
        "score_pass": ratios["full_score_over_b4"] <= 1.10,
        "roundtrip_pass": ratios["full_roundtrip_over_b4"] <= 1.10,
        "all_checks_pass": bool(
            ratios["full_score_over_b4"] <= 1.10
            and ratios["full_roundtrip_over_b4"] <= 1.10
        ),
        "peak_rss_kib_before": peak_before,
        "peak_rss_kib_after": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
    }


def _probe_image_positions(
    record_ids: np.ndarray, labels: np.ndarray
) -> np.ndarray:
    selected = []
    for class_id in range(10):
        candidates = np.flatnonzero(labels == class_id)
        order = np.argsort(record_ids[candidates], kind="stable")
        selected.extend(candidates[order[:10]].tolist())
    return np.asarray(selected, dtype=np.int64)


def _full_image_reversibility_probe(
    model,
    flow: CovarianceB4Flow,
    images: np.ndarray,
    record_ids: np.ndarray,
    labels: np.ndarray,
    coarse: np.ndarray,
    blocks: np.ndarray,
    seed: int,
) -> dict[str, object]:
    positions = _probe_image_positions(record_ids, labels)
    probe_blocks = blocks[positions]
    gaussian, b4_forward, _, residual, strata = PARENT._b4_coordinates(
        model, coarse[positions], probe_blocks
    )
    base, flow_forward = flow.forward(gaussian)
    restored, flow_inverse = flow.inverse(base)
    strata_after, _ = coarse_energy_strata(coarse[positions], model.boundaries)
    recovered_residual, b4_inverse = PARENT._b4_inverse_residual(
        model, restored, strata_after
    )
    inverse_location = model.location.predict_flat(
        coarse[positions], np.zeros_like(probe_blocks)
    ).reshape(-1, 3, 3)
    recovered_blocks = (inverse_location + recovered_residual).reshape(
        probe_blocks.shape
    )
    reference = paired_dequantize(images[positions], record_ids[positions], seed)
    reconstructed = image_haar_inverse(coarse[positions], recovered_blocks)
    class_counts = {
        str(class_id): int(np.count_nonzero(labels[positions] == class_id))
        for class_id in range(10)
    }
    return {
        "image_positions": positions.tolist(),
        "record_ids": record_ids[positions].tolist(),
        "record_ids_hash": _sha256_array(record_ids[positions].astype("<i8")),
        "class_counts": class_counts,
        "site_count": int(len(positions) * SITES_PER_IMAGE),
        "stratum_mismatch_count": int(np.count_nonzero(strata_after != strata)),
        "flow_value_roundtrip_max_error": float(
            np.max(np.abs(restored - gaussian))
        ),
        "residual_roundtrip_max_error": float(
            np.max(np.abs(recovered_residual - residual))
        ),
        "conditional_block_roundtrip_max_error": float(
            np.max(np.abs(recovered_blocks - probe_blocks))
        ),
        "full_log_det_cancellation_max_error": float(
            np.max(np.abs(b4_forward + flow_forward + flow_inverse + b4_inverse))
        ),
        "haar_roundtrip_max_error": float(np.max(np.abs(reconstructed - reference))),
    }


def _top_radius_site_indices(
    gaussian: np.ndarray, strata: np.ndarray, count: int = 32
) -> np.ndarray:
    radius_squared = np.einsum("ni,ni->n", gaussian, gaussian)
    row = np.arange(len(gaussian), dtype=np.int64)
    selected = []
    for stratum in range(4):
        candidates = row[strata == stratum]
        order = np.lexsort((candidates, -radius_squared[candidates]))
        if len(candidates) < count:
            raise ValueError("each stratum must contain at least 32 holdout sites")
        selected.extend(candidates[order[:count]].tolist())
    return np.asarray(selected, dtype=np.int64)


def _tail_reversibility_probe(
    model,
    flow: CovarianceB4Flow,
    coarse: np.ndarray,
    blocks: np.ndarray,
    holdout_record_ids: np.ndarray,
    gaussian_all: np.ndarray,
    strata_all: np.ndarray,
) -> dict[str, object]:
    indices = _top_radius_site_indices(gaussian_all, strata_all)
    gaussian, b4_forward, _, residual, strata = PARENT._b4_coordinates(
        model, coarse, blocks, indices
    )
    base, flow_forward = flow.forward(gaussian)
    restored, flow_inverse = flow.inverse(base)
    strata_after, _ = coarse_energy_strata(coarse, model.boundaries, indices)
    recovered, b4_inverse = PARENT._b4_inverse_residual(model, restored, strata_after)
    radii = np.linalg.norm(gaussian, axis=1)
    return {
        "global_site_indices": indices.tolist(),
        "global_site_indices_hash": _sha256_array(indices.astype("<i8")),
        "record_ids": holdout_record_ids[indices // SITES_PER_IMAGE].tolist(),
        "spatial_site_indices": (indices % SITES_PER_IMAGE).tolist(),
        "strata": strata.tolist(),
        "radii": radii.tolist(),
        "counts_by_stratum": {
            str(value): int(np.count_nonzero(strata == value)) for value in range(4)
        },
        "stratum_mismatch_count": int(np.count_nonzero(strata_after != strata)),
        "flow_value_roundtrip_max_error": float(
            np.max(np.abs(restored - gaussian))
        ),
        "residual_roundtrip_max_error": float(np.max(np.abs(recovered - residual))),
        "full_log_det_cancellation_max_error": float(
            np.max(np.abs(b4_forward + flow_forward + flow_inverse + b4_inverse))
        ),
    }


def run(
    seed: int,
    output: Path,
    data_root: Path,
    expected_source_commit: str,
) -> dict[str, object]:
    global RUN_CONTEXT
    started = time.perf_counter()
    RUN_CONTEXT = {
        "phase": "argument_validation",
        "opened_files": [],
        "first_failed_layer": "source_data_parent_reproduction",
    }
    if seed != REGISTERED_SEED:
        raise ValueError(f"this screen requires seed {REGISTERED_SEED}")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    if os.environ.get("OMP_NUM_THREADS") != "1":
        raise ValueError("this screen requires OMP_NUM_THREADS=1")
    actual_protocol_hash = sha256_file(PROTOCOL_PATH)
    if actual_protocol_hash != PROTOCOL_HASH:
        raise ValueError("frozen covariance-child protocol hash mismatch")
    _assert_registered_sources_committed()
    source_commit = _current_commit()
    if source_commit != expected_source_commit:
        raise ValueError(
            f"submitted source commit {expected_source_commit!r} does not equal HEAD {source_commit}"
        )
    source_hashes = _source_hashes()
    parent_evidence = _verify_parent_result()
    focused_tests = _run_focused_tests()
    if not focused_tests["passed"]:
        raise RuntimeError("focused covariance-child tests failed")

    RUN_CONTEXT["phase"] = "data_load"
    opened_files: list[str] = []
    RUN_CONTEXT["opened_files"] = opened_files
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
    fit_input_hash = _sha256_array(fit_images)
    holdout_input_hash = _sha256_array(holdout_images)
    if fit_input_hash != EXPECTED_FIT_INPUT_HASH:
        raise AssertionError("fitting-image hash differs from the frozen parent")
    if holdout_input_hash != EXPECTED_HOLDOUT_INPUT_HASH:
        raise AssertionError("repair-image hash differs from the frozen parent")
    if np.intersect1d(discovery_ids, fit_ids).size or np.intersect1d(
        discovery_ids, holdout_ids
    ).size:
        raise AssertionError("excluded discovery records entered downstream arrays")
    if any(Path(path).name == "test_batch" for path in opened_files):
        raise AssertionError("official CIFAR test_batch entered the opened-file ledger")
    if not np.array_equal(parent_evidence["record_ids"], holdout_ids):
        raise AssertionError("repair record order differs from the parent result")
    if not np.array_equal(parent_evidence["labels"], holdout_labels):
        raise AssertionError("repair labels differ from the parent result")

    print("build fitting and repair coefficients", flush=True)
    fit_coarse, fit_blocks, fit_dequant_hash = PARENT._build_coefficients(
        fit_images, fit_ids, seed, lambda message: print(f"fit {message}", flush=True)
    )
    holdout_coarse, holdout_blocks, holdout_dequant_hash = PARENT._build_coefficients(
        holdout_images,
        holdout_ids,
        seed,
        lambda message: print(f"holdout {message}", flush=True),
    )

    RUN_CONTEXT["phase"] = "parent_reproduction"
    print("refit frozen B4 parent", flush=True)
    models = fit_observed_b4_model(
        fit_coarse,
        fit_blocks,
        rng=np.random.default_rng(seed),
        maximum_sites=MAXIMUM_SITES,
        max_iterations=MAX_ITERATIONS,
        progress=lambda message: print(f"model {message}", flush=True),
    )
    if len(models.sample) != MAXIMUM_SITES or models.sample_hash != EXPECTED_SAMPLE_HASH:
        raise AssertionError("common fitting-site sample differs from the frozen parent")
    b4_parameter_hash = PARENT._b4_parameter_hash(models.b4)
    if b4_parameter_hash != EXPECTED_B4_PARAMETER_HASH:
        raise AssertionError("B4 parameter hash differs from the frozen parent")
    if not all(all(cell.converged for cell in band) for band in models.b4.fit_diagnostics):
        raise AssertionError("B4 parent did not converge in every cell")

    fit_gaussian, fit_log_det, fit_b4_log_prob, fit_residual, fit_strata = (
        PARENT._b4_coordinates(models.b4, fit_coarse, fit_blocks, models.sample)
    )
    standard_fit = -0.5 * (
        9 * math.log(2.0 * math.pi) + np.sum(fit_gaussian * fit_gaussian, axis=1)
    )
    fit_density_parity = float(
        np.max(np.abs(fit_b4_log_prob - standard_fit - fit_log_det))
    )
    recovered_fit, inverse_fit_log_det = PARENT._b4_inverse_residual(
        models.b4, fit_gaussian[:4096], fit_strata[:4096]
    )
    fit_roundtrip = float(np.max(np.abs(recovered_fit - fit_residual[:4096])))
    fit_log_det_cancellation = float(
        np.max(np.abs(inverse_fit_log_det + fit_log_det[:4096]))
    )
    if max(fit_density_parity, fit_roundtrip, fit_log_det_cancellation) > FLOAT64_TOLERANCE:
        raise AssertionError("B4 reversible-coordinate parity failed")

    print("reproduce frozen parent holdout scores before covariance fitting", flush=True)
    holdout_site_count = len(holdout_ids) * SITES_PER_IMAGE
    holdout_gaussian = np.empty((holdout_site_count, 9), dtype=np.float64)
    holdout_b4_site = np.empty(holdout_site_count, dtype=np.float64)
    holdout_strata = np.empty(holdout_site_count, dtype=np.int64)
    parent_b4_scores = np.empty(len(holdout_ids), dtype=np.float64)
    parent_student_scores = np.empty(len(holdout_ids), dtype=np.float64)
    benchmark_probe = None
    holdout_density_parity = 0.0
    for start in range(0, len(holdout_ids), 128):
        stop = min(start + 128, len(holdout_ids))
        gaussian, forward_log_det, b4_site, residual, strata = PARENT._b4_coordinates(
            models.b4,
            holdout_coarse[start:stop],
            holdout_blocks[start:stop],
        )
        site_start = start * SITES_PER_IMAGE
        site_stop = stop * SITES_PER_IMAGE
        holdout_gaussian[site_start:site_stop] = gaussian
        holdout_b4_site[site_start:site_stop] = b4_site
        holdout_strata[site_start:site_stop] = strata
        image_count = stop - start
        b4_image = np.sum(b4_site.reshape(image_count, SITES_PER_IMAGE), axis=1)
        parent_b4_scores[start:stop] = b4_image
        parent_student_scores[start:stop] = b4_image + np.sum(
            student_log_ratio(gaussian, FROZEN_STUDENT_TAU).reshape(
                image_count, SITES_PER_IMAGE
            ),
            axis=1,
        )
        standard = -0.5 * (
            9 * math.log(2.0 * math.pi) + np.sum(gaussian * gaussian, axis=1)
        )
        holdout_density_parity = max(
            holdout_density_parity,
            float(np.max(np.abs(b4_site - standard - forward_log_det))),
        )
        if benchmark_probe is None:
            benchmark_probe = (residual.copy(), gaussian.copy(), strata.copy())
    if _sha256_array(parent_b4_scores) != EXPECTED_B4_SCORE_HASH:
        raise AssertionError("reproduced B4 holdout-score hash mismatch")
    if _sha256_array(parent_student_scores) != EXPECTED_STUDENT_SCORE_HASH:
        raise AssertionError("reproduced Student holdout-score hash mismatch")
    if not np.array_equal(parent_b4_scores, parent_evidence["b4_scores"]):
        raise AssertionError("reproduced B4 scores differ from parent artifact")
    if not np.array_equal(parent_student_scores, parent_evidence["student_scores"]):
        raise AssertionError("reproduced Student scores differ from parent artifact")

    print("fit diagonal, block3, and full covariance controls", flush=True)
    covariance_fit_started = time.perf_counter()
    flows = _fit_covariance_arms(fit_gaussian)
    covariance_fit_seconds = time.perf_counter() - covariance_fit_started
    covariance_fits = {
        name: _flow_export(flow, fit_gaussian) for name, flow in flows.items()
    }
    RUN_CONTEXT["phase"] = "repair_holdout_evaluation"
    RUN_CONTEXT["first_failed_layer"] = "normalized_density_reversibility"

    scores = {
        "b4": parent_b4_scores.copy(),
        "student": parent_student_scores.copy(),
        "diagonal": np.empty(len(holdout_ids)),
        "block3": np.empty(len(holdout_ids)),
        "full": np.empty(len(holdout_ids)),
    }
    diagnostic_parts: dict[str, dict[str, list[np.ndarray]]] = {
        arm: defaultdict(list) for arm in ARMS
    }
    coordinate_parts: dict[str, dict[str, list[np.ndarray]]] = {
        arm: defaultdict(list) for arm in ARMS
    }
    stratum_coordinate_parts: dict[str, list[np.ndarray]] = defaultdict(list)
    covariance_reversibility = {
        "identity": {"value_max_error": 0.0, "log_det_max_error": 0.0},
        **{
            arm: {"value_max_error": 0.0, "log_det_max_error": 0.0}
            for arm in COVARIANCE_ARMS
        },
    }
    score_started = time.perf_counter()
    for start in range(0, len(holdout_ids), 128):
        stop = min(start + 128, len(holdout_ids))
        site_start = start * SITES_PER_IMAGE
        site_stop = stop * SITES_PER_IMAGE
        gaussian = holdout_gaussian[site_start:site_stop]
        b4_site = holdout_b4_site[site_start:site_stop]
        image_count = stop - start
        latent = {"b4": gaussian}
        b4_image = scores["b4"][start:stop]
        for arm, flow in flows.items():
            base, forward_log_det = flow.forward(gaussian)
            restored, inverse_log_det = flow.inverse(base)
            covariance_reversibility[arm]["value_max_error"] = max(
                covariance_reversibility[arm]["value_max_error"],
                float(np.max(np.abs(restored - gaussian))),
            )
            covariance_reversibility[arm]["log_det_max_error"] = max(
                covariance_reversibility[arm]["log_det_max_error"],
                float(np.max(np.abs(forward_log_det + inverse_log_det))),
            )
            scores[arm][start:stop] = b4_image + np.sum(
                flow.log_ratio(gaussian).reshape(image_count, SITES_PER_IMAGE),
                axis=1,
            )
            latent[arm] = base
        latent["student"] = student_to_base(gaussian, FROZEN_STUDENT_TAU)[0]
        for arm, values in latent.items():
            shaped = values.reshape(image_count, SITES_PER_IMAGE, 9)
            for name, array in PARENT._latent_image_statistics(shaped).items():
                diagnostic_parts[arm][name].append(array)
            for name, array in _coordinate_image_statistics(shaped).items():
                coordinate_parts[arm][name].append(array)
        for name, array in _stratum_coordinate_image_statistics(
            latent["full"].reshape(image_count, SITES_PER_IMAGE, 9),
            holdout_strata[site_start:site_stop].reshape(
                image_count, SITES_PER_IMAGE
            ),
        ).items():
            stratum_coordinate_parts[name].append(array)
    score_seconds = time.perf_counter() - score_started

    if any(not np.all(np.isfinite(values)) for values in scores.values()):
        raise AssertionError("nonfinite repair-holdout score")
    diagnostic_arrays = {
        arm: {name: np.concatenate(parts) for name, parts in values.items()}
        for arm, values in diagnostic_parts.items()
    }
    coordinate_arrays = {
        arm: {name: np.concatenate(parts) for name, parts in values.items()}
        for arm, values in coordinate_parts.items()
    }
    stratum_coordinate_arrays = {
        name: np.concatenate(parts) for name, parts in stratum_coordinate_parts.items()
    }
    diagnostic_summaries = {
        arm: PARENT._diagnostic_summary(values, holdout_labels)
        for arm, values in diagnostic_arrays.items()
    }
    coordinate_summaries = {
        arm: _coordinate_summary(values, holdout_labels)
        for arm, values in coordinate_arrays.items()
    }
    quality, contrasts = _quality_summary(scores, holdout_labels)
    holdout_oracle = _global_covariance_oracle(holdout_gaussian)
    stratum_coordinate_summary = _stratum_coordinate_summary(
        stratum_coordinate_arrays, holdout_labels
    )

    full_coordinate = coordinate_summaries["full"]
    full_diagnostic = diagnostic_summaries["full"]
    diagnostic_passes = {
        "coordinate_mean": bool(
            full_coordinate["mean_maximum_absolute"] <= 0.02
            and full_coordinate["mean_maximum_simultaneous_upper"] <= 0.03
        ),
        "aggregate_uncentered_second_moment": bool(
            full_coordinate[
                "uncentered_second_moment_maximum_absolute_deviation"
            ]
            <= 0.03
            and full_coordinate[
                "uncentered_second_moment_maximum_simultaneous_upper"
            ]
            <= 0.04
        ),
        "stratum_coordinate_mean": bool(
            stratum_coordinate_summary["coordinate_mean_maximum_absolute"] <= 0.02
            and stratum_coordinate_summary[
                "coordinate_mean_maximum_simultaneous_upper"
            ]
            <= 0.03
        ),
        "stratum_uncentered_second_moment": bool(
            stratum_coordinate_summary[
                "uncentered_second_moment_maximum_absolute_deviation"
            ]
            <= 0.03
            and stratum_coordinate_summary[
                "uncentered_second_moment_maximum_simultaneous_upper"
            ]
            <= 0.04
        ),
        "radial_pit": bool(
            full_diagnostic["radial_pit"]["maximum_deviation"] <= 0.02
            and full_diagnostic["radial_pit"]["maximum_simultaneous_upper"] <= 0.03
        ),
        "angular": bool(
            full_diagnostic["angular"]["maximum_deviation"] <= 0.03
            and full_diagnostic["angular"]["maximum_simultaneous_upper"] <= 0.04
        ),
        "band_energy_share": bool(
            full_diagnostic["band_energy_share"]["maximum_deviation"] <= 0.03
            and full_diagnostic["band_energy_share"]["maximum_simultaneous_upper"]
            <= 0.04
        ),
        "band_energy_correlation": bool(
            full_diagnostic["latent_band_energy_correlation"]["maximum_absolute"]
            <= 0.05
            and full_diagnostic["latent_band_energy_correlation"][
                "maximum_simultaneous_upper_absolute"
            ]
            <= 0.07
        ),
    }

    full_probe = _full_image_reversibility_probe(
        models.b4,
        flows["full"],
        holdout_images,
        holdout_ids,
        holdout_labels,
        holdout_coarse,
        holdout_blocks,
        seed,
    )
    tail_probe = _tail_reversibility_probe(
        models.b4,
        flows["full"],
        holdout_coarse,
        holdout_blocks,
        holdout_ids,
        holdout_gaussian,
        holdout_strata,
    )
    covariance_maximum = max(
        value
        for arm in covariance_reversibility.values()
        for value in arm.values()
    )
    probe_float64_maximum = max(
        value
        for probe in (full_probe, tail_probe)
        for name, value in probe.items()
        if name.endswith("max_error") and not name.startswith("haar")
    )
    reversibility_pass = bool(
        covariance_maximum <= FLOAT64_TOLERANCE
        and probe_float64_maximum <= FLOAT64_TOLERANCE
        and full_probe["stratum_mismatch_count"] == 0
        and tail_probe["stratum_mismatch_count"] == 0
        and full_probe["haar_roundtrip_max_error"] <= HAAR_TOLERANCE
        and holdout_density_parity <= FLOAT64_TOLERANCE
    )

    if benchmark_probe is None:
        raise AssertionError("benchmark probe was not created")
    benchmark = _benchmark(
        models.b4,
        benchmark_probe[0],
        benchmark_probe[1],
        benchmark_probe[2],
        flows["full"],
    )

    ordered_layers = [
        ("source_data_parent_reproduction", data_integrity_pass),
        ("covariance_positivity", True),
        ("normalized_density_reversibility", reversibility_pass),
        ("paired_nll", bool(quality["all_primary_pass"])),
        ("coordinate_mean", diagnostic_passes["coordinate_mean"]),
        (
            "aggregate_uncentered_second_moment",
            diagnostic_passes["aggregate_uncentered_second_moment"],
        ),
        ("stratum_coordinate_mean", diagnostic_passes["stratum_coordinate_mean"]),
        (
            "stratum_uncentered_second_moment",
            diagnostic_passes["stratum_uncentered_second_moment"],
        ),
        ("angular_moments", diagnostic_passes["angular"]),
        ("band_energy_share", diagnostic_passes["band_energy_share"]),
        ("band_energy_correlation", diagnostic_passes["band_energy_correlation"]),
        ("cumulative_radial_pit", diagnostic_passes["radial_pit"]),
        ("local_cpu", bool(benchmark["all_checks_pass"])),
    ]
    first_failed_layer = next((name for name, passed in ordered_layers if not passed), None)
    all_checks_pass = first_failed_layer is None
    RUN_CONTEXT["first_failed_layer"] = first_failed_layer

    summary: dict[str, object] = {
        "schema_version": "observed_b4_covariance_child_seed_v1",
        "status": "second_adaptive_repair_screen_no_confirmation_coverage",
        "seed": seed,
        "source_commit": source_commit,
        "protocol_hash": actual_protocol_hash,
        "source_hashes": source_hashes,
        "parent_evidence": {
            key: value
            for key, value in parent_evidence.items()
            if key not in {"record_ids", "labels", "b4_scores", "student_scores"}
        },
        "focused_tests": focused_tests,
        "split_hash": stable_json_hash(split),
        "fit_size": int(len(fit_ids)),
        "holdout_size": int(len(holdout_ids)),
        "excluded_discovery_size": int(len(discovery_ids)),
        "fit_input_hash": fit_input_hash,
        "holdout_input_hash": holdout_input_hash,
        "fit_record_ids_hash": _sha256_array(fit_ids.astype("<i8")),
        "holdout_record_ids_hash": _sha256_array(holdout_ids.astype("<i8")),
        "holdout_labels_hash": _sha256_array(holdout_labels.astype("<i8")),
        "fit_dequant_hash": fit_dequant_hash,
        "holdout_dequant_hash": holdout_dequant_hash,
        "opened_files": opened_files,
        "expected_opened_files": expected_opened_files,
        "opened_file_ledger_exact_match": data_integrity_pass,
        "official_test_deserialized": False,
        "site_sample_count": int(len(models.sample)),
        "site_sample_hash": models.sample_hash,
        "b4_parameter_hash": b4_parameter_hash,
        "b4_parameter_count": models.b4.parameter_count(),
        "b4_fit_trace": models.b4.fit_trace_export(),
        "parent_score_reproduction": {
            "b4_score_hash": _sha256_array(scores["b4"]),
            "student_score_hash": _sha256_array(scores["student"]),
            "student_tau": FROZEN_STUDENT_TAU,
            "holdout_density_parity_max_error": holdout_density_parity,
        },
        "b4_coordinate_parity": {
            "fit_density_max_error": fit_density_parity,
            "fit_roundtrip_max_error": fit_roundtrip,
            "fit_log_det_cancellation_max_error": fit_log_det_cancellation,
        },
        "covariance_fit": {
            "seconds": covariance_fit_seconds,
            "minimum_eigenvalue_gate": MIN_EIGENVALUE,
            "maximum_condition_number_gate": MAX_CONDITION_NUMBER,
            "arms": covariance_fits,
        },
        "quality": quality,
        "holdout_global_covariance_oracle": holdout_oracle,
        "coordinate_diagnostics": coordinate_summaries,
        "full_stratum_coordinate_diagnostics": stratum_coordinate_summary,
        "diagnostics": diagnostic_summaries,
        "diagnostic_passes": diagnostic_passes,
        "reversibility": {
            "all_holdout_covariance": covariance_reversibility,
            "full_image_probe": full_probe,
            "top_radius_probe": tail_probe,
            "inherited_parent_b4_inverse": True,
            "float64_tolerance": FLOAT64_TOLERANCE,
            "haar_tolerance": HAAR_TOLERANCE,
            "passed": reversibility_pass,
        },
        "benchmark": benchmark,
        "score_seconds": score_seconds,
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
    for arm, values in diagnostic_arrays.items():
        for name, array in values.items():
            diagnostic_output[f"{arm}_{name}"] = array
    for arm, values in coordinate_arrays.items():
        for name, array in values.items():
            diagnostic_output[f"{arm}_{name}"] = array
    for name, array in stratum_coordinate_arrays.items():
        diagnostic_output[f"full_{name}"] = array
    np.savez_compressed(output / "diagnostics.npz", **diagnostic_output)
    covariance_output = {}
    for arm, flow in flows.items():
        covariance_output[f"{arm}_covariance"] = flow.covariance
        covariance_output[f"{arm}_whitening"] = flow.whitening
        covariance_output[f"{arm}_coloring"] = flow.coloring
        covariance_output[f"{arm}_eigenvalues"] = flow.eigenvalues
    np.savez(output / "covariance_flows.npz", **covariance_output)
    for name in ("scores.npz", "diagnostics.npz", "covariance_flows.npz"):
        _fsync_file(output / name)
    _write_json(output / "summary.json", summary)
    _write_json(
        output / "config.json",
        {
            "seed": seed,
            "data_root": str(data_root),
            "source_commit": source_commit,
            "protocol_hash": actual_protocol_hash,
            "parent_result": str(PARENT_RESULT),
            "maximum_sites": MAXIMUM_SITES,
            "max_iterations": MAX_ITERATIONS,
            "student_tau": FROZEN_STUDENT_TAU,
            "float64_tolerance": FLOAT64_TOLERANCE,
            "haar_tolerance": HAAR_TOLERANCE,
        },
    )
    payload_names = (
        "config.json",
        "covariance_flows.npz",
        "diagnostics.npz",
        "scores.npz",
        "summary.json",
    )
    payload_hashes = {name: sha256_file(output / name) for name in payload_names}
    checksum_path = output / "SHA256SUMS"
    checksum_path.write_text(
        "".join(
            f"{digest}  {name}\n" for name, digest in sorted(payload_hashes.items())
        ),
        encoding="utf-8",
    )
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
            failure_path = arguments.output / "failure.json"
            _write_json(
                failure_path,
                {
                    "status": "failed_before_complete_result",
                    "phase": RUN_CONTEXT.get("phase"),
                    "first_failed_layer": RUN_CONTEXT.get("first_failed_layer"),
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
                    },
                    "command": [sys.executable, *sys.argv],
                },
            )
            failure_hash_path = arguments.output / "FAILURE.sha256"
            failure_hash_path.write_text(
                f"{sha256_file(failure_path)}  failure.json\n", encoding="utf-8"
            )
            _fsync_file(failure_hash_path)
            _fsync_directory(arguments.output)
            _fsync_directory(arguments.output.parent)
        raise
    print(
        json.dumps(
            {
                "all_checks_pass": summary["all_checks_pass"],
                "first_failed_layer": summary["first_failed_layer"],
                "quality": summary["quality"],
                "covariance_fit": summary["covariance_fit"],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
