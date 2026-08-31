"""Execute the frozen cubic-radial aligned-toy witness and controls."""

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
import sys
import time

import numpy as np
import scipy
from scipy.special import gammaln, roots_genlaguerre

from qalt.cubic_radial_flow import (
    band_energy_correlation,
    best_affine_variance,
    forward,
    inverse,
    isotropic_normal_log_prob,
    log_prob,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROTOCOL_PATH = PROJECT_ROOT / "qalt" / "theory" / "CUBIC_RADIAL_FLOW_TOY_PROTOCOL.md"
PROTOCOL_HASH = "9c8868bad939af0214d2e22fffc6212013cb7d964040239be7f854702077efa4"
DEVELOPMENT_SEEDS = tuple(range(3100, 3105))
CONFIRMATION_SEEDS = tuple(range(4100, 4130))
VECTORS_PER_CONDITION_AND_SEED = 100_000
DIMENSION = 9
PRIMARY_COEFFICIENTS = (0.030, 0.038)
CONTROL_COEFFICIENTS = (0.0, 0.0138)
PRACTICAL_MARGIN = 0.01
CORRELATION_LIMIT = 0.05
CORRELATION_TOLERANCE = 0.01
ONE_SIDED_95 = 1.6448536269514722
REGISTERED_SOURCE_PATHS = {
    "protocol": PROTOCOL_PATH,
    "implementation": PROJECT_ROOT / "qalt" / "src" / "qalt" / "cubic_radial_flow.py",
    "runner": Path(__file__).resolve(),
    "flow_tests": PROJECT_ROOT / "qalt" / "tests" / "test_cubic_radial_flow.py",
    "runner_tests": PROJECT_ROOT / "qalt" / "tests" / "test_cubic_radial_flow_runner.py",
    "readme": PROJECT_ROOT / "qalt" / "experiments" / "cubic_radial_flow_toy" / "README.md",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _current_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
    ).strip()


def _source_hashes() -> dict[str, str]:
    return {name: _sha256_file(path) for name, path in REGISTERED_SOURCE_PATHS.items()}


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
            committed_blob = subprocess.check_output(
                ["git", "rev-parse", f"HEAD:{relative}"],
                cwd=PROJECT_ROOT,
                text=True,
            ).strip()
            working_blob = subprocess.check_output(
                ["git", "hash-object", relative],
                cwd=PROJECT_ROOT,
                text=True,
            ).strip()
        except subprocess.CalledProcessError as error:
            raise ValueError(f"registered source is not committed: {relative}") from error
        if committed_blob != working_blob:
            raise ValueError(f"registered source differs from HEAD: {relative}")


def _exact_gap_per_coordinate(a: float) -> float:
    if a == 0.0:
        return 0.0
    nodes, weights = roots_genlaguerre(128, DIMENSION / 2.0 - 1.0)
    normalization = math.exp(gammaln(DIMENSION / 2.0))
    expected_log_one = float(np.sum(weights * np.log1p(2.0 * a * nodes)) / normalization)
    expected_log_three = float(np.sum(weights * np.log1p(6.0 * a * nodes)) / normalization)
    variance = best_affine_variance(a, dimension=DIMENSION)
    return (
        0.5 * math.log(variance)
        - ((DIMENSION - 1) / DIMENSION) * expected_log_one
        - expected_log_three / DIMENSION
    )


def _moment_key(exponents: tuple[int, int, int]) -> str:
    return ",".join(str(value) for value in exponents)


def _energy_raw_moment_sums(energies: np.ndarray) -> dict[str, float]:
    values = np.asarray(energies, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or len(values) < 4:
        raise ValueError("energies must have shape [independent vector, 3]")
    powers = [[np.ones(len(values))] for _ in range(3)]
    for coordinate in range(3):
        for order in range(1, 5):
            powers[coordinate].append(powers[coordinate][-1] * values[:, coordinate])
    moments: dict[str, float] = {}
    for first in range(5):
        for second in range(5 - first):
            for third in range(5 - first - second):
                moments[_moment_key((first, second, third))] = float(
                    np.sum(powers[0][first] * powers[1][second] * powers[2][third])
                )
    return moments


def _combine_energy_moments(records: list[dict[str, object]]) -> dict[str, float]:
    if not records:
        raise ValueError("at least one energy-moment record is required")
    keys = set(records[0]["energy_raw_moment_sums"])
    if any(set(record["energy_raw_moment_sums"]) != keys for record in records):
        raise ValueError("energy raw-moment schemas do not match")
    return {
        key: sum(float(record["energy_raw_moment_sums"][key]) for record in records)
        for key in sorted(keys)
    }


def _correlation_and_limits_from_moments(
    moments: dict[str, float], count: int
) -> dict[str, object]:
    if count < 4:
        raise ValueError("at least four independent vectors are required")

    def raw(first: int, second: int, third: int) -> float:
        return float(moments[_moment_key((first, second, third))]) / count

    means = np.array([raw(1, 0, 0), raw(0, 1, 0), raw(0, 0, 1)])

    def pair_raw(row: int, column: int, row_power: int, column_power: int) -> float:
        exponents = [0, 0, 0]
        exponents[row] = row_power
        exponents[column] = column_power
        return raw(*exponents)

    def pair_central(row: int, column: int, row_power: int, column_power: int) -> float:
        value = 0.0
        for first in range(row_power + 1):
            for second in range(column_power + 1):
                value += (
                    math.comb(row_power, first)
                    * math.comb(column_power, second)
                    * (-means[row]) ** (row_power - first)
                    * (-means[column]) ** (column_power - second)
                    * pair_raw(row, column, first, second)
                )
        return value

    variance = np.array(
        [pair_central(index, (index + 1) % 3, 2, 0) for index in range(3)]
    )
    standard = np.sqrt(variance)
    correlation = np.eye(3)
    standard_error = np.zeros((3, 3))
    lower = np.eye(3)
    upper = np.eye(3)
    for row in range(3):
        for column in range(row + 1, 3):
            rho = pair_central(row, column, 1, 1) / (
                standard[row] * standard[column]
            )
            correlation[row, column] = correlation[column, row] = rho
            central_22 = pair_central(row, column, 2, 2)
            influence_second_moment = (
                central_22 / (variance[row] * variance[column])
                - rho
                * (
                    pair_central(row, column, 3, 1)
                    / (standard[row] ** 3 * standard[column])
                    + pair_central(row, column, 1, 3)
                    / (standard[row] * standard[column] ** 3)
                )
                + 0.25
                * rho**2
                * (
                    pair_central(row, column, 4, 0) / variance[row] ** 2
                    + pair_central(row, column, 0, 4) / variance[column] ** 2
                    + 2.0 * central_22 / (variance[row] * variance[column])
                )
            )
            error = math.sqrt(max(0.0, influence_second_moment) / (count - 1))
            standard_error[row, column] = standard_error[column, row] = error
            lower_value = max(-1.0, rho - ONE_SIDED_95 * error)
            upper_value = min(1.0, rho + ONE_SIDED_95 * error)
            lower[row, column] = lower[column, row] = lower_value
            upper[row, column] = upper[column, row] = upper_value
    return {
        "estimate": correlation.tolist(),
        "sandwich_standard_error": standard_error.tolist(),
        "lower_95": lower.tolist(),
        "upper_95": upper.tolist(),
    }


def _correlation_and_limits(energies: np.ndarray) -> dict[str, object]:
    values = np.asarray(energies, dtype=np.float64)
    return _correlation_and_limits_from_moments(
        _energy_raw_moment_sums(values), len(values)
    )


def _simulate_condition(
    a: float, development_seed: int, stream: int, count: int
) -> dict[str, object]:
    rng_seed = 100 * development_seed + stream
    rng = np.random.default_rng(rng_seed)
    log_ratio_sum = 0.0
    log_ratio_square_sum = 0.0
    maximum_absolute_log_ratio = 0.0
    energy_sum = np.zeros(3)
    energy_outer_sum = np.zeros((3, 3))
    maximum_roundtrip_error = 0.0
    maximum_log_det_error = 0.0
    maximum_direction_error = 0.0
    energy_raw_moment_sums: dict[str, float] | None = None
    variance = best_affine_variance(a, dimension=DIMENSION)
    batch_size = 20_000
    for start in range(0, count, batch_size):
        size = min(batch_size, count - start)
        base = rng.normal(size=(size, DIMENSION))
        values, forward_log_det = forward(base, a)
        recovered, inverse_log_det = inverse(values, a)
        log_ratio = (log_prob(values, a) - isotropic_normal_log_prob(values, variance)) / DIMENSION
        energies = np.sum(values.reshape(size, 3, 3) ** 2, axis=2)
        batch_moments = _energy_raw_moment_sums(energies)
        if energy_raw_moment_sums is None:
            energy_raw_moment_sums = batch_moments
        else:
            for key, value in batch_moments.items():
                energy_raw_moment_sums[key] += value
        log_ratio_sum += float(np.sum(log_ratio))
        log_ratio_square_sum += float(np.sum(log_ratio * log_ratio))
        maximum_absolute_log_ratio = max(
            maximum_absolute_log_ratio, float(np.max(np.abs(log_ratio)))
        )
        energy_sum += np.sum(energies, axis=0)
        energy_outer_sum += energies.T @ energies
        maximum_roundtrip_error = max(
            maximum_roundtrip_error, float(np.max(np.abs(recovered - base)))
        )
        maximum_log_det_error = max(
            maximum_log_det_error, float(np.max(np.abs(forward_log_det + inverse_log_det)))
        )
        base_direction = base / np.linalg.norm(base, axis=1, keepdims=True)
        value_direction = values / np.linalg.norm(values, axis=1, keepdims=True)
        maximum_direction_error = max(
            maximum_direction_error,
            float(np.max(np.abs(base_direction - value_direction))),
        )
    return {
        "a": a,
        "development_seed": development_seed,
        "stream": stream,
        "rng_seed": rng_seed,
        "count": count,
        "log_ratio_sum": log_ratio_sum,
        "log_ratio_square_sum": log_ratio_square_sum,
        "maximum_absolute_log_ratio": maximum_absolute_log_ratio,
        "energy_sum": energy_sum.tolist(),
        "energy_outer_sum": energy_outer_sum.tolist(),
        "energy_raw_moment_sums": energy_raw_moment_sums,
        "maximum_roundtrip_error": maximum_roundtrip_error,
        "maximum_log_det_error": maximum_log_det_error,
        "maximum_direction_error": maximum_direction_error,
    }


def _aggregate_condition(records: list[dict[str, object]]) -> dict[str, object]:
    count = sum(int(record["count"]) for record in records)
    total = sum(float(record["log_ratio_sum"]) for record in records)
    square_total = sum(float(record["log_ratio_square_sum"]) for record in records)
    mean = total / count
    variance = max(0.0, (square_total - count * mean * mean) / (count - 1))
    standard_error = math.sqrt(variance / count)
    energy_moments = _combine_energy_moments(records)
    coefficient = float(records[0]["a"])
    sufficient_statistics = [
        {name: value for name, value in record.items() if not name.startswith("_")}
        for record in records
    ]
    return {
        "a": coefficient,
        "independent_vector_count": count,
        "normalized_log_density_advantage": {
            "estimate": mean,
            "standard_error": standard_error,
            "lower_95": mean - ONE_SIDED_95 * standard_error,
            "upper_95": mean + ONE_SIDED_95 * standard_error,
            "exact_expectation": _exact_gap_per_coordinate(coefficient),
        },
        "band_energy_correlation": _correlation_and_limits_from_moments(
            energy_moments, count
        ),
        "pooled_energy_raw_moment_sums": energy_moments,
        "analytic_band_energy_correlation": band_energy_correlation(coefficient),
        "maximum_roundtrip_error": max(float(record["maximum_roundtrip_error"]) for record in records),
        "maximum_log_det_error": max(float(record["maximum_log_det_error"]) for record in records),
        "maximum_direction_error": max(float(record["maximum_direction_error"]) for record in records),
        "maximum_absolute_log_ratio": max(
            float(record["maximum_absolute_log_ratio"]) for record in records
        ),
        "per_seed_sufficient_statistics": sufficient_statistics,
    }


def _permutation_control(
    a: float, development_seed: int, stream: int, count: int
) -> dict[str, object]:
    rng_seed = 100 * development_seed + stream
    rng = np.random.default_rng(rng_seed)
    base = rng.normal(size=(count, DIMENSION))
    values, _ = forward(base, a)
    energies = np.sum(values.reshape(count, 3, 3) ** 2, axis=2)
    permuted = energies.copy()
    permuted[:, 1] = energies[rng.permutation(count), 1]
    permuted[:, 2] = energies[rng.permutation(count), 2]
    original_sorted = np.sort(energies, axis=0)
    permuted_sorted = np.sort(permuted, axis=0)
    return {
        "a": a,
        "development_seed": development_seed,
        "stream": stream,
        "rng_seed": rng_seed,
        "count": count,
        "maximum_sorted_marginal_error": float(np.max(np.abs(original_sorted - permuted_sorted))),
        "energy_sum": np.sum(permuted, axis=0).tolist(),
        "energy_outer_sum": (permuted.T @ permuted).tolist(),
        "energy_raw_moment_sums": _energy_raw_moment_sums(permuted),
    }


def _aggregate_permutations(records: list[dict[str, object]]) -> dict[str, object]:
    count = sum(int(record["count"]) for record in records)
    energy_moments = _combine_energy_moments(records)
    return {
        "a": float(records[0]["a"]),
        "independent_vector_count": count,
        "band_energy_correlation": _correlation_and_limits_from_moments(
            energy_moments, count
        ),
        "pooled_energy_raw_moment_sums": energy_moments,
        "maximum_sorted_marginal_error": max(
            float(record["maximum_sorted_marginal_error"]) for record in records
        ),
        "per_seed_sufficient_statistics": [
            {name: value for name, value in record.items() if not name.startswith("_")}
            for record in records
        ],
    }


def _off_diagonal(values: list[list[float]]) -> np.ndarray:
    matrix = np.asarray(values)
    return matrix[np.triu_indices(3, k=1)]


def _numerical_jacobian_check() -> dict[str, object]:
    base = np.array([0.1, -0.3, 0.7, 0.4, -0.2, 0.9, 0.6, -0.8, 0.5])
    coefficient = 0.038
    scale = 1.2
    _, analytic = forward(base, coefficient, scale)
    epsilon = 1e-6
    jacobian = np.empty((DIMENSION, DIMENSION))
    for coordinate in range(DIMENSION):
        offset = np.zeros(DIMENSION)
        offset[coordinate] = epsilon
        plus, _ = forward(base + offset, coefficient, scale)
        minus, _ = forward(base - offset, coefficient, scale)
        jacobian[:, coordinate] = (plus - minus) / (2.0 * epsilon)
    numerical = float(np.linalg.slogdet(jacobian)[1])
    error = abs(numerical - float(analytic))
    return {
        "analytic_log_determinant": float(analytic),
        "numerical_log_determinant": numerical,
        "absolute_error": error,
        "tolerance": 1e-8,
        "passed": error <= 1e-8,
    }


def _balanced_primary_summary(conditions: list[dict[str, object]]) -> dict[str, float]:
    if len(conditions) != 2:
        raise ValueError("the frozen primary summary requires exactly two contexts")
    estimates = [
        float(condition["normalized_log_density_advantage"]["estimate"])
        for condition in conditions
    ]
    standard_errors = [
        float(condition["normalized_log_density_advantage"]["standard_error"])
        for condition in conditions
    ]
    estimate = sum(estimates) / 2.0
    standard_error = math.sqrt(sum(value * value for value in standard_errors)) / 2.0
    return {
        "estimate": estimate,
        "standard_error": standard_error,
        "lower_95": estimate - ONE_SIDED_95 * standard_error,
    }


def _run_focused_tests() -> dict[str, object]:
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "qalt/tests/test_cubic_radial_flow.py",
        "qalt/tests/test_cubic_radial_flow_runner.py",
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


def _validate_witness(
    witness_path: Path | None,
    source_commit: str,
    source_hashes: dict[str, str],
) -> dict[str, object]:
    if witness_path is None:
        raise ValueError("the controls require --witness pointing to a passed witness result")
    try:
        witness = json.loads(witness_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read a valid witness result: {witness_path}") from error
    required = {
        "schema_version": "cubic_radial_flow_toy_v1",
        "status": "prospective_synthetic_development_no_confirmation_coverage",
        "stage": "witness",
        "source_commit": source_commit,
        "protocol_hash": PROTOCOL_HASH,
        "source_hashes": source_hashes,
        "development_seeds": list(DEVELOPMENT_SEEDS),
        "reserved_confirmation_seeds": list(CONFIRMATION_SEEDS),
        "vectors_per_condition_and_seed": VECTORS_PER_CONDITION_AND_SEED,
        "dimension": DIMENSION,
        "official_cifar_test_deserialized": False,
    }
    for field, expected in required.items():
        if witness.get(field) != expected:
            raise ValueError(f"witness field {field} does not match the frozen controls")
    if witness.get("decisions", {}).get("all_checks_pass") is not True:
        raise ValueError("the controls require a witness with all_checks_pass=true")
    conditions = witness.get("conditions")
    expected_conditions = {"a_0.0300": (0.030, 0), "a_0.0380": (0.038, 1)}
    if not isinstance(conditions, dict) or set(conditions) != set(expected_conditions):
        raise ValueError("witness conditions do not match the frozen primary contexts")
    for key, (coefficient, stream) in expected_conditions.items():
        condition = conditions[key]
        if (
            condition.get("a") != coefficient
            or condition.get("independent_vector_count")
            != len(DEVELOPMENT_SEEDS) * VECTORS_PER_CONDITION_AND_SEED
        ):
            raise ValueError(f"witness condition {key} has the wrong coefficient or count")
        records = condition.get("per_seed_sufficient_statistics")
        if not isinstance(records, list) or len(records) != len(DEVELOPMENT_SEEDS):
            raise ValueError(f"witness condition {key} has the wrong seed records")
        actual_streams = {
            (
                record.get("development_seed"),
                record.get("stream"),
                record.get("rng_seed"),
                record.get("count"),
            )
            for record in records
        }
        expected_streams = {
            (
                seed,
                stream,
                100 * seed + stream,
                VECTORS_PER_CONDITION_AND_SEED,
            )
            for seed in DEVELOPMENT_SEEDS
        }
        if actual_streams != expected_streams:
            raise ValueError(f"witness condition {key} has incorrect random streams")
    test_pass = witness.get("focused_tests", {}).get("passed") is True
    jacobian_pass = witness.get("numerical_jacobian_check", {}).get("passed") is True
    decisions = witness.get("decisions", {})
    recomputed_pass = bool(
        decisions.get("primary_margin_pass") is True
        and decisions.get("condition_checks_pass") is True
        and test_pass
        and jacobian_pass
    )
    if decisions.get("all_checks_pass") is not recomputed_pass:
        raise ValueError("witness all_checks_pass is inconsistent with required checks")
    return witness


def run(stage: str, output: Path, witness_path: Path | None = None) -> dict[str, object]:
    started = time.perf_counter()
    if stage not in {"witness", "controls"}:
        raise ValueError("stage must be 'witness' or 'controls'")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    actual_protocol_hash = _sha256_file(PROTOCOL_PATH)
    if actual_protocol_hash != PROTOCOL_HASH:
        raise ValueError(
            f"frozen protocol hash mismatch: {actual_protocol_hash} != {PROTOCOL_HASH}"
        )
    _assert_registered_sources_committed()
    source_commit = _current_commit()
    source_hashes = _source_hashes()
    witness = (
        _validate_witness(witness_path, source_commit, source_hashes)
        if stage == "controls"
        else None
    )
    focused_tests = _run_focused_tests()
    jacobian_check = _numerical_jacobian_check()
    if not focused_tests["passed"] or not jacobian_check["passed"]:
        raise RuntimeError("focused tests or the numerical-Jacobian check failed")
    coefficients = PRIMARY_COEFFICIENTS if stage == "witness" else CONTROL_COEFFICIENTS
    by_condition: dict[str, object] = {}
    for coefficient_index, coefficient in enumerate(coefficients):
        records = []
        for seed in DEVELOPMENT_SEEDS:
            records.append(
                _simulate_condition(
                    coefficient,
                    seed,
                    coefficient_index if stage == "witness" else coefficient_index + 2,
                    VECTORS_PER_CONDITION_AND_SEED,
                )
            )
        key = f"a_{coefficient:.4f}"
        by_condition[key] = _aggregate_condition(records)

    if stage == "witness":
        primary_summary = _balanced_primary_summary(list(by_condition.values()))
        condition_checks = []
        for condition in by_condition.values():
            estimate = _off_diagonal(condition["band_energy_correlation"]["estimate"])
            lower = _off_diagonal(condition["band_energy_correlation"]["lower_95"])
            expected = float(condition["analytic_band_energy_correlation"])
            condition_checks.append(
                bool(
                    np.max(np.abs(estimate - expected)) < CORRELATION_TOLERANCE
                    and np.min(lower) > CORRELATION_LIMIT
                    and float(condition["maximum_roundtrip_error"]) <= 1e-10
                    and float(condition["maximum_log_det_error"]) <= 1e-10
                    and float(condition["maximum_direction_error"]) <= 1e-12
                )
            )
        decisions = {
            "pooled_normalized_log_density_advantage": {
                **primary_summary,
            },
            "primary_margin_pass": primary_summary["lower_95"] > PRACTICAL_MARGIN,
            "condition_checks_pass": all(condition_checks),
        }
        decisions["all_checks_pass"] = bool(
            decisions["primary_margin_pass"]
            and decisions["condition_checks_pass"]
            and focused_tests["passed"]
            and jacobian_check["passed"]
        )
    else:
        permutation_controls = {}
        for coefficient_index, coefficient in enumerate(PRIMARY_COEFFICIENTS):
            records = [
                _permutation_control(
                    coefficient,
                    seed,
                    coefficient_index + 10,
                    VECTORS_PER_CONDITION_AND_SEED,
                )
                for seed in DEVELOPMENT_SEEDS
            ]
            permutation_controls[f"a_{coefficient:.4f}"] = _aggregate_permutations(records)
        null = by_condition["a_0.0000"]
        low = by_condition["a_0.0138"]
        null_pointwise = float(null["maximum_absolute_log_ratio"]) <= 1e-12
        low_margin = float(low["normalized_log_density_advantage"]["upper_95"]) < PRACTICAL_MARGIN
        permutation_pass = all(
            float(control["maximum_sorted_marginal_error"]) == 0.0
            and np.max(np.abs(_off_diagonal(control["band_energy_correlation"]["estimate"]))) < 0.01
            for control in permutation_controls.values()
        )
        decisions = {
            "null_pointwise_density_pass": null_pointwise,
            "low_headroom_upper_margin_pass": low_margin,
            "permutation_controls_pass": permutation_pass,
            "all_checks_pass": bool(
                null_pointwise
                and low_margin
                and permutation_pass
                and focused_tests["passed"]
                and jacobian_check["passed"]
            ),
        }
        by_condition["permutation_controls"] = permutation_controls

    result = {
        "schema_version": "cubic_radial_flow_toy_v1",
        "status": "prospective_synthetic_development_no_confirmation_coverage",
        "stage": stage,
        "command": [sys.executable, *sys.argv],
        "source_commit": source_commit,
        "protocol_hash": actual_protocol_hash,
        "source_hashes": source_hashes,
        "development_seeds": list(DEVELOPMENT_SEEDS),
        "reserved_confirmation_seeds": list(CONFIRMATION_SEEDS),
        "vectors_per_condition_and_seed": VECTORS_PER_CONDITION_AND_SEED,
        "dimension": DIMENSION,
        "conditions": by_condition,
        "decisions": decisions,
        "focused_tests": focused_tests,
        "numerical_jacobian_check": jacobian_check,
        "witness_result_sha256": (
            _sha256_file(witness_path) if witness is not None and witness_path is not None else None
        ),
        "official_cifar_test_deserialized": False,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        },
        "runtime_seconds": time.perf_counter() - started,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        handle.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("witness", "controls"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--witness", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    summary = run(arguments.stage, arguments.output, arguments.witness)
    print(json.dumps(summary["decisions"], indent=2, sort_keys=True))
