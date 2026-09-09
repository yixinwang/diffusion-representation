import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest


RUNNER_PATH = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "cubic_radial_flow_toy"
    / "run.py"
)
SPEC = importlib.util.spec_from_file_location("cubic_radial_flow_toy_run", RUNNER_PATH)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def test_frozen_protocol_hash_and_one_sided_constant() -> None:
    assert runner._sha256_file(runner.PROTOCOL_PATH) == runner.PROTOCOL_HASH
    assert runner.ONE_SIDED_95 == 1.6448536269514722


def test_sandwich_correlation_interval_uses_vector_influence() -> None:
    rng = np.random.default_rng(501)
    values = rng.normal(size=(20_000, 3))
    values[:, 1] = 0.4 * values[:, 0] + values[:, 1]
    energies = np.exp(values)
    result = runner._correlation_and_limits(energies)
    estimate = np.asarray(result["estimate"])
    standard_error = np.asarray(result["sandwich_standard_error"])
    assert 0.20 < estimate[0, 1] < 0.5
    assert standard_error[0, 1] > 0.0
    np.testing.assert_allclose(standard_error, standard_error.T, atol=1e-15, rtol=0.0)
    expected_lower = estimate[0, 1] - runner.ONE_SIDED_95 * standard_error[0, 1]
    assert abs(result["lower_95"][0][1] - expected_lower) < 1e-15

    centered = energies - np.mean(energies, axis=0)
    standard = np.sqrt(np.mean(centered * centered, axis=0))
    rho = estimate[0, 1]
    influence = (
        centered[:, 0] * centered[:, 1] / (standard[0] * standard[1])
        - 0.5
        * rho
        * (centered[:, 0] ** 2 / standard[0] ** 2 + centered[:, 1] ** 2 / standard[1] ** 2)
    )
    brute_force_error = math.sqrt(
        np.sum((influence - np.mean(influence)) ** 2)
        / (len(energies) - 1)
        / len(energies)
    )
    assert abs(standard_error[0, 1] - brute_force_error) < 2e-14

    moments = runner._energy_raw_moment_sums(energies)
    reconstructed = runner._correlation_and_limits_from_moments(moments, len(energies))
    np.testing.assert_allclose(
        reconstructed["sandwich_standard_error"],
        result["sandwich_standard_error"],
        atol=1e-15,
        rtol=0.0,
    )


def test_balanced_context_standard_error_formula() -> None:
    conditions = [
        {"normalized_log_density_advantage": {"estimate": 0.018, "standard_error": 0.002}},
        {"normalized_log_density_advantage": {"estimate": 0.026, "standard_error": 0.003}},
    ]
    result = runner._balanced_primary_summary(conditions)
    expected = 0.5 * math.sqrt(0.002**2 + 0.003**2)
    assert result["estimate"] == 0.022
    assert result["standard_error"] == expected
    assert result["lower_95"] == 0.022 - runner.ONE_SIDED_95 * expected


def test_simulation_records_actual_stream_and_sufficient_statistics() -> None:
    record = runner._simulate_condition(0.03, 3100, 0, 1_000)
    assert record["development_seed"] == 3100
    assert record["stream"] == 0
    assert record["rng_seed"] == 310000
    assert record["count"] == 1_000
    assert np.asarray(record["energy_sum"]).shape == (3,)
    assert np.asarray(record["energy_outer_sum"]).shape == (3, 3)
    assert len(record["energy_raw_moment_sums"]) == 35


def test_serialized_seed_moments_reproduce_aggregate_sandwich_error() -> None:
    records = [
        runner._simulate_condition(0.03, seed, 0, 2_000)
        for seed in (3100, 3101)
    ]
    aggregate = runner._aggregate_condition(records)
    combined = runner._combine_energy_moments(
        aggregate["per_seed_sufficient_statistics"]
    )
    reconstructed = runner._correlation_and_limits_from_moments(combined, 4_000)
    np.testing.assert_allclose(
        reconstructed["sandwich_standard_error"],
        aggregate["band_energy_correlation"]["sandwich_standard_error"],
        atol=0.0,
        rtol=0.0,
    )


def test_numerical_jacobian_is_machine_checked() -> None:
    result = runner._numerical_jacobian_check()
    assert result["passed"]
    assert result["absolute_error"] <= result["tolerance"] == 1e-8


def test_invalid_stage_existing_output_and_controls_without_witness_are_rejected(tmp_path) -> None:
    with pytest.raises(ValueError, match="stage must"):
        runner.run("invalid", tmp_path / "invalid.json")
    existing = tmp_path / "existing.json"
    existing.write_text("preserve me")
    with pytest.raises(FileExistsError):
        runner.run("witness", existing)
    with pytest.raises(ValueError, match="require --witness"):
        runner._validate_witness(None, "commit", {})
