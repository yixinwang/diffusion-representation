import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest


RUNNER_PATH = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "observed_b4_covariance_child"
    / "run.py"
)
SPEC = importlib.util.spec_from_file_location(
    "observed_b4_covariance_child_run", RUNNER_PATH
)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def test_frozen_constants_and_protocol_binding() -> None:
    assert runner.REGISTERED_SEED == 2100
    assert runner.MAXIMUM_SITES == 250_000
    assert runner.FROZEN_STUDENT_TAU == 0.4722205736227421
    assert runner.ARMS == ("b4", "diagonal", "block3", "full", "student")
    assert runner.PRACTICAL_MARGIN == 0.01
    actual = runner.sha256_file(runner.PROTOCOL_PATH)
    assert runner.PROTOCOL_HASH == actual


def test_parent_result_and_every_parent_source_are_bound() -> None:
    evidence = runner._verify_parent_result()
    assert evidence["source_commit"] == runner.EXPECTED_PARENT_COMMIT
    assert evidence["first_failed_layer"] == "fit_boundary"
    assert evidence["inherited_all_holdout_b4_inverse"]["passed"]
    assert runner._sha256_array(evidence["b4_scores"]) == runner.EXPECTED_B4_SCORE_HASH
    assert (
        runner._sha256_array(evidence["student_scores"])
        == runner.EXPECTED_STUDENT_SCORE_HASH
    )
    registered = set(runner.REGISTERED_SOURCE_PATHS.values())
    assert set(runner.PARENT.REGISTERED_SOURCE_PATHS.values()) <= registered


def test_quality_summary_uses_three_paired_image_endpoints_and_strict_json() -> None:
    labels = np.repeat(np.arange(10), 500)
    detail = runner.DETAILS_PER_IMAGE
    scores = {
        "b4": np.zeros(5_000),
        "diagonal": np.full(5_000, 0.005 * detail),
        "block3": np.full(5_000, 0.015 * detail),
        "full": np.full(5_000, 0.03 * detail),
        "student": np.full(5_000, 0.025 * detail),
    }
    summary, contrasts = runner._quality_summary(scores, labels)
    assert summary["all_primary_pass"]
    np.testing.assert_allclose(contrasts["b4_minus_full"], 0.03)
    np.testing.assert_allclose(contrasts["block_minus_full"], 0.015)
    np.testing.assert_allclose(contrasts["student_minus_full"], 0.005)
    degrees = summary["b4_minus_full"]["degrees_of_freedom"]
    assert degrees is None or math.isfinite(degrees)
    assert summary["b4_minus_full"]["degrees_of_freedom_is_infinite"] == (
        degrees is None
    )
    json.dumps(summary, allow_nan=False)


def test_quality_exports_finite_two_sided_balanced_welch_effect_intervals() -> None:
    labels = np.repeat(np.arange(10), 500)
    detail = runner.DETAILS_PER_IMAGE
    variation = np.tile(np.linspace(-1.0, 1.0, 500), 10)
    scores = {
        "b4": np.zeros(5_000),
        "diagonal": np.zeros(5_000),
        "block3": (0.015 - 0.002 * variation) * detail,
        "full": (0.04 + 0.005 * variation) * detail,
        "student": (0.035 + 0.001 * variation) * detail,
    }
    summary, _ = runner._quality_summary(scores, labels)
    for name in ("b4_minus_full", "block_minus_full", "student_minus_full"):
        endpoint = summary[name]
        interval = endpoint["balanced_class_welch_effect_interval_95"]
        assert endpoint["effect_interval_method"] == "two_sided_balanced_class_welch_t"
        assert endpoint["effect_interval_confidence_level"] == 0.95
        assert endpoint["degrees_of_freedom"] is not None
        expected_critical = runner.stats.t.ppf(
            0.975, endpoint["degrees_of_freedom"]
        )
        assert endpoint["effect_interval_critical_value"] == pytest.approx(
            expected_critical
        )
        assert interval[0] < endpoint["mean"] < interval[1]
        assert interval[0] == pytest.approx(
            endpoint["mean"] - expected_critical * endpoint["standard_error"]
        )
        assert interval[1] == pytest.approx(
            endpoint["mean"] + expected_critical * endpoint["standard_error"]
        )
    json.dumps(summary, allow_nan=False)


def test_quality_boundary_null_fails_without_non_json_infinity() -> None:
    labels = np.repeat(np.arange(10), 500)
    scores = {arm: np.zeros(5_000) for arm in runner.ARMS}
    summary, _ = runner._quality_summary(scores, labels)
    assert not summary["all_primary_pass"]
    assert all(
        summary[name]["degrees_of_freedom"] is None
        for name in ("b4_minus_full", "block_minus_full", "student_minus_full")
    )
    assert all(
        summary[name]["balanced_class_welch_effect_interval_95"] == [0.0, 0.0]
        for name in ("b4_minus_full", "block_minus_full", "student_minus_full")
    )
    assert all(
        math.isfinite(summary[name]["effect_interval_critical_value"])
        for name in ("b4_minus_full", "block_minus_full", "student_minus_full")
    )
    json.dumps(summary, allow_nan=False)


def test_coordinate_statistics_and_exact_null_summary() -> None:
    labels = np.repeat(np.arange(10), 500)
    random = np.random.default_rng(2).normal(size=(runner.SITES_PER_IMAGE, 9))
    orthogonal, _ = np.linalg.qr(random - np.mean(random, axis=0, keepdims=True))
    sites = np.sqrt(runner.SITES_PER_IMAGE) * orthogonal
    latent = np.broadcast_to(sites, (5_000, runner.SITES_PER_IMAGE, 9)).copy()
    statistics = runner._coordinate_image_statistics(latent)
    summary = runner._coordinate_summary(statistics, labels)
    assert statistics["coordinate_mean"].shape == (5_000, 9)
    assert statistics["coordinate_second"].shape == (5_000, 9, 9)
    assert summary["mean_maximum_absolute"] < 1e-14
    assert summary["uncentered_second_moment_maximum_absolute_deviation"] < 1e-14
    assert summary["mean_maximum_simultaneous_upper"] < 1e-12
    assert (
        summary["uncentered_second_moment_maximum_simultaneous_upper"] < 1e-12
    )
    assert "derived_covariance" in summary


def test_stratum_coordinate_summary_uses_site_estimates_and_image_clusters() -> None:
    labels = np.repeat(np.arange(10), 500)
    random = np.random.default_rng(3).normal(size=(64, 9))
    orthogonal, _ = np.linalg.qr(random - np.mean(random, axis=0, keepdims=True))
    stratum_sites = np.sqrt(64.0) * orthogonal
    one_image = np.concatenate([stratum_sites for _ in range(4)], axis=0)
    assignments = np.repeat(np.arange(4), 64)
    latent = np.broadcast_to(one_image, (5_000, runner.SITES_PER_IMAGE, 9)).copy()
    strata = np.broadcast_to(assignments, (5_000, runner.SITES_PER_IMAGE)).copy()
    statistics = runner._stratum_coordinate_image_statistics(latent, strata)
    summary = runner._stratum_coordinate_summary(statistics, labels)
    assert statistics["stratum_count"].shape == (5_000, 4)
    assert summary["site_counts"] == [320_000] * 4
    assert summary["coordinate_mean_maximum_absolute"] < 1e-14
    assert summary["coordinate_mean_maximum_simultaneous_upper"] < 1e-12
    assert summary["uncentered_second_moment_maximum_absolute_deviation"] < 1e-14
    assert (
        summary["uncentered_second_moment_maximum_simultaneous_upper"] < 1e-12
    )


def test_holdout_oracle_matches_registered_formula_and_is_rejection_only() -> None:
    covariance = np.diag(np.linspace(0.8, 1.2, 9))
    root = np.linalg.cholesky(covariance)
    rows = np.concatenate((root.T, -root.T)) * np.sqrt(9.0)
    oracle = runner._global_covariance_oracle(rows)
    expected = (
        np.trace(covariance) - 9.0 - np.linalg.slogdet(covariance)[1]
    ) / 18.0
    np.testing.assert_allclose(oracle["uncentered_second_moment"], covariance)
    assert oracle["maximum_gain_nat_per_detail"] == pytest.approx(expected)
    assert oracle["role"] == "rejection_only_not_a_fitted_or_selected_arm"


def test_full_image_probe_selects_ten_lowest_record_ids_per_class() -> None:
    labels = np.repeat(np.arange(10), 20)
    record_ids = np.concatenate(
        [1000 * class_id + np.arange(20)[::-1] for class_id in range(10)]
    )
    positions = runner._probe_image_positions(record_ids, labels)
    assert positions.shape == (100,)
    for class_id in range(10):
        selected = positions[labels[positions] == class_id]
        assert len(selected) == 10
        expected = np.sort(record_ids[labels == class_id])[:10]
        np.testing.assert_array_equal(np.sort(record_ids[selected]), expected)


def test_top_radius_probe_uses_32_per_stratum_and_row_order_ties() -> None:
    gaussian = np.zeros((4 * 40, 9))
    strata = np.repeat(np.arange(4), 40)
    gaussian[:, 0] = np.tile(np.arange(40, dtype=float), 4)
    selected = runner._top_radius_site_indices(gaussian, strata)
    assert selected.shape == (128,)
    for stratum in range(4):
        rows = selected[strata[selected] == stratum]
        assert len(rows) == 32
        expected_local = np.arange(39, 7, -1)
        np.testing.assert_array_equal(rows - stratum * 40, expected_local)

    tied = np.ones((4 * 32, 9))
    tied_strata = np.repeat(np.arange(4), 32)
    tied_selected = runner._top_radius_site_indices(tied, tied_strata)
    np.testing.assert_array_equal(tied_selected, np.arange(128))


def test_time_callable_records_all_repetitions() -> None:
    result = runner._time_callable(lambda: 7.0, warmups=1, repetitions=3)
    assert result["warmups"] == 1
    assert result["repetitions"] == 3
    assert len(result["seconds"]) == 3
    assert result["checksum_range"] == [7.0, 7.0]


def test_covariance_rejection_maps_to_covariance_positivity() -> None:
    runner.RUN_CONTEXT = {
        "phase": "source_data_parent_reproduction",
        "opened_files": [],
        "first_failed_layer": "source_data_parent_reproduction",
    }
    with pytest.raises(ValueError, match="minimum eigenvalue"):
        runner._fit_covariance_arms(np.zeros((32, 9)))
    assert runner.RUN_CONTEXT["phase"] == "covariance_fit"
    assert runner.RUN_CONTEXT["first_failed_layer"] == "covariance_positivity"


def test_early_source_mismatch_maps_to_source_parent_layer(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setattr(runner, "PROTOCOL_HASH", runner.sha256_file(runner.PROTOCOL_PATH))

    def reject_source() -> None:
        raise ValueError("synthetic registered-source mismatch")

    monkeypatch.setattr(runner, "_assert_registered_sources_committed", reject_source)
    with pytest.raises(ValueError, match="registered-source mismatch"):
        runner.run(2100, tmp_path / "unused", runner.DEFAULT_DATA, "unused")
    assert runner.RUN_CONTEXT["phase"] == "argument_validation"
    assert (
        runner.RUN_CONTEXT["first_failed_layer"]
        == "source_data_parent_reproduction"
    )


def test_invalid_seed_and_existing_output_stop_before_data(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    with pytest.raises(ValueError, match="requires seed"):
        runner.run(2101, tmp_path / "wrong_seed", runner.DEFAULT_DATA, "unused")
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError):
        runner.run(2100, existing, runner.DEFAULT_DATA, "unused")


def test_main_durably_hashes_strict_json_failure(tmp_path, monkeypatch) -> None:
    output = tmp_path / "failed"
    monkeypatch.setattr(
        runner.sys,
        "argv",
        [
            str(runner.RUNNER_PATH) if hasattr(runner, "RUNNER_PATH") else "run.py",
            "--seed",
            "2101",
            "--output",
            str(output),
            "--source-commit",
            "unused",
        ],
    )
    with pytest.raises(ValueError, match="requires seed"):
        runner.main()
    failure = json.loads((output / "failure.json").read_text(encoding="utf-8"))
    json.dumps(failure, allow_nan=False)
    assert failure["first_failed_layer"] == "source_data_parent_reproduction"
    assert (output / "FAILURE.sha256").read_text(encoding="utf-8") == (
        f"{runner.sha256_file(output / 'failure.json')}  failure.json\n"
    )
