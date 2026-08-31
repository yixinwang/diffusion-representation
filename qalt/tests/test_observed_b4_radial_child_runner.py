import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from qalt.rgb_block import FixedShapeGSM
from qalt.observed_block import detail_to_blocks, image_haar_inverse
from qalt.observed_routing import image_haar, paired_dequantize


RUNNER_PATH = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "observed_b4_radial_child"
    / "run.py"
)
SPEC = importlib.util.spec_from_file_location("observed_b4_radial_child_run", RUNNER_PATH)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def _model() -> SimpleNamespace:
    mixture = FixedShapeGSM(
        np.array([0.25, 0.75]),
        np.array([0.3, 1.2]),
        np.eye(3),
    )
    return SimpleNamespace(
        mixtures=tuple(tuple(mixture for _ in range(4)) for _ in range(3))
    )


def test_frozen_protocol_hash_and_constants() -> None:
    assert runner.sha256_file(runner.PROTOCOL_PATH) == runner.PROTOCOL_HASH
    assert runner.REGISTERED_SEED == 2100
    assert runner.DETAILS_PER_IMAGE == 2_304
    np.testing.assert_allclose(runner.PIT_GRID, np.linspace(0.05, 0.95, 19))


def test_grouped_b4_coordinate_roundtrip_and_density_parity() -> None:
    rng = np.random.default_rng(701)
    residual = rng.normal(size=(2_000, 3, 3))
    strata = np.arange(len(residual)) % 4
    gaussian, log_det, b4_log_prob, returned, labels = runner._coordinates_from_residual(
        _model(), residual, strata
    )
    recovered, inverse_log_det = runner._b4_inverse_residual(_model(), gaussian, strata)
    np.testing.assert_allclose(returned, residual)
    np.testing.assert_array_equal(labels, strata)
    np.testing.assert_allclose(recovered, residual, atol=2e-12, rtol=2e-12)
    np.testing.assert_allclose(inverse_log_det, -log_det, atol=2e-12, rtol=2e-12)
    standard = -0.5 * (9 * np.log(2.0 * np.pi) + np.sum(gaussian * gaussian, axis=1))
    np.testing.assert_allclose(b4_log_prob, standard + log_det, atol=3e-13, rtol=3e-13)


def test_latent_sufficient_statistics_have_registered_shapes() -> None:
    rng = np.random.default_rng(703)
    latent = rng.normal(size=(7, 256, 9))
    result = runner._latent_image_statistics(latent)
    assert result["pit_fraction"].shape == (7, 19)
    assert result["direction_second"].shape == (7, 9, 9)
    assert result["direction_fourth"].shape == (7, 9, 9)
    assert result["share_first"].shape == (7, 3)
    assert result["share_second"].shape == (7, 3, 3)
    assert result["energy_sum"].shape == (7, 3)
    assert result["energy_outer_sum"].shape == (7, 3, 3)


def test_registered_radial_maps_preserve_direction_and_band_shares() -> None:
    rng = np.random.default_rng(704)
    values = rng.normal(size=(7, 256, 9))
    statistics = {
        "b4": runner._latent_image_statistics(values),
        "cubic": runner._latent_image_statistics(runner.cubic_to_base(values, 0.03)[0]),
        "student": runner._latent_image_statistics(runner.student_to_base(values, 0.15)[0]),
    }
    result = runner._radial_invariance_summary(statistics)
    assert result["passed"]
    assert result["maximum_absolute_difference"] <= runner.RADIAL_INVARIANCE_TOLERANCE
    for candidate in (
        runner.cubic_to_base(values, 0.03)[0],
        runner.student_to_base(values, 0.15)[0],
    ):
        sitewise = runner._sitewise_radial_invariance(values, candidate)
        assert max(sitewise.values()) <= runner.RADIAL_INVARIANCE_TOLERANCE


def test_frozen_float32_haar_roundtrip_uses_registered_tolerance() -> None:
    rng = np.random.default_rng(705)
    images = rng.integers(0, 256, size=(4, 3, 32, 32), dtype=np.uint8)
    record_ids = np.array([11, 29, 47, 83], dtype=np.int64)
    values = paired_dequantize(images, record_ids, runner.REGISTERED_SEED)
    coarse, detail = image_haar(values)
    restored = image_haar_inverse(coarse, detail_to_blocks(detail))
    error = float(np.max(np.abs(restored - values)))
    assert error <= runner.HAAR_ROUNDTRIP_TOLERANCE


def test_exact_null_diagnostics_pass_clustered_limits() -> None:
    labels = np.repeat(np.arange(10), 500)
    count = np.full(5_000, 256)
    direction_second = np.broadcast_to(np.eye(9) / 9.0, (5_000, 9, 9)).copy()
    direction_fourth_target = np.full((9, 9), 1.0 / 99.0)
    np.fill_diagonal(direction_fourth_target, 1.0 / 33.0)
    share_second_target = np.full((3, 3), 1.0 / 11.0)
    np.fill_diagonal(share_second_target, 5.0 / 33.0)
    energy_sum = np.full((5_000, 3), 3.0 * 256)
    energy_outer = np.broadcast_to(
        np.array([[15.0, 9.0, 9.0], [9.0, 15.0, 9.0], [9.0, 9.0, 15.0]]) * 256,
        (5_000, 3, 3),
    ).copy()
    summary = runner._diagnostic_summary(
        {
            "pit_fraction": np.broadcast_to(runner.PIT_GRID, (5_000, 19)).copy(),
            "direction_second": direction_second,
            "direction_fourth": np.broadcast_to(
                direction_fourth_target, (5_000, 9, 9)
            ).copy(),
            "share_first": np.full((5_000, 3), 1.0 / 3.0),
            "share_second": np.broadcast_to(
                share_second_target, (5_000, 3, 3)
            ).copy(),
            "energy_sum": energy_sum,
            "energy_outer_sum": energy_outer,
            "count": count,
        },
        labels,
    )
    assert summary["radial_pit"]["maximum_simultaneous_upper"] < 1e-12
    assert summary["angular"]["maximum_simultaneous_upper"] < 1e-12
    assert summary["band_energy_share"]["maximum_simultaneous_upper"] < 1e-12
    assert (
        summary["latent_band_energy_correlation"][
            "maximum_simultaneous_upper_absolute"
        ]
        < 1e-12
    )


def test_quality_summary_uses_paired_image_units_and_frozen_margins() -> None:
    labels = np.repeat(np.arange(10), 500)
    scores = {
        "b4": np.zeros(5_000),
        "cubic": np.full(5_000, 0.02 * runner.DETAILS_PER_IMAGE),
        "student": np.full(5_000, 0.015 * runner.DETAILS_PER_IMAGE),
    }
    summary, contrasts = runner._quality_summary(scores, labels)
    assert summary["all_primary_pass"]
    np.testing.assert_allclose(contrasts["b4_minus_cubic"], 0.02)
    np.testing.assert_allclose(contrasts["student_minus_cubic"], 0.005)
    assert summary["b4_minus_cubic"]["degrees_of_freedom"] is None
    assert summary["b4_minus_cubic"]["degrees_of_freedom_is_infinite"]
    json.dumps(summary, allow_nan=False)


def test_boundary_null_quality_summary_is_strict_json() -> None:
    labels = np.repeat(np.arange(10), 500)
    scores = {arm: np.zeros(5_000) for arm in runner.ARMS}
    summary, _ = runner._quality_summary(scores, labels)
    assert not summary["all_primary_pass"]
    assert summary["b4_minus_cubic"]["degrees_of_freedom"] is None
    assert summary["student_minus_cubic"]["degrees_of_freedom"] is None
    json.dumps(summary, allow_nan=False)


def test_invalid_seed_and_existing_output_are_rejected(tmp_path) -> None:
    with pytest.raises(ValueError, match="requires seed"):
        runner.run(2101, tmp_path / "wrong_seed", runner.DEFAULT_DATA, "unused")
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError):
        runner.run(2100, existing, runner.DEFAULT_DATA, "unused")


def test_benchmark_summary_uses_all_repetitions() -> None:
    result = runner._time_callable(lambda: 3.0, warmups=1, repetitions=3)
    assert result["warmups"] == 1
    assert result["repetitions"] == 3
    assert len(result["seconds"]) == 3
    assert result["checksum_range"] == [3.0, 3.0]
