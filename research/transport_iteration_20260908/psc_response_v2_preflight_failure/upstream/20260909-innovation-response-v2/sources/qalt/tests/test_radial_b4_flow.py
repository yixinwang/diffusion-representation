import numpy as np
import pytest

from qalt.cubic_radial_flow import forward as raw_cubic_forward
from qalt.radial_b4_flow import (
    covariance_normalized_cubic_scale,
    cubic_from_base,
    cubic_log_ratio,
    cubic_to_base,
    fit_cubic_parameter,
    fit_student_parameter,
    gaussianize_gsm,
    invert_gaussianized_gsm,
    student_from_base,
    student_log_prob,
    student_log_ratio,
    student_to_base,
)
from qalt.rgb_block import FixedShapeGSM


def _mixture() -> FixedShapeGSM:
    cholesky = np.array([[1.0, 0.0, 0.0], [0.4, 0.8, 0.0], [-0.2, 0.3, 1.25]])
    shape = cholesky @ cholesky.T
    assert abs(np.linalg.det(shape) - 1.0) < 1e-14
    return FixedShapeGSM(np.array([0.2, 0.5, 0.3]), np.array([0.1, 0.7, 1.8]), shape)


def test_gsm_gaussianizer_roundtrip_density_and_extreme_radii() -> None:
    mixture = _mixture()
    direction = np.array([1.0, -2.0, 0.5])
    direction /= np.linalg.norm(direction)
    radii = np.array([0.0, 1e-12, 0.01, 0.2, 1.0, 5.0, 15.0, 30.0, 60.0])
    whitened = radii[:, None] * direction
    residual = whitened @ mixture.cholesky.T
    gaussian, forward_log_det = gaussianize_gsm(mixture, residual)
    recovered, inverse_log_det = invert_gaussianized_gsm(mixture, gaussian)
    np.testing.assert_allclose(recovered, residual, atol=2e-12, rtol=2e-12)
    np.testing.assert_allclose(inverse_log_det, -forward_log_det, atol=3e-12, rtol=3e-12)

    standard_log_prob = -0.5 * (
        3 * np.log(2.0 * np.pi) + np.sum(gaussian * gaussian, axis=1)
    )
    np.testing.assert_allclose(
        mixture.log_prob(residual),
        standard_log_prob + forward_log_det,
        atol=2e-13,
        rtol=2e-13,
    )

    extreme_gaussian = np.array([[38.0, 0.0, 0.0]])
    extreme_residual, _ = invert_gaussianized_gsm(mixture, extreme_gaussian)
    extreme_recovered, _ = gaussianize_gsm(mixture, extreme_residual)
    np.testing.assert_allclose(extreme_recovered, extreme_gaussian, atol=2e-11, rtol=2e-11)


def test_covariance_normalized_cubic_nests_gaussian_and_roundtrips() -> None:
    rng = np.random.default_rng(601)
    base = rng.normal(size=(20_000, 9))
    identity_values, identity_log_det = cubic_from_base(base, 0.0)
    np.testing.assert_array_equal(identity_values, base)
    np.testing.assert_array_equal(identity_log_det, np.zeros(len(base)))
    np.testing.assert_array_equal(cubic_log_ratio(base, 0.0), np.zeros(len(base)))

    values, forward_log_det = cubic_from_base(base, 0.03)
    recovered, inverse_log_det = cubic_to_base(values, 0.03)
    np.testing.assert_allclose(recovered, base, atol=2e-14, rtol=2e-14)
    np.testing.assert_allclose(inverse_log_det, -forward_log_det, atol=2e-14, rtol=2e-14)
    np.testing.assert_allclose(np.var(values, axis=0), np.ones(9), atol=0.05, rtol=0.0)


def test_cubic_training_search_recovers_aligned_parameter() -> None:
    rng = np.random.default_rng(603)
    base = rng.normal(size=(80_000, 9))
    values, _ = cubic_from_base(base, 0.03)
    fitted = fit_cubic_parameter(values)
    assert abs(fitted.parameter - 0.03) < 0.004
    assert fitted.objective > 0.01
    assert fitted.evaluations >= 101


def test_student_radial_flow_roundtrip_density_and_direction() -> None:
    rng = np.random.default_rng(605)
    base = rng.normal(size=(30_000, 9))
    tau = 1.0 / 7.0
    values, forward_log_det = student_from_base(base, tau)
    recovered, inverse_log_det = student_to_base(values, tau)
    np.testing.assert_allclose(recovered, base, atol=2e-11, rtol=2e-11)
    np.testing.assert_allclose(inverse_log_det, -forward_log_det, atol=2e-11, rtol=2e-11)
    np.testing.assert_allclose(
        student_log_prob(values, tau),
        -0.5 * (9 * np.log(2.0 * np.pi) + np.sum(base * base, axis=1)) - forward_log_det,
        atol=3e-11,
        rtol=3e-11,
    )
    base_direction = base / np.linalg.norm(base, axis=1, keepdims=True)
    value_direction = values / np.linalg.norm(values, axis=1, keepdims=True)
    np.testing.assert_allclose(value_direction, base_direction, atol=2e-14, rtol=2e-14)
    assert np.max(np.abs(np.var(values, axis=0) - 1.0)) < 0.12


@pytest.mark.parametrize(
    ("degrees", "extreme_value_radius"),
    [(2.1, 1e135), (7.0, 1e40)],
)
def test_student_radial_flow_origin_and_extreme_tail_roundtrips(
    degrees: float,
    extreme_value_radius: float,
) -> None:
    direction = np.arange(1.0, 10.0)
    direction /= np.linalg.norm(direction)
    base_radii = np.array([0.0, 1e-12, 0.25, 3.0, 10.0, 20.0, 28.0, 30.0, 37.0])
    base = base_radii[:, None] * direction

    values, forward_log_det = student_from_base(base, 1.0 / degrees)
    recovered, inverse_log_det = student_to_base(values, 1.0 / degrees)
    assert np.all(np.isfinite(values))
    np.testing.assert_array_equal(values[0], np.zeros(9))
    np.testing.assert_allclose(recovered, base, atol=2e-12, rtol=2e-13)
    np.testing.assert_allclose(inverse_log_det, -forward_log_det, atol=2e-12, rtol=2e-13)

    value_radii = np.array([0.0, 1e-12, 0.25, 3.0, 1e8, extreme_value_radius])
    target = value_radii[:, None] * direction
    radius_squared = extreme_value_radius**2
    assert radius_squared / (radius_squared + degrees - 2.0) == 1.0
    encoded, target_inverse_log_det = student_to_base(target, 1.0 / degrees)
    rebuilt, target_forward_log_det = student_from_base(encoded, 1.0 / degrees)
    assert np.all(np.isfinite(encoded))
    np.testing.assert_array_equal(encoded[0], np.zeros(9))
    np.testing.assert_allclose(rebuilt, target, atol=2e-12, rtol=5e-13)
    np.testing.assert_allclose(
        target_forward_log_det,
        -target_inverse_log_det,
        atol=2e-12,
        rtol=2e-13,
    )


def test_student_training_search_beats_gaussian_on_student_target() -> None:
    rng = np.random.default_rng(607)
    base = rng.normal(size=(60_000, 9))
    values, _ = student_from_base(base, 1.0 / 8.0)
    fitted = fit_student_parameter(values)
    assert fitted.parameter > 0.0
    assert abs(1.0 / fitted.parameter - 8.0) < 2.0
    assert fitted.objective > 0.005


def test_invalid_inputs_and_parameters_are_rejected() -> None:
    mixture = _mixture()
    with pytest.raises(ValueError):
        gaussianize_gsm(mixture, np.zeros((3, 4)))
    with pytest.raises(ValueError):
        invert_gaussianized_gsm(mixture, np.zeros((3, 3)), iterations=0)
    with pytest.raises(ValueError):
        student_log_ratio(np.zeros((3, 9)), -0.1)
    with pytest.raises(ValueError):
        student_from_base(np.zeros((3, 9)), 0.5)
    with pytest.raises(ValueError):
        cubic_to_base(np.zeros((3, 8)), 0.03)


def test_raw_cubic_scale_matches_registered_covariance_normalization() -> None:
    rng = np.random.default_rng(609)
    base = rng.normal(size=(100, 9))
    expected, _ = raw_cubic_forward(base, 0.038, covariance_normalized_cubic_scale(0.038))
    actual, _ = cubic_from_base(base, 0.038)
    np.testing.assert_array_equal(actual, expected)


def test_search_refines_boundary_cells_and_rejects_hidden_edge_peak() -> None:
    result = __import__("qalt.radial_b4_flow", fromlist=["_grid_refine_maximum"])._grid_refine_maximum(
        lambda value: 3.0 - 10.0 * (value - 0.9) ** 2,
        np.array([0.0, 0.5, 1.0]),
    )
    assert abs(result.parameter - 0.9) < 1e-6
    assert result.objective > 2.999999
