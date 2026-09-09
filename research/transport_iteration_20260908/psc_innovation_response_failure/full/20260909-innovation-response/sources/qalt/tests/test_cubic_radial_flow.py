import numpy as np
import pytest

from qalt.cubic_radial_flow import (
    band_energy_correlation,
    best_affine_variance,
    forward,
    inverse,
    isotropic_normal_log_prob,
    jensen_gap_lower_bound,
    log_prob,
    standard_normal_log_prob,
)


def test_forward_inverse_and_log_determinants_cancel() -> None:
    rng = np.random.default_rng(301)
    base = rng.normal(size=(1_000, 9))
    for coefficient, scale in ((0.0, 1.0), (0.03, 0.7), (0.038, 1.4)):
        values, forward_log_det = forward(base, coefficient, scale)
        recovered, inverse_log_det = inverse(values, coefficient, scale)
        np.testing.assert_allclose(recovered, base, atol=2e-14, rtol=2e-14)
        np.testing.assert_allclose(inverse_log_det, -forward_log_det, atol=2e-14, rtol=2e-14)


def test_log_determinant_matches_numerical_jacobian() -> None:
    base = np.array([0.1, -0.3, 0.7, 0.4, -0.2, 0.9, 0.6, -0.8, 0.5])
    coefficient = 0.038
    scale = 1.2
    _, analytic = forward(base, coefficient, scale)
    epsilon = 1e-6
    jacobian = np.empty((9, 9))
    for coordinate in range(9):
        offset = np.zeros(9)
        offset[coordinate] = epsilon
        plus, _ = forward(base + offset, coefficient, scale)
        minus, _ = forward(base - offset, coefficient, scale)
        jacobian[:, coordinate] = (plus - minus) / (2.0 * epsilon)
    numerical = np.linalg.slogdet(jacobian)[1]
    np.testing.assert_allclose(numerical, analytic, atol=1e-8, rtol=0.0)


def test_density_is_normalized_change_of_variables() -> None:
    rng = np.random.default_rng(303)
    base = rng.normal(size=(2_000, 9))
    values, forward_log_det = forward(base, 0.03, 0.8)
    np.testing.assert_allclose(
        log_prob(values, 0.03, 0.8),
        standard_normal_log_prob(base) - forward_log_det,
        atol=3e-13,
        rtol=3e-13,
    )
    variance = best_affine_variance(0.03, 0.8)
    assert np.all(np.isfinite(isotropic_normal_log_prob(values, variance)))


def test_checked_bounds_and_energy_correlations() -> None:
    np.testing.assert_allclose(
        [jensen_gap_lower_bound(0.03), jensen_gap_lower_bound(0.038)],
        [0.012359877392783339, 0.017157347240907506],
        atol=2e-15,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        [band_energy_correlation(0.03), band_energy_correlation(0.038)],
        [0.23476090096287944, 0.26861655588755784],
        atol=2e-15,
        rtol=0.0,
    )
    assert jensen_gap_lower_bound(0.0138) < 0.01
    assert jensen_gap_lower_bound(0.0) == 0.0
    assert band_energy_correlation(0.0) == 0.0


def test_direction_is_unchanged_and_invalid_parameters_are_rejected() -> None:
    rng = np.random.default_rng(305)
    base = rng.normal(size=(100, 9))
    values, _ = forward(base, 0.03, 1.7)
    base_direction = base / np.linalg.norm(base, axis=1, keepdims=True)
    value_direction = values / np.linalg.norm(values, axis=1, keepdims=True)
    np.testing.assert_allclose(value_direction, base_direction, atol=2e-15, rtol=2e-15)

    for coefficient in (-0.01, np.nan, np.inf):
        with pytest.raises(ValueError):
            forward(base, coefficient)
    for scale in (0.0, -1.0, np.nan, np.inf):
        with pytest.raises(ValueError):
            inverse(base, 0.03, scale)
    with pytest.raises(ValueError):
        forward(np.empty((3, 0)), 0.03)
    with pytest.raises(ValueError):
        log_prob(np.full((2, 9), np.nan), 0.03)
