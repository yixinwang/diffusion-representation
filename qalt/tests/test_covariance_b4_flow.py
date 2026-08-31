import numpy as np
import pytest

from qalt.covariance_b4_flow import (
    DIMENSION,
    fit_covariance_b4_flow,
    standard_normal_log_prob,
)


def _exact_second_moment_rows(covariance):
    """Return 2d zero-mean rows whose second moment is covariance."""

    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    root = (eigenvectors * np.sqrt(eigenvalues)) @ eigenvectors.T
    base = np.concatenate(
        (np.sqrt(DIMENSION) * np.eye(DIMENSION), -np.sqrt(DIMENSION) * np.eye(DIMENSION))
    )
    return base @ root.T


def _cross_band_covariance(sign, rho):
    covariance = np.eye(DIMENSION)
    covariance[0:3, 3:6] = sign * rho * np.eye(3)
    covariance[3:6, 0:3] = sign * rho * np.eye(3)
    return covariance


def test_identity_is_an_exact_tie_and_parameters_are_immutable():
    rows = _exact_second_moment_rows(np.eye(DIMENSION))
    for structure, parameter_count in (
        ("diagonal", 9),
        ("block3", 18),
        ("full", 45),
    ):
        flow = fit_covariance_b4_flow(rows, structure=structure)
        np.testing.assert_allclose(flow.covariance, np.eye(DIMENSION), atol=2e-15)
        np.testing.assert_allclose(flow.whitening, np.eye(DIMENSION), atol=2e-15)
        transformed, log_det = flow.forward(rows)
        np.testing.assert_allclose(transformed, rows, atol=3e-15)
        np.testing.assert_allclose(log_det, 0.0, atol=2e-15)
        np.testing.assert_allclose(flow.log_ratio(rows), 0.0, atol=2e-14)
        assert flow.parameter_count == parameter_count
        assert not flow.covariance.flags.writeable
        assert not flow.whitening.flags.writeable
        assert not flow.coloring.flags.writeable
        assert not flow.eigenvalues.flags.writeable
        with pytest.raises(ValueError):
            flow.whitening[0, 0] = 2.0


def test_random_roundtrip_density_and_finite_difference_jacobian():
    rng = np.random.default_rng(4129)
    mixing = rng.normal(size=(DIMENSION, DIMENSION))
    covariance = mixing @ mixing.T / DIMENSION + 0.7 * np.eye(DIMENSION)
    training = rng.multivariate_normal(np.zeros(DIMENSION), covariance, size=5000)
    flow = fit_covariance_b4_flow(training, structure="full")

    values = rng.normal(size=(17, DIMENSION))
    base, forward_log_det = flow.forward(values)
    recovered, inverse_log_det = flow.inverse(base)
    np.testing.assert_allclose(recovered, values, rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(forward_log_det + inverse_log_det, 0.0, atol=1e-14)
    expected_ratio = (
        standard_normal_log_prob(base)
        + forward_log_det
        - standard_normal_log_prob(values)
    )
    np.testing.assert_allclose(flow.log_ratio(values), expected_ratio, atol=2e-14)

    point = values[0]
    step = 1.0e-6
    jacobian = np.empty((DIMENSION, DIMENSION))
    for column in range(DIMENSION):
        displacement = np.zeros(DIMENSION)
        displacement[column] = step
        plus, _ = flow.forward(point + displacement)
        minus, _ = flow.forward(point - displacement)
        jacobian[:, column] = (plus - minus) / (2.0 * step)
    np.testing.assert_allclose(jacobian, flow.whitening, rtol=2e-9, atol=2e-9)
    sign, numerical_log_det = np.linalg.slogdet(jacobian)
    assert sign == 1.0
    np.testing.assert_allclose(numerical_log_det, forward_log_det[0], atol=3e-9)


def test_diagonal_and_block3_masks_are_exact_controls():
    covariance = _cross_band_covariance(+1.0, 0.3)
    covariance[0, 1] = covariance[1, 0] = 0.15
    covariance[0, 7] = covariance[7, 0] = 0.10
    rows = _exact_second_moment_rows(covariance)
    diagonal = fit_covariance_b4_flow(rows, structure="diagonal")
    block = fit_covariance_b4_flow(rows, structure="block3")
    full = fit_covariance_b4_flow(rows, structure="full")
    block_mask = np.arange(DIMENSION)[:, None] // 3 == np.arange(DIMENSION)[None, :] // 3
    diagonal_mask = np.eye(DIMENSION, dtype=bool)

    np.testing.assert_allclose(diagonal.covariance[~diagonal_mask], 0.0, atol=1e-15)
    np.testing.assert_allclose(diagonal.whitening[~diagonal_mask], 0.0, atol=1e-15)
    np.testing.assert_allclose(block.covariance[~block_mask], 0.0, atol=1e-15)
    np.testing.assert_allclose(block.whitening[~block_mask], 0.0, atol=1e-14)
    np.testing.assert_allclose(block.coloring[~block_mask], 0.0, atol=1e-14)
    assert np.max(np.abs(full.covariance[~block_mask])) > 0.09


def test_arbitrary_batch_leading_dimensions_are_preserved():
    rows = _exact_second_moment_rows(_cross_band_covariance(+1.0, 0.2))
    flow = fit_covariance_b4_flow(rows)
    rng = np.random.default_rng(833)
    values = rng.normal(size=(2, 3, DIMENSION))
    base, log_det = flow.forward(values)
    assert base.shape == values.shape
    assert log_det.shape == values.shape[:-1]
    for index in np.ndindex(values.shape[:-1]):
        np.testing.assert_allclose(base[index], flow.whitening @ values[index])
        assert log_det[index] == flow.log_det_whitening
    recovered, inverse_log_det = flow.inverse(base)
    np.testing.assert_allclose(recovered, values, rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(log_det + inverse_log_det, 0.0, atol=1e-14)


@pytest.mark.parametrize("rho", [0.30, 0.0])
def test_same_sign_cross_band_toy_matches_analytic_advantage(rho):
    covariance = _cross_band_covariance(+1.0, rho)
    # Four independently constructed strata all carry the same dependence.
    training = np.concatenate([_exact_second_moment_rows(covariance) for _ in range(4)])
    full = fit_covariance_b4_flow(training, structure="full")
    block = fit_covariance_b4_flow(training, structure="block3")
    diagonal = fit_covariance_b4_flow(training, structure="diagonal")

    expected_per_coefficient = -np.log1p(-(rho**2)) / 6.0
    observed_per_coefficient = full.log_det_whitening / DIMENSION
    np.testing.assert_allclose(
        observed_per_coefficient, expected_per_coefficient, rtol=2e-13, atol=2e-15
    )
    # The constrained controls cannot represent the only non-identity entries.
    np.testing.assert_allclose(block.covariance, np.eye(DIMENSION), atol=2e-15)
    np.testing.assert_allclose(diagonal.covariance, np.eye(DIMENSION), atol=2e-15)
    np.testing.assert_allclose(block.log_ratio(training), 0.0, atol=3e-14)
    np.testing.assert_allclose(diagonal.log_ratio(training), 0.0, atol=3e-14)
    np.testing.assert_allclose(
        np.mean(full.log_ratio(training)) / DIMENSION,
        expected_per_coefficient,
        atol=3e-14,
    )
    if rho == 0.30:
        assert expected_per_coefficient > 0.01
    else:
        assert expected_per_coefficient == 0.0


def test_alternating_sign_is_a_global_cancellation_negative_control():
    rho = 0.30
    covariances = [
        _cross_band_covariance(+1.0, rho),
        _cross_band_covariance(-1.0, rho),
        _cross_band_covariance(+1.0, rho),
        _cross_band_covariance(-1.0, rho),
    ]
    training = np.concatenate([_exact_second_moment_rows(item) for item in covariances])
    flow = fit_covariance_b4_flow(training, structure="full")
    np.testing.assert_allclose(flow.covariance, np.eye(DIMENSION), atol=2e-15)
    np.testing.assert_allclose(flow.log_ratio(training), 0.0, atol=3e-14)


def test_eigenvalue_and_condition_checks_reject_unusable_second_moments():
    rank_deficient = np.zeros((DIMENSION, DIMENSION))
    rank_deficient[:, 0] = 1.0
    with pytest.raises(ValueError, match="minimum eigenvalue"):
        fit_covariance_b4_flow(rank_deficient)

    covariance = np.eye(DIMENSION)
    covariance[0, 0] = 2.0e6
    with pytest.raises(ValueError, match="condition number"):
        fit_covariance_b4_flow(_exact_second_moment_rows(covariance))
