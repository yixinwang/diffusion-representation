import numpy as np
import pytest
from scipy.special import ndtr

from qalt.conditional_cdf import ConditionalCDFFlow


def _observed_nonlinear_data(seed=4902, count=1200):
    rng = np.random.default_rng(seed)
    first = rng.beta(2.0, 4.0, size=count)
    second = 0.05 + 0.9 * (
        0.55 * first**2 + 0.45 * rng.beta(0.8, 1.3, size=count)
    )
    third = 0.05 + 0.9 * (
        0.6 * np.sin(np.pi * first) ** 2 + 0.4 * rng.uniform(size=count)
    )
    fourth = 0.05 + 0.9 * (
        0.5 * second**2 + 0.5 * rng.beta(3.0, 1.0, size=count)
    )
    return np.column_stack((first, second, third, fourth))


def test_learned_nongaussian_roundtrip_log_density_and_batch_shapes():
    data = _observed_nonlinear_data()
    flow = ConditionalCDFFlow.fit(data, [-1, 0, 0, 1], bins=7)
    gaussian = np.random.default_rng(1439).normal(size=(2, 3, 4))
    observations, decode_log_det = flow.decode(gaussian)
    recovered, encode_log_det = flow.encode(observations)
    np.testing.assert_allclose(recovered, gaussian, rtol=2e-13, atol=3e-14)
    np.testing.assert_allclose(encode_log_det + decode_log_det, 0.0, atol=3e-13)
    expected_log_base = -0.5 * (
        4 * np.log(2 * np.pi) + np.sum(gaussian**2, axis=-1)
    )
    np.testing.assert_allclose(
        flow.log_prob(observations), expected_log_base - decode_log_det, atol=3e-13
    )
    assert observations.shape == gaussian.shape
    assert decode_log_det.shape == gaussian.shape[:-1]
    assert np.all((observations > 0.0) & (observations < 1.0))
    assert np.max(np.abs(observations - ndtr(gaussian))) > 0.1
    # Observations are the only fit input, including for the independent test rows.
    held_out = _observed_nonlinear_data(seed=1234, count=20)
    encoded, forward = flow.encode(held_out)
    reconstructed, inverse = flow.decode(encoded)
    np.testing.assert_allclose(reconstructed, held_out, atol=3e-14)
    np.testing.assert_allclose(forward + inverse, 0.0, atol=3e-12)


def test_jacobian_matches_independent_finite_difference_and_uses_every_source():
    flow = ConditionalCDFFlow.fit(_observed_nonlinear_data(), [-1, 0, 0, 1], bins=7)
    point = np.array([-0.43, 0.28, -0.32, 0.77])
    observations, analytic = flow.decode(point)
    # Derivatives are asserted away from response and interpolation knots.
    assert np.min(np.abs(observations * 7 - np.round(observations * 7))) > 1e-3
    contexts = observations[[0, 1]] * 7 - 0.5
    assert np.min(np.abs(contexts - np.round(contexts))) > 1e-3
    step = 1e-6
    jacobian = np.empty((4, 4))
    for coordinate in range(4):
        perturbation = np.eye(4)[coordinate] * step
        plus, _ = flow.decode(point + perturbation)
        minus, _ = flow.decode(point - perturbation)
        jacobian[:, coordinate] = (plus - minus) / (2 * step)
        assert plus[coordinate] != minus[coordinate]
        np.testing.assert_array_equal(plus[:coordinate], minus[:coordinate])
    sign, numeric = np.linalg.slogdet(jacobian)
    assert sign == 1.0
    assert np.all(np.diag(jacobian) > 0.0)
    np.testing.assert_allclose(numeric, analytic, atol=2e-8)
    assert np.max(np.abs(np.tril(jacobian, -1))) > 0.01


def test_context_interpolation_is_continuous_at_centers_and_count_cell_boundaries():
    data = np.array([[0.08, 0.12]] * 50 + [[0.38, 0.82]] * 60
                    + [[0.67, 0.28]] * 40 + [[0.94, 0.93]] * 30)
    flow = ConditionalCDFFlow.fit(data, [-1, 0], bins=4)
    displacement = 1e-9
    for context in [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875]:
        left, _ = flow.encode(np.array([context - displacement, 0.43]))
        right, _ = flow.encode(np.array([context + displacement, 0.43]))
        np.testing.assert_allclose(left, right, atol=1e-6)
    # At c=0.25, interpolation must average the two neighboring count rows.
    root_density = 4 * flow.probabilities[0][0, 1]
    conditional_density = 4 * 0.5 * (
        flow.probabilities[1][0, 1] + flow.probabilities[1][1, 1]
    )
    np.testing.assert_allclose(
        np.exp(flow.log_prob(np.array([0.25, 0.43]))),
        root_density * conditional_density,
        atol=1e-13,
    )


def test_joint_density_integrates_to_one_by_aligned_midpoint_quadrature():
    flow = ConditionalCDFFlow.fit(_observed_nonlinear_data()[:, :2], [-1, 0], bins=5)
    # Half-bin subdivisions align both response boundaries and context centers.
    grid = (np.arange(2 * flow.bins) + 0.5) / (2 * flow.bins)
    first, second = np.meshgrid(grid, grid, indexing="ij")
    points = np.column_stack((first.ravel(), second.ravel()))
    integral = np.mean(np.exp(flow.log_prob(points)))
    np.testing.assert_allclose(integral, 1.0, atol=4e-15)


def test_empty_context_cells_are_positive_and_no_training_array_is_retained():
    data = np.array([[0.06, 0.15], [0.07, 0.16], [0.09, 0.82]])
    flow = ConditionalCDFFlow.fit(data, [-1, 0], bins=8)
    np.testing.assert_allclose(flow.probabilities[1][1:], 1.0 / 8.0, atol=0.0)
    np.testing.assert_allclose(flow.probabilities[1][0].sum(), 1.0, atol=1e-15)
    probe = np.array([0.9, 0.47])
    expected = flow.log_prob(probe).copy()
    data[:] = 0.5
    np.testing.assert_array_equal(flow.log_prob(probe), expected)
    assert np.isfinite(expected)
    assert not flow.parents.flags.writeable
    assert all(not table.flags.writeable for table in flow.probabilities)
    with pytest.raises(ValueError):
        flow.probabilities[1][0, 0] = 0.0


@pytest.mark.parametrize("parents", [[-1, 1], [-1, 2], [-2, 0], [-1.0, 0.0], [-1]])
def test_invalid_graph_is_rejected(parents):
    with pytest.raises(ValueError, match="parent"):
        ConditionalCDFFlow.fit(np.full((3, 2), 0.5), parents)


@pytest.mark.parametrize("bins", [0, 1, 2.5, True])
def test_invalid_bin_counts_are_rejected(bins):
    with pytest.raises(ValueError, match="bins"):
        ConditionalCDFFlow.fit(np.full((3, 2), 0.5), [-1, 0], bins=bins)


def test_unit_cube_support_input_validation_and_explicit_tail_failure():
    for data in [np.array([[0.0]]), np.array([[1.0]]), np.array([[np.nan]])]:
        with pytest.raises(ValueError, match="unit cube"):
            ConditionalCDFFlow.fit(data, [-1])
    with pytest.raises(ValueError, match="nonempty"):
        ConditionalCDFFlow.fit(np.empty((0, 1)), [-1])
    flow = ConditionalCDFFlow.fit(np.array([[0.2], [0.4], [0.6]]), [-1])
    np.testing.assert_array_equal(
        flow.log_prob(np.array([[-0.1], [0.0], [1.0], [1.1]])), -np.inf
    )
    with pytest.raises(ValueError, match="strictly inside"):
        flow.encode(np.array([0.0]))
    with pytest.raises(ValueError, match="floating-point"):
        flow.decode(np.array([40.0]))
    with pytest.raises(ValueError, match="finite"):
        flow.log_prob(np.array([np.nan]))
    with pytest.raises(ValueError, match="final dimension"):
        flow.decode(np.array([0.0, 0.0]))
    values, determinant = flow.decode(np.empty((0, 1)))
    assert values.shape == (0, 1)
    assert determinant.shape == (0,)

