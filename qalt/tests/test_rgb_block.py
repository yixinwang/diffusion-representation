import numpy as np

from qalt.rgb_block import (
    FixedShapeGSM,
    fit_determinant_one_shape,
    fit_fixed_shape_gsm,
    full_parent_masks,
    project_log_eigenvalues,
    recycle_component_normal,
    water_filled_weights,
    within_band_parent_masks,
)


def _shape() -> np.ndarray:
    raw = np.array([[1.0, 0.55, -0.15], [0.55, 1.4, 0.3], [-0.15, 0.3, 0.8]])
    determinant_scale = np.linalg.det(raw) ** (1.0 / 3.0)
    return raw / determinant_scale


def test_log_spectrum_projection_and_shape_fit_are_canonical() -> None:
    spectrum = np.array([1e-7, 2.0, 4e5])
    first = project_log_eigenvalues(spectrum)
    second = project_log_eigenvalues(19.0 * spectrum)
    np.testing.assert_allclose(first, second, atol=1e-13, rtol=1e-13)
    assert np.all(first >= 0.1 - 1e-14)
    assert np.all(first <= 10.0 + 1e-14)
    assert abs(np.prod(first) - 1.0) < 1e-12

    rng = np.random.default_rng(101)
    residual = rng.normal(size=(2_000, 3)) @ np.array([[1.0, 0.0, 0.0], [0.7, 0.6, 0.0], [0.2, -0.1, 0.3]]).T
    shape = fit_determinant_one_shape(residual)
    eigenvalues = np.linalg.eigvalsh(shape)
    assert np.all(eigenvalues > 0.0)
    assert abs(np.linalg.det(shape) - 1.0) < 1e-12
    assert eigenvalues[-1] / eigenvalues[0] <= 100.0 + 1e-10


def test_water_filling_satisfies_simplex_and_kkt_conditions() -> None:
    masses = np.array([100.0, 3.0, 0.01, 0.0])
    floor = 0.05
    weights = water_filled_weights(masses, floor)
    assert abs(np.sum(weights) - 1.0) < 1e-14
    assert np.all(weights >= floor)
    assert weights[-1] == floor
    active = weights > floor + 1e-14
    ratios = masses[active] / weights[active]
    np.testing.assert_allclose(ratios, ratios[0], atol=1e-12, rtol=1e-12)
    assert np.all(masses[~active] / floor <= ratios[0] + 1e-12)


def test_joint_density_matches_exact_scalarization_and_ablation_marginals() -> None:
    rng = np.random.default_rng(103)
    model = FixedShapeGSM(np.array([0.3, 0.7]), np.array([0.35, 1.4]), _shape())
    residual = rng.normal(size=(1_000, 3))
    scalar = model.exact_scalar_log_prob(residual)
    np.testing.assert_allclose(model.log_prob(residual), np.sum(scalar, axis=1), atol=2e-12, rtol=2e-12)
    np.testing.assert_allclose(model.product_log_prob(residual), np.sum(model.marginal_log_prob(residual), axis=1))

    zero = model.zero_correlation()
    np.testing.assert_allclose(
        zero.marginal_log_prob(residual),
        model.marginal_log_prob(residual),
        atol=2e-12,
        rtol=2e-12,
    )
    original_component_variance = model.scales[:, None] ** 2 * np.diag(model.shape)[None, :]
    zero_component_variance = zero.scales[:, None] ** 2 * np.diag(zero.shape)[None, :]
    np.testing.assert_allclose(zero_component_variance, original_component_variance, atol=1e-14, rtol=1e-14)
    assert abs(np.linalg.det(zero.shape) - 1.0) < 1e-12


def test_density_and_water_filling_match_independent_known_answers() -> None:
    cholesky = np.array([[1.0, 0.0, 0.0], [0.5, 1.0, 0.0], [-0.25, 0.3, 1.0]])
    model = FixedShapeGSM(
        np.array([0.25, 0.75]),
        np.array([1.0, 2.0]),
        cholesky @ cholesky.T,
    )
    residual = np.array([[0.4, -1.0, 0.24]])
    expected_scalar = np.array([-1.4325119468787777, -1.730209760332529, -1.4259335072897499])
    np.testing.assert_allclose(model.exact_scalar_log_prob(residual)[0], expected_scalar, atol=1e-14, rtol=0.0)
    np.testing.assert_allclose(model.log_prob(residual), [-4.588655214501056], atol=1e-14, rtol=0.0)

    expected_weights = np.array([0.7578947368421053, 0.14210526315789473, 0.1])
    np.testing.assert_allclose(water_filled_weights(np.array([80.0, 15.0, 5.0]), 0.1), expected_weights)


def test_same_dimensional_sampler_recycles_only_supplied_normals() -> None:
    rng = np.random.default_rng(107)
    base = rng.normal(size=(100_000, 3))
    weights = np.array([0.2, 0.5, 0.3])
    component, remapped = recycle_component_normal(base[:, 0], weights)
    frequencies = np.bincount(component, minlength=3) / len(component)
    np.testing.assert_allclose(frequencies, weights, atol=0.004, rtol=0.0)
    for index in range(3):
        selected = remapped[component == index]
        assert abs(np.mean(selected)) < 0.02
        assert abs(np.var(selected) - 1.0) < 0.03

    boundary_component, _ = recycle_component_normal(np.array([0.0]), np.array([0.5, 0.5]))
    np.testing.assert_array_equal(boundary_component, [0])

    one = FixedShapeGSM(np.array([1.0]), np.array([0.8]), _shape())
    expected = (base @ one.cholesky.T) * 0.8
    np.testing.assert_allclose(one.sample(base), expected, atol=2e-12, rtol=2e-12)


def test_fixed_shape_em_is_monotone_and_k1_has_closed_form_scale() -> None:
    rng = np.random.default_rng(109)
    truth = FixedShapeGSM(np.array([0.65, 0.35]), np.array([0.3, 1.3]), _shape())
    residual = truth.sample(rng.normal(size=(30_000, 3)))
    fitted, diagnostics = fit_fixed_shape_gsm(residual, truth.shape, 2)
    differences = np.diff(diagnostics.log_likelihood)
    assert np.all(differences >= -1e-7)
    assert diagnostics.converged
    assert np.all(fitted.weights >= 1e-4)
    np.testing.assert_allclose(fitted.scales, truth.scales, atol=0.08, rtol=0.0)
    np.testing.assert_allclose(fitted.weights, truth.weights, atol=0.06, rtol=0.0)

    one, one_diagnostics = fit_fixed_shape_gsm(residual, truth.shape, 1)
    whitened = np.linalg.solve(np.linalg.cholesky(truth.shape), residual.T).T
    expected_scale = np.sqrt(np.mean(np.sum(whitened**2, axis=1)) / 3.0)
    assert abs(one.scales[0] - expected_scale) < 1e-10
    assert one_diagnostics.converged


def test_parent_masks_distinguish_parallel_bands_from_global_order() -> None:
    assert within_band_parent_masks() == (
        (),
        (0,),
        (0, 1),
        (),
        (3,),
        (3, 4),
        (),
        (6,),
        (6, 7),
    )
    assert full_parent_masks(4) == ((), (0,), (0, 1), (0, 1, 2))
