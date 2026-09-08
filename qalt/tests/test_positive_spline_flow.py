import numpy as np
import pytest
from scipy.optimize._numdiff import approx_derivative
from qalt.positive_density_spline import PositiveDensitySpline, response_integral_weights
from qalt.positive_spline_flow import PositiveSplineFlow


def nonidentity_model():
    rng = np.random.default_rng(72)
    models = []
    for dimension in range(3):
        coefficients = rng.uniform(.6, 1.4, ((4+2)**dimension, 5))
        coefficients /= (coefficients @ response_integral_weights(4))[:, None]
        models.append(PositiveDensitySpline(coefficients, 4, dimension))
    return PositiveSplineFlow(((), (0,), (0, 1), (0,)), (0, 1, 2, 1), tuple(models))


def test_complete_nonidentity_gaussian_jacobian_and_roundtrip():
    model = nonidentity_model()
    source = np.random.default_rng(4).normal(size=(2, 3, 4))
    x, inverse_ld = model.decode(source)
    recovered, forward_ld = model.encode(x)
    np.testing.assert_allclose(recovered, source, atol=2e-13)
    np.testing.assert_allclose(inverse_ld + forward_ld, 0, atol=2e-13)
    expected = -.5*(4*np.log(2*np.pi) + np.sum(source**2, axis=-1))-inverse_ld
    np.testing.assert_allclose(model.log_prob(x), expected, atol=2e-13)
    z = np.array([-.34, .61, -.12, .43])
    jacobian = approx_derivative(lambda v: model.decode(v)[0], z, method='3-point')
    np.testing.assert_allclose(np.linalg.slogdet(jacobian)[1], model.decode(z)[1], atol=2e-9)
    assert np.min(np.abs(np.diag(jacobian))) > .1
    np.testing.assert_allclose(np.triu(jacobian, 1), 0, atol=1e-10)


def test_shared_loss_is_whole_array_mean_and_gap_is_weighted():
    rng = np.random.default_rng(19)
    data = rng.uniform(.01, .99, (120, 4))
    # Sites 1,2 are deliberately identical: pooling must not pretend 240 IID arrays.
    data[:, 2] = data[:, 1]
    model, info = PositiveSplineFlow.fit(data, ((), (0,), (0,), (1, 2)), (0, 1, 1, 2),
                                       bins=3, max_iterations=8)
    assert info.independent_array_count == 120
    assert info.group_site_counts == (1, 2, 1)
    assert info.group_fits[1].sample_count == 240
    weighted_loss = sum(k*f.objective for k, f in zip(info.group_site_counts, info.group_fits))/4
    np.testing.assert_allclose(weighted_loss, -np.mean(model.log_prob(data))/4, atol=1e-13)
    assert info.per_coordinate_optimization_gap == sum(k*f.frank_wolfe_gap for k,f in zip(info.group_site_counts, info.group_fits))/4
    assert info.converged == all(f.converged for f in info.group_fits)


def test_boundaries_and_causal_validation():
    model = nonidentity_model()
    assert model.log_prob(np.array([0., .4, .5, .6])) == -np.inf
    with pytest.raises(ValueError, match='open unit cube'):
        model.encode([0., .4, .5, .6])
    with pytest.raises(FloatingPointError, match='boundary'):
        model.decode([40., 0., 0., 0.])
    with pytest.raises(ValueError, match='earlier'):
        PositiveSplineFlow(((0,),), (0,), (model.models[1],))
    with pytest.raises(ValueError, match='equal context'):
        PositiveSplineFlow(((), (0,)), (0, 0), (model.models[0],))
    with pytest.raises(ValueError, match='distinct'):
        PositiveSplineFlow(((), (0, 0)), (0, 1), (model.models[0], model.models[2]))


def test_depth_batches_cross_workspace_limit_without_changing_conditional_map():
    from scipy.special import ndtr
    base = nonidentity_model()
    dimension = 512
    parents = tuple(() if i % 2 == 0 else (i-1,) for i in range(dimension))
    model = PositiveSplineFlow(parents, tuple(i % 2 for i in range(dimension)), base.models[:2])
    source = np.random.default_rng(811).normal(size=(257, dimension))
    uniforms = ndtr(source)
    expected = np.empty_like(source)
    expected[:, ::2] = base.models[0].icdf(None, uniforms[:, ::2])
    expected[:, 1::2] = base.models[1].icdf(expected[:, ::2, None], uniforms[:, 1::2])
    actual, ld = model.decode(source)
    np.testing.assert_allclose(actual, expected, atol=2e-15)
    log_density = (base.models[0].log_prob(None, expected[:, ::2]).sum(axis=1)
                   + base.models[1].log_prob(expected[:, ::2, None], expected[:, 1::2]).sum(axis=1))
    log_base = -.5*(dimension*np.log(2*np.pi)+np.sum(source**2,axis=1))
    np.testing.assert_allclose(ld, log_base-log_density, atol=2e-12)
