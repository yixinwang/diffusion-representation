import numpy as np
import pytest
from scipy.optimize import minimize

from qalt.positive_density_spline import response_integral_weights, _joint_features, bounded_row_linear_oracle
from qalt.positive_spline_optimizer import (
    weighted_row_projection, fit_positive_density_spline_accelerated,
    _similar_triangles_point, _similar_triangles_update,
)


def test_exact_row_projection_matches_independent_slsqp_and_kkt():
    rng = np.random.default_rng(204)
    weights = response_integral_weights(4)
    values = rng.normal(size=(16, 5)) * 3
    actual, info = weighted_row_projection(values, weights, lower=.175, upper=3.3)
    for row, result, multiplier in zip(values, actual, info.multipliers):
        reference = minimize(lambda a: .5 * np.sum((a - row)**2), np.ones(5),
            jac=lambda a: a - row, bounds=[(.175, 3.3)] * 5,
            constraints=[{'type': 'eq', 'fun': lambda a: weights @ a - 1, 'jac': lambda a: weights}],
            method='SLSQP', options={'ftol': 1e-12, 'maxiter': 300})
        assert reference.success
        np.testing.assert_allclose(result, reference.x, atol=5e-8, rtol=1e-8)
        stationarity = result - row + multiplier * weights
        assert np.all(stationarity[result == .175] >= -1e-12)
        assert np.all(stationarity[result == 3.3] <= 1e-12)
        np.testing.assert_allclose(stationarity[(result > .175) & (result < 3.3)], 0, atol=1e-12)
    np.testing.assert_allclose(actual @ weights, 1, atol=2e-15)
    assert info.row_integral_max_error < 2e-15 and info.kkt_max_residual < 2e-14
    repeated, _ = weighted_row_projection(actual, weights, .175, 3.3)
    np.testing.assert_allclose(repeated, actual, atol=2e-15)


def test_projection_plateau_and_nonexpansiveness():
    weights = response_integral_weights(4)
    # Upper coefficients of total weight .25 and all other coefficients lower
    # already satisfy normalization: the multiplier can lie in an interval.
    vertex = np.array([[.25, 3.25, .25, .25, .25]])
    outward = np.array([[-10., 10., -10., -10., -10.]])
    projected, _ = weighted_row_projection(outward, weights, .25, 3.25)
    np.testing.assert_array_equal(projected, vertex)
    rng = np.random.default_rng(205)
    first = rng.normal(size=(10,5))
    second = rng.normal(size=(10,5))
    p1, _ = weighted_row_projection(first, weights)
    p2, _ = weighted_row_projection(second, weights)
    assert np.all(np.linalg.norm(p1-p2,axis=1) <= np.linalg.norm(first-second,axis=1)+1e-12)
    with pytest.raises(ValueError): weighted_row_projection(first, weights * 2)


def test_similar_triangles_potential_for_known_quadratic_optimum():
    rng = np.random.default_rng(206)
    weights = response_integral_weights(3)
    target, _ = weighted_row_projection(rng.normal(size=(2,4)), weights)
    matrix = rng.normal(size=(8,8))
    hessian = matrix.T @ matrix + np.eye(8)
    lipschitz = float(np.linalg.eigvalsh(hessian)[-1])
    def objective(a):
        residual = (a-target).ravel()
        return .5 * residual @ hessian @ residual
    x = np.ones((2,4)); z = x.copy(); accumulated = 0.
    radius_squared = np.sum((x-target)**2)
    potential = .5 * radius_squared
    for k in range(1,41):
        y,alpha,next_weight = _similar_triangles_point(x,z,accumulated,lipschitz)
        assert np.all(y>=.25-1e-14) and np.all(y<=4+1e-14)
        np.testing.assert_allclose(y@weights,1,atol=2e-15)
        gradient = (hessian @ (y-target).ravel()).reshape(2,4)
        x,z,_ = _similar_triangles_update(x,z,gradient,alpha,next_weight,weights,.25,4.)
        accumulated = next_weight
        np.testing.assert_allclose(lipschitz*alpha**2,accumulated,rtol=1e-14)
        next_potential = accumulated*objective(x)+.5*np.sum((z-target)**2)
        assert next_potential <= potential + 2e-13
        assert objective(x) <= 2*lipschitz*radius_squared/(k+1)**2+1e-12
        potential = next_potential


def test_fabricated_likelihood_final_gap_independent_optimum_and_charged_counts():
    response = ((np.arange(160)+.5)/160)**2
    model, info = fit_positive_density_spline_accelerated(None,response,bins=4,
        max_iterations=90,gap_tolerance=1e-5,lower=.175,upper=3.3)
    features = _joint_features(np.empty((len(response),0)),response,4)
    density = features @ model.coefficients.ravel()
    objective = -np.log(density).mean()
    gradient = np.asarray(features.T @ (-1/(len(response)*density))).reshape(1,5)
    vertex = bounded_row_linear_oracle(gradient,response_integral_weights(4),.175,3.3)
    gap = np.sum(gradient*(model.coefficients-vertex))
    np.testing.assert_allclose(objective,info.objective,atol=1e-14)
    np.testing.assert_allclose(gap,info.frank_wolfe_gap,atol=1e-14)
    weights = response_integral_weights(4)
    optimum = minimize(lambda a:-np.log(features@a).mean(),np.ones(5),
        jac=lambda a:np.asarray(features.T@(-1/(len(response)*(features@a)))),
        bounds=[(.175,3.3)]*5,
        constraints=[{'type':'eq','fun':lambda a:weights@a-1,'jac':lambda a:weights}],
        method='SLSQP',options={'ftol':1e-12,'maxiter':500})
    assert optimum.success
    assert -1e-10 <= info.objective-optimum.fun <= info.frank_wolfe_gap+1e-10
    assert np.all(np.diff(info.incumbent_objective_trace)<=0)
    assert info.converged == (info.frank_wolfe_gap<=info.requested_gap)
    assert info.final_gap_recomputed
    assert info.projection_calls == info.iterations == len(info.gradient_step_weights)
    assert info.projected_row_count == info.iterations
    assert info.gradient_evaluations == info.transpose_feature_products == info.linear_minimizer_calls
    assert info.forward_feature_products == info.objective_evaluations == 1+info.gradient_evaluations+info.iterations
    assert len(info.gradient_objective_trace)==info.gradient_evaluations-1
    assert info.maximum_evaluation_feasibility_error < 2e-12


def test_conditional_unused_rows_feasible_lipschitz_bound_and_capped_failure():
    # A singular empirical design: all observations have one constant context.
    response = ((np.arange(60)+.5)/60)**1.4
    context = np.full((60,1),.2)
    model, info = fit_positive_density_spline_accelerated(context,response,bins=4,
        max_iterations=12,gap_tolerance=1e-12)
    features = _joint_features(context,response,4)
    empirical_hessian = (features.T @ features).toarray()/(len(response)*.25**2)
    assert np.linalg.eigvalsh(empirical_hessian)[-1] <= info.lipschitz_constant+1e-12
    np.testing.assert_allclose(model.coefficients@response_integral_weights(4),1,atol=2e-15)
    assert model.coefficients.min()>=.25-1e-14 and model.coefficients.max()<=4+1e-14
    assert info.maximum_evaluation_feasibility_error < 2e-12
    assert info.projected_row_count == 6*info.iterations
    assert not info.converged and info.termination=='projected_update_cap'
    _, zero = fit_positive_density_spline_accelerated(None,response,bins=4,max_iterations=0)
    assert zero.iterations==0 and zero.projection_calls==0 and zero.gradient_evaluations==1
    assert not zero.converged and zero.final_gap_recomputed


def test_known_uniform_optimum_stops_by_recomputed_gap_without_projection():
    response = np.repeat(np.arange(5)/4,[1,2,2,2,1])
    model, info = fit_positive_density_spline_accelerated(None,response,bins=4,max_iterations=100)
    assert info.converged and info.termination=='gap_satisfied'
    assert info.frank_wolfe_gap<=1e-6 and info.iterations==0
    assert info.gradient_evaluations==2 and info.linear_minimizer_calls==2
    np.testing.assert_array_equal(model.coefficients,np.ones((1,5)))
