import numpy as np
import pytest
from scipy.optimize import linprog, minimize
from qalt.positive_density_spline import (
    PositiveDensitySpline, response_integral_weights,
    bounded_row_linear_oracle, fit_positive_density_spline,
    _context_basis, _joint_features,
)


def model(dimension=2,bins=6):
    rows=(bins+2)**dimension
    amplitude=.6*np.sin(np.arange(rows)*1.3+.2)
    profile=np.cos(2*np.pi*np.arange(bins+1)/bins)
    profile=profile-response_integral_weights(bins)@profile
    return PositiveDensitySpline(1+amplitude[:,None]*profile[None],bins,dimension)


def test_positive_partition_row_integrals_and_conditional_density_quadrature():
    rng=np.random.default_rng(191)
    for dimension in range(3):
        m=model(dimension)
        context=rng.uniform(size=(12,dimension))
        basis=_context_basis(context,m.bins)
        np.testing.assert_allclose(np.asarray(basis.sum(axis=1)).ravel(),1,atol=3e-16)
        assert np.all(basis.data>=0) and basis.nnz<=len(context)*3**dimension
        np.testing.assert_allclose(m.coefficients@response_integral_weights(m.bins),1,atol=3e-16)
        grid=np.linspace(0,1,601)
        expanded=np.broadcast_to(context[:,None,:],(12,len(grid),dimension))
        values=np.broadcast_to(grid,(12,len(grid)))
        density=m.density(expanded,values)
        np.testing.assert_allclose(np.trapezoid(density,grid,axis=1),1,atol=5e-16)
        assert density.min()>=.25 and density.max()<=4
        assert not m.coefficients.flags.writeable


def test_nonlinear_cdf_inverse_knots_endpoints_and_empty_shapes():
    rng=np.random.default_rng(192)
    m=model(2)
    u=np.concatenate((np.linspace(0,1,61),rng.uniform(size=100)))
    c=rng.uniform(size=(len(u),2))
    t=m.icdf(c,u)
    np.testing.assert_allclose(m.cdf(c,t),u,atol=5e-16,rtol=5e-16)
    assert np.max(np.abs(t-u))>.005
    knots=np.tile(np.arange(m.bins+1)/m.bins,(5,1))
    ck=np.broadcast_to(np.array([.27,.59]),knots.shape+(2,))
    np.testing.assert_allclose(m.icdf(ck,m.cdf(ck,knots)),knots,atol=5e-16)
    assert m.cdf(np.array([.2,.4]),0.)==0 and m.icdf(np.array([.2,.4]),1.)==1
    assert m.icdf(np.empty((0,2)),np.empty(0)).shape==(0,)
    identity=PositiveDensitySpline(np.ones((1,5)),4,0)
    np.testing.assert_allclose(identity.icdf(None,u),u,atol=1e-16)


def test_cdf_dense_jacobian_and_continuous_context_and_response_derivatives():
    m=model(2)
    def transform(v): return np.r_[v[:2],m.cdf(v[:2],v[2])]
    point=np.array([.31,.57,.42]);step=1e-6
    jac=np.column_stack([(transform(point+step*e)-transform(point-step*e))/(2*step) for e in np.eye(3)])
    np.testing.assert_allclose(np.linalg.det(jac),m.density(point[:2],point[2]),rtol=2e-9)
    inverse_step=1e-6;u=float(m.cdf(point[:2],point[2]))
    inverse_derivative=(m.icdf(point[:2],u+inverse_step)-m.icdf(point[:2],u-inverse_step))/(2*inverse_step)
    np.testing.assert_allclose(inverse_derivative,1/m.density(point[:2],point[2]),rtol=2e-9)
    for knot in np.arange(1,m.bins)/m.bins:
        c=np.array([knot,.57]);eps=1e-7;direction=np.array([eps,0.])
        left=(m.cdf(c,.42)-m.cdf(c-direction,.42))/eps
        right=(m.cdf(c+direction,.42)-m.cdf(c,.42))/eps
        np.testing.assert_allclose(left,right,atol=2e-6)
        fixed=np.array([.31,.57])
        left=(m.cdf(fixed,knot)-m.cdf(fixed,knot-eps))/eps
        right=(m.cdf(fixed,knot+eps)-m.cdf(fixed,knot))/eps
        np.testing.assert_allclose(left,right,atol=2e-6)
        np.testing.assert_allclose((left+right)/2,m.density(fixed,knot),atol=2e-7)


def test_bounded_knapsack_oracle_matches_independent_linear_program():
    g=np.random.default_rng(194).normal(size=(8,7));w=response_integral_weights(6)
    actual=bounded_row_linear_oracle(g,w)
    for gradient,solution in zip(g,actual):
        reference=linprog(gradient,A_eq=w[None],b_eq=[1],bounds=[(.25,4)]*7,method='highs')
        assert reference.success
        np.testing.assert_allclose(gradient@solution,reference.fun,atol=2e-12)
    np.testing.assert_allclose(actual@w,1,atol=1e-15)
    assert actual.min()>=.25 and actual.max()<=4


def test_frank_wolfe_objective_descent_gap_and_cap_status():
    response=((np.arange(240)+.5)/240)**2
    m,d=fit_positive_density_spline(None,response,bins=4,max_iterations=120,gap_tolerance=1e-6)
    assert d.objective<-.05
    assert np.all(np.diff(d.objective_trace)<=1e-12)
    assert d.converged==(d.frank_wolfe_gap<=d.requested_gap)
    features=_joint_features(np.empty((len(response),0)),response,4)
    gradient=np.asarray(features.T@(-1/(len(response)*(features@m.coefficients.ravel())))).reshape(1,5)
    vertex=bounded_row_linear_oracle(gradient,response_integral_weights(4))
    independent_gap=float(np.sum(gradient*(m.coefficients-vertex)))
    np.testing.assert_allclose(d.frank_wolfe_gap,independent_gap,atol=1e-14)
    w=response_integral_weights(4)
    optimum=minimize(lambda a:-np.mean(np.log(features@a)),np.ones(5),
        jac=lambda a:np.asarray(features.T@(-1/(len(response)*(features@a)))),
        constraints=[{'type':'eq','fun':lambda a:w@a-1,'jac':lambda a:w}],
        bounds=[(.25,4)]*5,method='SLSQP',options={'ftol':1e-12,'maxiter':500})
    assert optimum.success
    assert -1e-10<=d.objective-optimum.fun<=d.frank_wolfe_gap+1e-10
    _,capped=fit_positive_density_spline(None,response,bins=4,max_iterations=0,gap_tolerance=1e-6)
    assert not capped.converged and capped.iterations==0 and capped.frank_wolfe_gap>1e-6


def test_conditional_sparse_fit_normalization_and_input_rejection():
    c=np.linspace(0,1,120)[:,None]
    response=.1+.8*c[:,0]**2
    m,d=fit_positive_density_spline(c,response,bins=4,max_iterations=15,gap_tolerance=1e-5)
    assert d.feature_nonzeros<=120*6 and np.all(np.diff(d.objective_trace)<=1e-12)
    np.testing.assert_allclose(m.coefficients@response_integral_weights(4),1,atol=2e-15)
    assert np.all(np.isfinite(m.log_prob(c,response)))
    with pytest.raises(ValueError): m.icdf(c,np.full(120,1.01))
    with pytest.raises(ValueError): fit_positive_density_spline(np.ones(120),response)
    with pytest.raises(ValueError): PositiveDensitySpline(np.ones((1,5))*1.1,4,0)


def test_uniform_empirical_optimum_has_zero_gap_and_reports_convergence():
    response=np.repeat(np.arange(5)/4,[1,2,2,2,1])
    m,d=fit_positive_density_spline(None,response,bins=4,max_iterations=5,gap_tolerance=1e-10)
    assert d.converged and d.iterations==0 and d.frank_wolfe_gap<=1e-10
    np.testing.assert_array_equal(m.coefficients,np.ones((1,5)))
