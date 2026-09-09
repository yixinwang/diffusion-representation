from dataclasses import FrozenInstanceError
import math
import numpy as np
import pytest
from scipy.special import ndtr
from qalt.quadratic_innovation_flow import QuadraticInnovationFlow


def test_nonlinear_high_dimension_roundtrip_density_and_all_sources():
    rng=np.random.default_rng(45)
    model=QuadraticInnovationFlow(.61,257)
    z=rng.normal(size=(2,3,257))
    y,ld=model.decode(z);back,ild=model.encode(y)
    assert y.shape==z.shape and ld.shape==(2,3)
    np.testing.assert_allclose(back,z,rtol=1e-11,atol=1e-11)
    np.testing.assert_allclose(ld+ild,0,atol=1e-10)
    log_base=-.5*(257*math.log(2*math.pi)+np.sum(z*z,axis=-1))
    np.testing.assert_allclose(model.log_prob(y)+ld,log_base,atol=1e-11)
    assert np.max(np.abs(y[...,1:]-ndtr(z[...,1:])))>.01
    assert np.array_equal(y[...,0],ndtr(z[...,0]))
    # Every residual source drives its own response with a positive derivative.
    shifted=z.copy();shifted[...,1:]+=.01
    assert np.all(model.decode(shifted)[0][...,1:]>y[...,1:])


def test_dense_finite_jacobian_has_cross_context_derivative():
    model=QuadraticInnovationFlow(-.6,5)
    z=np.array([-.65,.2,-.3,.5,-.1])
    y,ld=model.decode(z);eps=1e-5
    jac=np.column_stack([(model.decode(z+delta)[0]-model.decode(z-delta)[0])/(2*eps)
        for delta in np.eye(5)*eps])
    sign,det=np.linalg.slogdet(jac)
    assert sign==1 and abs(det-ld)<1e-9
    assert np.all(np.abs(jac[1:,0])>1e-5)
    np.testing.assert_allclose(jac[0,1:],0,atol=1e-12)
    np.testing.assert_allclose(jac[1:,1:]-np.diag(np.diag(jac[1:,1:])),0,atol=1e-12)


def test_normalization_pair_moments_and_shared_context_variance():
    model=QuadraticInnovationFlow(.65,2)
    nodes,weights=np.polynomial.legendre.leggauss(72)
    x=(nodes+1)/2;w=weights/2
    c,r=np.meshgrid(x,x,indexing='ij')
    density=np.exp(model.log_prob(np.stack([c,r],axis=-1)))
    measure=w[:,None]*w[None,:]*density
    statistic=6*np.cos(2*np.pi*c)*(2*r-1)
    np.testing.assert_allclose(measure.sum(),1,atol=1e-13)
    np.testing.assert_allclose((measure*statistic).sum(),.65,atol=1e-13)
    np.testing.assert_allclose((measure*(statistic-.65)**2).sum(),6-.65**2,atol=1e-12)
    # Integrate two residuals given the SAME context, preserving dependence.
    c,r1,r2=np.meshgrid(x,x,x,indexing='ij')
    points=np.stack([c,r1,r2],axis=-1)
    density3=np.exp(QuadraticInnovationFlow(.65,3).log_prob(points))
    mass=w[:,None,None]*w[None,:,None]*w[None,None,:]*density3
    cluster=3*np.cos(2*np.pi*c)*((2*r1-1)+(2*r2-1))
    expected=(6-1.5*.65**2)/2+.5*.65**2
    np.testing.assert_allclose(np.sum(mass*(cluster-.65)**2),expected,atol=1e-12)


def test_fit_only_observations_and_counts_whole_arrays():
    observed=np.array([[.12,.3,.9],[.27,.8,.6],[.72,.1,.4],[.91,.7,.2]])
    before=observed.copy()
    model,fit=QuadraticInnovationFlow.fit(observed)
    cluster=6*np.cos(2*np.pi*observed[:,0])*np.mean(2*observed[:,1:]-1,axis=1)
    assert fit.independent_array_count==4 and fit.residuals_per_array==2
    assert fit.cluster_statistic_sample_variance==pytest.approx(cluster.var(ddof=1))
    assert model.theta==pytest.approx(np.clip(cluster.mean(),-.65,.65))
    assert not fit.residuals_assumed_independent_statistical_units
    np.testing.assert_array_equal(observed,before)
    # Duplicating sites cannot multiply the declared number of independent units.
    repeated=np.c_[observed[:,0],np.repeat(observed[:,1:],3,axis=1)]
    duplicated,diagnostic=QuadraticInnovationFlow.fit(repeated)
    assert diagnostic.independent_array_count==4 and duplicated.theta==pytest.approx(model.theta)
    assert diagnostic.cluster_statistic_sample_variance==pytest.approx(fit.cluster_statistic_sample_variance)
    with pytest.raises(FrozenInstanceError):model.theta=0
    with pytest.raises(FrozenInstanceError):fit.independent_array_count=40


def test_clipping_and_boundary_contracts():
    observed=np.array([[.001,.999,.999],[.002,.998,.998]])
    model,fit=QuadraticInnovationFlow.fit(observed)
    assert model.theta==.65 and fit.parameter_was_clipped
    np.testing.assert_array_equal(observed[:,1:],[[.999,.999],[.998,.998]])
    with pytest.raises(ValueError):QuadraticInnovationFlow.fit([[0,.5]])
    with pytest.raises(ValueError):model.encode([.5,1.,.5])
    with pytest.raises(FloatingPointError):model.decode([0.,40.,0.])
    with pytest.raises(ValueError):model.log_prob([.5,np.nan,.5])
    assert model.log_prob([.5,1.,.5])==-np.inf
    with pytest.raises(ValueError):QuadraticInnovationFlow(.7,3)
    with pytest.raises(ValueError):QuadraticInnovationFlow(.1,1)
    with pytest.raises(ValueError):model.decode([0.,0.])
    with pytest.raises(ValueError):model.decode([0j,0j,0j])


def test_same_information_conditional_copy_is_exact():
    model=QuadraticInnovationFlow(.5,7)
    copy=QuadraticInnovationFlow(model.theta,model.dimension,model.rho)
    source=np.linspace(-1.1,1.7,35).reshape(5,7)
    y,ld=model.decode(source);copied,copied_ld=copy.decode(source)
    np.testing.assert_array_equal(y,copied)
    np.testing.assert_array_equal(ld,copied_ld)
    np.testing.assert_array_equal(model.log_prob(y),copy.log_prob(y))
