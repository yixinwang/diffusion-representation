import json
import numpy as np
import pytest
from learner import Config,Learner
import reference


def model():
    return Learner(Config(root_dim=1,root_bins=2,blocks=1,block_size=2,context_bins=2),
                   [[.3,.7]],[[[0,1]]],[[.35,-.25]],[False],{})


def test_nonidentity_gaussian_roundtrip_density_jacobian(tmp_path):
    m=model();z=np.array([[.21,-.32,.46]])
    x,ld=m.sample_from_gaussian(z);back,ild=m.encode_gaussian(x)
    np.testing.assert_allclose(back,z,rtol=0,atol=2e-14)
    np.testing.assert_allclose(ld+ild,0,atol=2e-14)
    np.testing.assert_allclose(m.log_prob(x),(-.5*z*z-.5*np.log(2*np.pi)).sum(1)-ld,atol=2e-14)
    eps=1e-6; jac=np.column_stack([(m.sample_from_gaussian(z+np.eye(3)[j]*eps)[0]-m.sample_from_gaussian(z-np.eye(3)[j]*eps)[0]).ravel()/(2*eps) for j in range(3)])
    np.testing.assert_allclose(np.linalg.slogdet(jac)[1],ld[0],atol=1e-9)
    assert np.linalg.matrix_rank(jac)==3
    path=tmp_path/'m.json';m.save(path);copy=Learner.load(path)
    assert copy.state_dict()==m.state_dict()
    np.testing.assert_array_equal(copy.sample_from_gaussian(z)[0],x)
    with pytest.raises(FileExistsError):m.save(path)


def test_observed_unknown_matching_and_split_accounting():
    # Only the fabricated observation constructor knows the matching.
    rng=np.random.default_rng(902);n=2048;cfg=Config(root_dim=1,root_bins=8,blocks=1,block_size=4,context_bins=4)
    x=rng.uniform(.0001,.9999,(n,cfg.dimension))
    for a,b in [(1,3),(2,4)]:x[:,b]=reference.decode(x[:,a],x[:,b],.43)[0]
    m=Learner.fit(x,cfg);ablation=Learner.fit(x,cfg,constant_context=True)
    assert not m.graph_failures[0]
    assert {tuple(v) for v in m.pairs[0]}=={(0,2),(1,3)}
    assert ablation.graph_failures==m.graph_failures
    assert m.diagnostics['graph_arrays']==1024 and m.diagnostics['regression_arrays']==1024
    assert m.diagnostics['regression_pair_products']==2048
    f=reference.psi(x[1024:,1:]);response=(f[:,0]*f[:,2]+f[:,1]*f[:,3])/2
    np.testing.assert_allclose(ablation.coefficients[0,0],np.clip(response.mean(),-.45,.45),rtol=0,atol=2e-15)
    assert np.all(m.coefficients>0.2)
    expected=np.bincount((x[:,0]*8).astype(int),minlength=8)+1
    np.testing.assert_array_equal(m.root_probabilities[0],expected/(n+8))
    source=rng.normal(size=(8,5));y,_=m.sample_from_gaussian(source)
    np.testing.assert_allclose(m.encode_gaussian(y)[0],source,atol=3e-14)


def test_degree_failure_is_product_and_empty_bins_zero():
    cfg=Config(root_dim=1,root_bins=2,blocks=1,block_size=4,context_bins=4)
    x=np.full((12,5),.1) # graph has degree 3, cannot choose a favored matching
    m=Learner.fit(x,cfg)
    assert m.graph_failures==(True,) and m.pairs[0].shape==(0,2)
    np.testing.assert_array_equal(m.coefficients,0)
    u=np.array([[.2,.3,.4,.5,.6]])
    np.testing.assert_array_equal(m.decode_uniform(u)[0][:,1:],u[:,1:])
    cfg2=Config(root_dim=1,root_bins=2,blocks=1,block_size=2,context_bins=4)
    m2=Learner.fit(np.full((12,3),.1),cfg2)
    assert not m2.graph_failures[0]
    np.testing.assert_array_equal(m2.coefficients[0,1:],0)
    assert m2.coefficients[0,0]==.45


def test_source_tail_rejection_no_hidden_rng(monkeypatch):
    m=model()
    def forbidden(*a,**k):raise AssertionError('sampling attempted RNG')
    monkeypatch.setattr(np.random,'default_rng',forbidden)
    z=np.array([[.1,.2,.3]])
    np.testing.assert_array_equal(m.sample_from_gaussian(z)[0],m.sample_from_gaussian(z)[0])
    for value in [50.,-50.,np.inf,np.nan]:
        bad=z.copy();bad[0,1]=value
        with pytest.raises(ValueError):m.sample_from_gaussian(bad)
    with pytest.raises(ValueError):m.sample_from_gaussian(z[:,:2])
    for value in [0.,1.,np.nan]:
        x=np.full((2,3),.2);x[0,0]=value
        with pytest.raises(ValueError):Learner.fit(x,m.config)


def test_continuous_context_old_boundaries_jacobian_and_state(tmp_path):
    cfg=Config(root_dim=1,root_bins=2,blocks=1,block_size=2,context_bins=4)
    m=Learner(cfg,[[.3,.7]],[[[0,1]]],[[.4,-.3,.2,-.4]],[False],{},continuous_context=True)
    # Unit-source root quantiles corresponding to former context-bin boundaries.
    for c in [.25,.5,.75]:
        x=np.array([[c,.21,.42]])
        center=m.encode_uniform(x)[0][0,0]
        source=np.array([[center-1e-9,.21,.42],[center+1e-9,.21,.42]])
        result=m.decode_uniform(source)[0]
        assert np.max(abs(result[1]-result[0]))<1e-7
    np.testing.assert_array_equal(m._theta(np.array([[.01],[.99]])),[[.4,-.4]])
    z=np.array([[.11,-.31,.41]])
    x,ld=m.sample_from_gaussian(z)
    np.testing.assert_allclose(m.encode_gaussian(x)[0],z,atol=3e-14)
    eps=1e-6
    jac=np.column_stack([(m.sample_from_gaussian(z+np.eye(3)[j]*eps)[0]-m.sample_from_gaussian(z-np.eye(3)[j]*eps)[0]).ravel()/(2*eps) for j in range(3)])
    assert abs(jac[2,0])>1e-4 # nonzero context derivative is present
    np.testing.assert_allclose(np.linalg.slogdet(jac)[1],ld[0],atol=2e-9)
    path=tmp_path/'continuous.json';m.save(path);copy=Learner.load(path)
    assert copy.continuous_context
    np.testing.assert_array_equal(copy.sample_from_gaussian(z)[0],x)
    old=m.state_dict();old.pop('continuous_context');path2=tmp_path/'old.json';path2.write_text(json.dumps(old))
    assert not Learner.load(path2).continuous_context


def test_opt_in_preserves_fit_and_constant_ablation():
    cfg=Config(root_dim=1,root_bins=2,blocks=1,block_size=2,context_bins=4)
    x=np.full((12,3),.1)
    old=Learner.fit(x,cfg);new=Learner.fit(x,cfg,continuous_context=True)
    np.testing.assert_array_equal(old.coefficients,new.coefficients)
    np.testing.assert_array_equal(old.root_probabilities,new.root_probabilities)
    assert old.diagnostics==new.diagnostics and not old.continuous_context
    a=Learner.fit(x,cfg,constant_context=True)
    b=Learner.fit(x,cfg,constant_context=True,continuous_context=True)
    u=np.array([[.41,.22,.36]])
    np.testing.assert_array_equal(a.decode_uniform(u)[0],b.decode_uniform(u)[0])
