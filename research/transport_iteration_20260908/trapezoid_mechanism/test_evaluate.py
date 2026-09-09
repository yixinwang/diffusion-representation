import numpy as np
from numpy.polynomial.legendre import leggauss
from evaluate import scalar_terms,population_kl


def test_entropy_series_against_independent_mixture_quadrature():
    q,w=leggauss(100);a=np.sqrt(1.5)
    vals=np.r_[-a,a,a*q];mass=np.r_[.25,.25,w/4]
    z=vals[:,None]*vals[None,:];m=mass[:,None]*mass[None,:]
    for t in (-.45,-.21,0,.38,.45):
        lm,cross,ent=scalar_terms(t);log=np.log1p(t*z)
        np.testing.assert_allclose([lm,cross,ent],[(m*log).sum(),(m*z*log).sum(),(m*(1+t*z)*log).sum()],rtol=0,atol=3e-15)


def test_constant_projection_and_wrong_matching_penalty():
    truth={'root_probabilities':[[.5,.5]],'pairs':[[[0,1],[2,3]]],
           'signs':[1],'offset':.35,'phases':[0.]}
    # Nonconstant truth remains .35+.075sin; constant coefficient is its mean.
    state={'config':{'root_bins':2,'context_bins':4,'block_size':4},
           'root_probabilities':[[.5,.5]],'pairs':truth['pairs'],
           'coefficients':[[.35]],'constant_context':True}
    correct=population_kl(truth,state,32)
    wrong=dict(state,pairs=[[[0,2],[1,3]]]);bad=population_kl(truth,wrong,32)
    assert correct['correct_pairs_per_block']==[2] and bad['correct_pairs_per_block']==[0]
    assert 0<correct['joint_kl']<bad['joint_kl']
    np.testing.assert_allclose(correct['joint_kl'],population_kl(truth,state,64)['joint_kl'],rtol=0,atol=2e-15)


def test_small_observed_fit_and_independent_density():
    from learner import Config,Learner
    from fixture import make_truth,observe
    cfg=Config(root_dim=1,root_bins=2,blocks=1,block_size=4,context_bins=4)
    truth=make_truth(519,cfg,.35)
    x=observe(np.random.default_rng(520).standard_normal((2048,5)),truth,cfg)
    model=Learner.fit(x,cfg,continuous_context=True)
    z=np.random.default_rng(521).standard_normal((32,5));draw,ld=model.sample_from_gaussian(z)
    recovered,ild=model.encode_gaussian(draw)
    np.testing.assert_allclose(z,recovered,rtol=0,atol=1e-12)
    np.testing.assert_allclose(ld+ild,0,rtol=0,atol=1e-12)
    s=model.state_dict();risk=population_kl(truth,s,32)
    assert risk['correct_pairs_per_block']==[2] and risk['joint_kl']>0
    # Point density via explicit true fitted pairs and independent trapezoid formula.
    c=draw[:,0];co=np.interp(c,(np.arange(4)+.5)/4,s['coefficients'][0])
    def feature(u):
        return np.sqrt(1.5)*np.where(u<.125,1,np.where(u<.375,2-8*u,np.where(u<.625,-1,np.where(u<.875,8*u-6,1))))
    independent=np.log(2*np.asarray(s['root_probabilities'])[0,(c*2).astype(int)])
    for a,b in s['pairs'][0]:independent+=np.log1p(co*feature(draw[:,1+a])*feature(draw[:,1+b]))
    np.testing.assert_allclose(independent,model.log_prob(draw),rtol=0,atol=1e-14)
