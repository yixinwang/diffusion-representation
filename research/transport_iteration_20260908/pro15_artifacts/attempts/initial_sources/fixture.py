"""Evaluator-owned fabricated distributions. Never imported by discovery code."""
import numpy as np
from scipy.special import ndtr
from model import pair_decode,observed

def make_truth(seed,cfg,case):
    rng=np.random.default_rng(seed)
    p=rng.uniform(.5,1.5,(cfg.root_dim,cfg.root_bins));p/=p.sum(1,keepdims=True)
    p=.5/cfg.root_bins+.5*p;p[0]=1/cfg.root_bins
    return dict(seed=seed,case=case,amplitude=.075,root=p.tolist(),
      pairs=[rng.permutation(cfg.block_size).reshape(-1,2).tolist() for _ in range(cfg.blocks)],
      signs=rng.choice([-1,1],size=cfg.blocks).tolist(),phases=rng.uniform(0,2*np.pi,cfg.blocks).tolist(),
      first_root_uniform=True,conditional_pair_independence=True)

def theta(c,truth):
    c=np.asarray(c);p=np.asarray(truth['phases']);s=np.asarray(truth['signs'])
    if truth['case']=='null':return np.zeros((len(c),len(p)))
    if truth['case']=='harmonic':return truth['amplitude']*s*np.sin(2*np.pi*c[:,None]+p)
    if truth['case']=='off_basis':
        return truth['amplitude']*s*np.tanh(3*np.sin(6*np.pi*c[:,None]+p))/np.tanh(3)
    raise ValueError('unknown fabricated law')

def observe(z,truth,cfg):
    u=ndtr(np.asarray(z,dtype=np.float64));observed(u,cfg);x=u.copy()
    for j in range(1,cfg.root_dim):
        p=np.asarray(truth['root'][j]);e=np.r_[0,np.cumsum(p)];e[-1]=1.
        k=np.searchsorted(e,u[:,j],side='right')-1
        x[:,j]=(k+(u[:,j]-e[k])/p[k])/cfg.root_bins
    t=theta(x[:,0],truth)
    for b,p in enumerate(truth['pairs']):
        p=np.asarray(p);start=cfg.root_dim+b*cfg.block_size
        x[:,start+p[:,1]],_=pair_decode(x[:,start+p[:,0]],u[:,start+p[:,1]],t[:,b,None])
    return observed(x,cfg)
