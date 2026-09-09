"""Evaluator-owned synthetic truth. Only observed arrays go to fit()."""
import numpy as np
from scipy.special import ndtr
import reference


def make_truth(seed, config, offset):
    rng=np.random.default_rng(seed)
    p=rng.uniform(.5,1.5,(config.root_dim,config.root_bins));p/=p.sum(1,keepdims=True)
    p=.5/config.root_bins+.5*p
    p[0]=1/config.root_bins
    return {'fixture_seed':seed,'offset':offset,'root_probabilities':p.tolist(),
            'pairs':[rng.permutation(config.block_size).reshape(-1,2).tolist() for _ in range(config.blocks)],
            'signs':rng.choice([-1,1],size=config.blocks).tolist(),
            'phases':rng.uniform(0,2*np.pi,config.blocks).tolist(),
            'first_root_uniform':True,'all_root_density_floor':.5,'true_lipschitz':float(2*np.pi*.075)}


def observe(z, truth, config):
    u=ndtr(np.asarray(z,dtype=np.float64));x=u.copy()
    if not np.isfinite(u).all() or np.any((u<=0)|(u>=1)):raise ArithmeticError('source saturation')
    for j,p in enumerate(np.asarray(truth['root_probabilities'])):
        edges=np.r_[0,np.cumsum(p)];edges[-1]=1.
        k=np.searchsorted(edges,u[:,j],side='right')-1
        x[:,j]=(k+(u[:,j]-edges[k])/p[k])/config.root_bins
    for b,raw in enumerate(truth['pairs']):
        pairs=np.asarray(raw);start=config.root_dim+b*config.block_size
        theta=truth['signs'][b]*(truth['offset']+.075*np.sin(2*np.pi*x[:,0]+truth['phases'][b]))
        x[:,start+pairs[:,1]],_=reference.decode(x[:,start+pairs[:,0]],u[:,start+pairs[:,1]],theta[:,None])
    if not np.isfinite(x).all() or np.any((x<=0)|(x>=1)):raise ArithmeticError('observation saturation')
    return x
