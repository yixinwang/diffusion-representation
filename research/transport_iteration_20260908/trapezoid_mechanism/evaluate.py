"""Independent ideal-density population evaluator; no fitting or generator imports."""
import numpy as np
from numpy.polynomial.legendre import leggauss


def scalar_terms(theta, terms=128):
    """E log p_theta, E[Z log p_theta], E[p_theta log p_theta].

    Under the independent uniform pair, psi has half uniform[-A,A] and
    quarter mass at each endpoint. The exact even moments give the series.
    Numerical evaluation is float64, not an interval certificate.
    """
    t=np.asarray(theta,dtype=np.float64)
    if not np.isfinite(t).all() or np.any(abs(t)>.45):raise ValueError('theta range')
    logmean=np.zeros_like(t);cross=np.zeros_like(t);entropy=np.zeros_like(t)
    for k in range(1,terms+1):
        moment=1.5**(2*k)*((k+1)/(2*k+1))**2
        even=t**(2*k)*moment
        logmean-=even/(2*k)
        cross+=t**(2*k-1)*moment/(2*k-1)
        entropy+=even/(2*k*(2*k-1))
    return logmean,cross,entropy


def population_kl(truth, state, order=32):
    """True graph appears only in post-fit evaluation, never estimator APIs."""
    cfg=state['config'];root=np.asarray(truth['root_probabilities'])
    fitted=np.asarray(state['root_probabilities'])
    root_kl=float(np.sum(root*np.log(root/fitted)))
    bins=cfg['context_bins'];continuous=state.get('continuous_context',False)
    constant=state.get('constant_context',False)
    breaks=np.unique(np.r_[np.linspace(0,1,cfg['root_bins']+1),
        (np.arange(bins)+.5)/bins if continuous and not constant else np.linspace(0,1,bins+1)])
    nodes,weights=leggauss(order);cs=[];ws=[]
    for lo,hi in zip(breaks[:-1],breaks[1:]):
        c=(lo+hi)/2+(hi-lo)/2*nodes
        density=cfg['root_bins']*root[0,np.minimum((c*cfg['root_bins']).astype(int),cfg['root_bins']-1)]
        cs.extend(c);ws.extend(weights*(hi-lo)/2*density)
    c=np.asarray(cs);w=np.asarray(ws);values=[];overlaps=[]
    for b,pairs in enumerate(state['pairs']):
        theta=truth['signs'][b]*(truth['offset']+.075*np.sin(2*np.pi*c+truth['phases'][b]))
        coef=np.asarray(state['coefficients'][b])
        if constant:eta=np.full_like(c,coef[0])
        elif continuous:eta=np.interp(c,(np.arange(bins)+.5)/bins,coef)
        else:eta=coef[np.minimum((c*bins).astype(int),bins-1)]
        true_pairs={tuple(sorted(p)) for p in truth['pairs'][b]}
        correct=sum(tuple(sorted(p)) in true_pairs for p in pairs)
        lm,cross,_=scalar_terms(eta);_,_,ent=scalar_terms(theta)
        contribution=cfg['block_size']//2*ent-len(pairs)*lm-correct*theta*cross
        values.append(float(w@contribution));overlaps.append(correct)
    return {'root_kl':root_kl,'residual_block_kl':values,'joint_kl':root_kl+sum(values),
            'correct_pairs_per_block':overlaps,'context_quadrature_order':order,
            'series_terms':128,'assurance':'ordinary numerical population evaluation, not interval certificate'}
