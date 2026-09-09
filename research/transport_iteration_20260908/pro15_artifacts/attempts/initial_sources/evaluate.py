"""Population evaluation from frozen states; no fitting and no sample selection."""
import numpy as np
from numpy.polynomial.legendre import leggauss
from model import A,KAPPA
from fixture import theta

def context_rule(bins=8,order=32):
    z,w=leggauss(order)
    return ((np.arange(bins)[:,None]+(z+1)/2)/bins).ravel(),np.tile(w/(2*bins),bins)

def entropy_and_derivative(t,terms=64):
    t=np.asarray(t);F=np.zeros_like(t);dF=np.zeros_like(t)
    for k in range(1,terms+1):
        m=(1.5**k)*(k+1)/(2*k+1)
        F+=m*m*t**(2*k)/((2*k)*(2*k-1))
        dF+=m*m*t**(2*k-1)/(2*k-1)
    return F,dF

def edge_counts(truth,model):
    out=[]
    for truep,fitp in zip(truth['pairs'],model.pairs):
        ts=set() if truth['case']=='null' else {tuple(sorted(p)) for p in truep}
        fs={tuple(sorted(p)) for p in fitp}
        out.append(dict(true=len(ts),selected=len(fs),correct=len(ts&fs),missed=len(ts-fs),false=len(fs-ts)))
    return out

def population(truth,model,order=32):
    c,w=context_rule(model.cfg.context_bins,order);tt=theta(c,truth);hh=model.theta(c)
    ft,_=entropy_and_derivative(tt);fh,dh=entropy_and_derivative(hh)
    d=ft-fh-(tt-hh)*dh;j=-fh+hh*dh
    counts=edge_counts(truth,model);parts=[];ub=[]
    for b,k in enumerate(counts):
        parts.append(float(w@(k['correct']*d[:,b]+k['missed']*ft[:,b]+k['false']*j[:,b])))
        ub.append(float(w@(k['correct']*(tt[:,b]-hh[:,b])**2+k['missed']*tt[:,b]**2+k['false']*hh[:,b]**2)/(2*(1-1.5*KAPPA))))
    p=np.asarray(truth['root']);root=float(np.sum(p*np.log(p/model.root)))
    return dict(joint_kl=root+sum(parts),root_kl=root,residual_block_kl=parts,
      graph=counts,deterministic_joint_upper=root+sum(ub),context_quadrature_order=order,
      spectral_projection_norm=np.linalg.norm(np.column_stack([np.sqrt(2)*np.cos(2*np.pi*c),np.sqrt(2)*np.sin(2*np.pi*c)]).T@(w[:,None]*tt),axis=0).tolist(),
      assurance='Ordinary float64 quadrature/series, independently checked; not interval arithmetic.')

def direct_population(truth,model,context_order=48,feature_order=24):
    """Independent density integration, not the power-series evaluator."""
    c,w=context_rule(model.cfg.context_bins,context_order);tt=theta(c,truth);hh=model.theta(c)
    z,v=leggauss(feature_order)
    # Pushforward psi(U): .25 atoms at +/-A and .5 Uniform[-A,A].
    f=np.r_[A,-A,A*z];weights=np.r_[.25,.25,.25*v]
    W=np.outer(f,f).ravel();vw=np.outer(weights,weights).ravel()
    counts=edge_counts(truth,model);parts=[]
    for b,k in enumerate(counts):
        p=1+tt[:,b,None]*W;q=1+hh[:,b,None]*W
        I=(p*np.log(p))@vw;D=(p*(np.log(p)-np.log(q)))@vw;J=(-np.log(q))@vw
        parts.append(float(w@(k['correct']*D+k['missed']*I+k['false']*J)))
    rp=np.asarray(truth['root']);root=float(np.sum(rp*np.log(rp/model.root)))
    return root+sum(parts)
