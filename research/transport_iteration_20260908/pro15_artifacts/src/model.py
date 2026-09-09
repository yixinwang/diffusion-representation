"""Normalized full-dimensional observed-coordinate transport, float64.

One independent Gaussian coordinate per output coordinate; no VAE or hidden
source correspondence. Mathematical densities are continuous laws; floating
point sampling is a numerical implementation, not an atomic-law KL claim.
"""
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import numpy as np
from scipy.special import ndtr, ndtri

A=np.sqrt(1.5)
KNOTS=np.array([0.,.125,.375,.625,.875,1.])
HEIGHTS=A*np.array([1.,1.,-1.,-1.,1.,1.])
SLOPES=np.diff(HEIGHTS)/np.diff(KNOTS)
INTEGRALS=A*np.array([0.,.125,.125,-.125,-.125,0.])
KAPPA=.45

def unit(a, *, closed=True):
    a=np.asarray(a,dtype=np.float64)
    bad=(a<0)|(a>1) if closed else (a<=0)|(a>=1)
    if not np.isfinite(a).all() or np.any(bad):
        raise ValueError('finite unit-interval input required; no clipping')
    return a

def psi(u):
    u=unit(u)
    j=np.searchsorted(KNOTS[1:-1],u,side='right')
    return HEIGHTS[j]+SLOPES[j]*(u-KNOTS[j])

def integral_psi(u):
    u=unit(u);j=np.searchsorted(KNOTS[1:-1],u,side='right');t=u-KNOTS[j]
    return INTEGRALS[j]+HEIGHTS[j]*t+.5*SLOPES[j]*t*t

def pair_encode(u,v,theta):
    u,v,theta=np.broadcast_arrays(unit(u),unit(v),np.asarray(theta,dtype=np.float64))
    if not np.isfinite(theta).all() or np.any(abs(theta)>KAPPA): raise ValueError('bad theta')
    k=theta*psi(u);p=v+k*integral_psi(v);density=1+k*psi(v)
    unit(p)
    if np.any(density<=0):raise ArithmeticError('nonpositive density')
    return p,np.log(density)

def pair_decode(u,p,theta):
    u,p,theta=np.broadcast_arrays(unit(u),unit(p),np.asarray(theta,dtype=np.float64))
    if not np.isfinite(theta).all() or np.any(abs(theta)>KAPPA): raise ValueError('bad theta')
    k=theta*psi(u)
    j=np.sum(p[...,None]>=KNOTS[1:-1]+k[...,None]*INTEGRALS[1:-1],axis=-1)
    delta=p-(KNOTS[j]+k*INTEGRALS[j]);d0=1+k*HEIGHTS[j];beta=k*SLOPES[j]
    disc=d0*d0+2*beta*delta
    if np.any(disc<=0):raise ArithmeticError('invalid quadratic discriminant')
    v=KNOTS[j]+2*delta/(d0+np.sqrt(disc))
    # Exact affine branch identities, including p=0 and p=1; not clipping.
    edge_density=1+k*A
    v=np.where(j==0,p/edge_density,v)
    v=np.where(j==4,1-(1-p)/edge_density,v)
    unit(v)
    return v,-np.log(1+k*psi(v))

def design(c):
    c=unit(c)
    return np.column_stack([np.ones(len(c)),np.sqrt(2)*np.cos(2*np.pi*c),np.sqrt(2)*np.sin(2*np.pi*c)])

def project_simplex_floor(v,floor_mass):
    """Euclidean projection; truth is assumed inside this same closed set."""
    v=np.asarray(v,dtype=np.float64);b=len(v);lo=floor_mass/b
    u=np.sort(v-lo)[::-1];css=np.cumsum(u)-(1-floor_mass)
    ind=np.arange(1,b+1);rho=np.flatnonzero(u-css/ind>0)[-1]
    return np.maximum(v-lo-css[rho]/(rho+1),0)+lo

@dataclass(frozen=True)
class Config:
    root_dim:int=4
    blocks:int=2
    block_size:int=8
    root_bins:int=8
    context_bins:int=8
    graph_arrays:int=65536
    parameter_arrays:int=4096
    delta_graph:float=.01
    root_floor:float=.5
    def __post_init__(self):
        for name in ('root_dim','blocks','block_size','root_bins','context_bins','graph_arrays','parameter_arrays'):
            if type(getattr(self,name)) is not int or getattr(self,name)<=0:raise ValueError('positive integer sizes required')
        if self.block_size%2 or not 0<self.delta_graph<1 or not 0<self.root_floor<1:raise ValueError('invalid configuration')
    @property
    def dimension(self):return self.root_dim+self.blocks*self.block_size
    @property
    def edges(self):return self.blocks*self.block_size*(self.block_size-1)//2

def observed(a,cfg):
    a=unit(a,closed=False)
    if a.ndim!=2 or a.shape[1]!=cfg.dimension:raise ValueError('complete N by D arrays required')
    return a

class Model:
    def __init__(self,cfg,root,pairs,coef,mode='harmonic',diagnostics=None):
        self.cfg=cfg;self.mode=mode
        if mode not in ('harmonic','histogram'):raise ValueError('unknown context mode')
        self.root=np.array(root,dtype=np.float64,copy=True)
        if self.root.shape!=(cfg.root_dim,cfg.root_bins) or not np.isfinite(self.root).all() or np.any(self.root<=0) or not np.allclose(self.root.sum(1),1,rtol=0,atol=2e-14):raise ValueError('bad root density')
        if not np.array_equal(self.root[0],np.full(cfg.root_bins,1/cfg.root_bins)):raise ValueError('first root must be exactly uniform')
        if len(pairs)!=cfg.blocks:raise ValueError('bad block count')
        ps=[]
        for raw in pairs:
            raw=np.asarray(raw)
            if raw.size and not np.issubdtype(raw.dtype,np.integer):raise ValueError('integer edges required')
            p=np.array(raw,dtype=np.int64).reshape(-1,2)
            if np.any(p<0) or np.any(p>=cfg.block_size) or len(np.unique(p))!=p.size:raise ValueError('edges must be a partial matching')
            p=np.sort(p,axis=1);p=p[np.argsort(p[:,0])] if len(p) else p
            p.setflags(write=False);ps.append(p)
        self.pairs=tuple(ps)
        self.coef=np.array(coef,dtype=np.float64,copy=True)
        width=3 if mode=='harmonic' else cfg.context_bins
        if self.coef.shape!=(cfg.blocks,width) or not np.isfinite(self.coef).all():raise ValueError('bad coefficients')
        self.root.setflags(write=False);self.coef.setflags(write=False)
        self.diagnostics={} if diagnostics is None else dict(diagnostics)
    def theta(self,c):
        if self.mode=='harmonic':v=design(c)@self.coef.T
        else:v=self.coef[:,np.minimum((unit(c)*self.cfg.context_bins).astype(int),self.cfg.context_bins-1)].T
        return np.clip(v,-KAPPA,KAPPA)  # registered density-parameter constraint
    def decode_uniform(self,u):
        u=observed(u,self.cfg);x=u.copy();ld=np.zeros(len(u));cfg=self.cfg
        # C is known uniform, not estimated and not transformed.
        for j in range(1,cfg.root_dim):
            p=self.root[j];e=np.r_[0,np.cumsum(p)];e[-1]=1.
            k=np.searchsorted(e,u[:,j],side='right')-1
            x[:,j]=(k+(u[:,j]-e[k])/p[k])/cfg.root_bins
            ld-=np.log(cfg.root_bins*p[k])
        t=self.theta(x[:,0])
        for b,p in enumerate(self.pairs):
            if not len(p):continue
            start=cfg.root_dim+b*cfg.block_size;a=start+p[:,0];v=start+p[:,1]
            x[:,v],q=pair_decode(x[:,a],u[:,v],t[:,b,None]);ld+=q.sum(1)
        observed(x,cfg)
        return x,ld
    def encode_uniform(self,x):
        x=observed(x,self.cfg);u=x.copy();ld=np.zeros(len(x));cfg=self.cfg
        for j in range(1,cfg.root_dim):
            p=self.root[j];e=np.r_[0,np.cumsum(p)];e[-1]=1.
            k=np.minimum((x[:,j]*cfg.root_bins).astype(int),cfg.root_bins-1)
            u[:,j]=e[k]+p[k]*(x[:,j]*cfg.root_bins-k);ld+=np.log(cfg.root_bins*p[k])
        t=self.theta(x[:,0])
        for b,p in enumerate(self.pairs):
            if not len(p):continue
            start=cfg.root_dim+b*cfg.block_size;a=start+p[:,0];v=start+p[:,1]
            u[:,v],q=pair_encode(x[:,a],x[:,v],t[:,b,None]);ld+=q.sum(1)
        observed(u,cfg)
        return u,ld
    def sample(self,z):
        z=np.asarray(z,dtype=np.float64)
        if z.ndim!=2 or z.shape[1]!=self.cfg.dimension or not np.isfinite(z).all():raise ValueError('full Gaussian array required')
        x,ld=self.decode_uniform(ndtr(z))
        return x,ld+(-.5*z*z-.5*np.log(2*np.pi)).sum(1)
    def encode(self,x):
        u,ld=self.encode_uniform(x);z=ndtri(u)
        if not np.isfinite(z).all():raise ArithmeticError('Gaussian saturation')
        return z,ld-(-.5*z*z-.5*np.log(2*np.pi)).sum(1)
    def log_prob(self,x):return self.encode_uniform(x)[1]
    def state(self):
        return dict(schema=1,config=asdict(self.cfg),root=self.root.tolist(),pairs=[p.tolist() for p in self.pairs],coef=self.coef.tolist(),mode=self.mode,diagnostics=self.diagnostics)
    def save(self,path):
        with Path(path).open('x') as f:json.dump(self.state(),f,indent=2,sort_keys=True);f.write('\n')
    @classmethod
    def load(cls,path):
        s=json.loads(Path(path).read_text())
        if s.pop('schema')!=1:raise ValueError('bad schema')
        s['cfg']=Config(**s.pop('config'))
        return cls(**s)

class ExactCopyDecoder:
    """Equally informed stochastic decoder, copying the fitted map exactly.
    This deliberately denies architectural exclusivity, not an independent fit.
    """
    def __init__(self,model):self.model=Model(model.cfg,model.root,model.pairs,model.coef,model.mode,model.diagnostics)
    def sample(self,z):return self.model.sample(z)
    def log_prob(self,x):return self.model.log_prob(x)
