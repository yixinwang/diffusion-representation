"""Observed-only split-sample matching/histogram learner for the public pair family.

Standard NumPy/SciPy float64 implementation. Density and Jacobians describe the
ideal continuous map a.e.; corners/histogram boundaries are not smooth claims.
No sampling RNG, oracle graph, or truth coefficients enter this interface.
"""
from dataclasses import dataclass, asdict
import json
from pathlib import Path
import numpy as np
from scipy.special import ndtr, ndtri
import reference


@dataclass(frozen=True)
class Config:
    root_dim: int = 192
    root_bins: int = 8
    blocks: int = 4
    block_size: int = 720
    context_bins: int = 32
    kappa: float = .45
    threshold: float = .175

    def __post_init__(self):
        if any(type(x) is not int or x <= 0 for x in (self.root_dim,self.root_bins,self.blocks,self.block_size,self.context_bins)):
            raise ValueError('positive integer sizes required')
        if self.block_size % 2 or not 0 < self.kappa <= reference.KAPPA or not 0 < self.threshold < 1:
            raise ValueError('invalid matching/coefficient configuration')

    @property
    def dimension(self):
        return self.root_dim+self.blocks*self.block_size


def unit(x, dimension):
    x=np.asarray(x,dtype=np.float64)
    if x.ndim!=2 or x.shape[1]!=dimension or not np.isfinite(x).all() or np.any((x<=0)|(x>=1)):
        raise ValueError('finite open-unit N by D array required; no clipping')
    return x


def gaussian(x, dimension):
    x=np.asarray(x,dtype=np.float64)
    if x.ndim!=2 or x.shape[1]!=dimension or not np.isfinite(x).all():
        raise ValueError('finite full-dimensional Gaussian source required')
    return x


class Learner:
    def __init__(self, config, root_probabilities, pairs, coefficients, graph_failures, diagnostics, *, constant_context=False, continuous_context=False):
        self.config=config
        self.constant_context=bool(constant_context)
        self.continuous_context=bool(continuous_context)
        self.root_probabilities=np.array(root_probabilities,dtype=np.float64,copy=True)
        if self.root_probabilities.shape!=(config.root_dim,config.root_bins) or not np.isfinite(self.root_probabilities).all() or np.any(self.root_probabilities<=0) or not np.allclose(self.root_probabilities.sum(1),1,rtol=0,atol=1e-14):
            raise ValueError('invalid root probabilities')
        raw_pairs=[np.asarray(p) for p in pairs]
        if any(p.size and (not np.issubdtype(p.dtype,np.integer)) for p in raw_pairs):
            raise ValueError('pair indices must be integers')
        self.pairs=tuple(np.array(p,dtype=np.int64).reshape(-1,2) for p in raw_pairs)
        self.coefficients=np.array(coefficients,dtype=np.float64,copy=True)
        bins=1 if self.constant_context else config.context_bins
        self.graph_failures=tuple(bool(f) for f in graph_failures)
        if len(self.pairs)!=config.blocks or len(self.graph_failures)!=config.blocks or self.coefficients.shape!=(config.blocks,bins) or not np.isfinite(self.coefficients).all() or np.any(abs(self.coefficients)>config.kappa):
            raise ValueError('invalid state dimensions/coefficients')
        for p, failed, theta in zip(self.pairs,self.graph_failures,self.coefficients):
            if failed:
                if p.size or np.any(theta):raise ValueError('failed graph must be product')
            elif p.shape!=(config.block_size//2,2) or sorted(p.ravel().tolist())!=list(range(config.block_size)):
                raise ValueError('pairs must partition entire block')
        for a in (self.root_probabilities,self.coefficients,*self.pairs):a.setflags(write=False)
        self.diagnostics=dict(diagnostics)

    @classmethod
    def fit(cls, observed, config=Config(), *, constant_context=False, continuous_context=False):
        x=unit(observed,config.dimension);n=len(x)
        if n<2:raise ValueError('at least two complete arrays required')
        ns=n//2; nr=n-ns
        root_counts=np.ones((config.root_dim,config.root_bins),dtype=np.float64)
        indices=np.minimum((x[:,:config.root_dim]*config.root_bins).astype(int),config.root_bins-1)
        for j in range(config.root_dim):root_counts[j]+=np.bincount(indices[:,j],minlength=config.root_bins)
        probabilities=root_counts/(n+config.root_bins)
        bins=1 if constant_context else config.context_bins
        cbin=np.zeros(nr,dtype=int) if constant_context else np.minimum((x[ns:,0]*bins).astype(int),bins-1)
        counts=np.bincount(cbin,minlength=bins)
        pairs=[];failures=[];coefficients=[];degrees=[]
        for block in range(config.blocks):
            start=config.root_dim+block*config.block_size
            features=reference.psi(x[:ns,start:start+config.block_size])
            gram=features.T@features/ns
            adjacency=np.abs(gram)>config.threshold
            np.fill_diagonal(adjacency,False)
            degree=adjacency.sum(1);failed=not bool(np.all(degree==1))
            p=np.empty((0,2),dtype=int) if failed else np.argwhere(np.triu(adjacency,1))
            theta=np.zeros(bins)
            if not failed:
                f=reference.psi(x[ns:,start:start+config.block_size])
                # One response per complete observed array; pairs are not iid units.
                response=np.mean(f[:,p[:,0]]*f[:,p[:,1]],axis=1)
                sums=np.bincount(cbin,weights=response,minlength=bins)
                np.divide(sums,counts,out=theta,where=counts>0)
                theta=np.clip(theta,-config.kappa,config.kappa)
            pairs.append(p);failures.append(failed);coefficients.append(theta);degrees.append(degree.tolist())
        diagnostics={'observed_arrays':n,'graph_arrays':ns,'regression_arrays':nr,
          'root_scalar_bin_updates':n*config.root_dim,'gram_multiplications':config.blocks*ns*config.block_size**2,
          'gram_additions':config.blocks*max(ns-1,0)*config.block_size**2,
          'gram_entries':config.blocks*config.block_size**2,
          'regression_pair_products':sum(len(p) for p in pairs)*nr,
          'context_array_counts':counts.tolist(),'graph_degrees':degrees,
          'pair_responses_are_independent_samples':False,
          'operation_counts':'dense mathematical dot-product counts; not measured BLAS instructions or latency'}
        return cls(config,probabilities,pairs,coefficients,failures,diagnostics,constant_context=constant_context,continuous_context=continuous_context)

    def _theta(self, root):
        if self.continuous_context and not self.constant_context:
            # Linear interpolation at fitted bin centers; constant endpoint extension.
            # Index bounding selects an extension, not a clipped coefficient/output.
            position=root[:,0]*self.config.context_bins-.5
            left=np.floor(position).astype(int)
            weight=position-left
            lo=np.maximum(left,0)
            hi=np.minimum(left+1,self.config.context_bins-1)
            lo=np.minimum(lo,self.config.context_bins-1)
            hi=np.maximum(hi,0)
            return self.coefficients[:,lo]+weight[None,:]*(self.coefficients[:,hi]-self.coefficients[:,lo])
        idx=np.zeros(len(root),dtype=int) if self.constant_context else np.minimum((root[:,0]*self.config.context_bins).astype(int),self.config.context_bins-1)
        return self.coefficients[:,idx]

    def decode_uniform(self, source):
        u=unit(source,self.config.dimension);x=u.copy();ld=np.zeros(len(u));cfg=self.config
        for j,p in enumerate(self.root_probabilities):
            edges=np.r_[0,np.cumsum(p)];edges[-1]=1.
            k=np.searchsorted(edges,u[:,j],side='right')-1
            x[:,j]=(k+(u[:,j]-edges[k])/p[k])/cfg.root_bins
            ld-=np.log(cfg.root_bins*p[k])
        theta=self._theta(x[:,:cfg.root_dim])
        for b,pairs in enumerate(self.pairs):
            if not len(pairs):continue
            a=cfg.root_dim+b*cfg.block_size+pairs[:,0];v=cfg.root_dim+b*cfg.block_size+pairs[:,1]
            x[:,v],pairld=reference.decode(x[:,a],u[:,v],theta[b,:,None]);ld+=pairld.sum(1)
        unit(x,cfg.dimension)
        return x,ld

    def encode_uniform(self, observed):
        x=unit(observed,self.config.dimension);u=x.copy();ld=np.zeros(len(x));cfg=self.config
        for j,p in enumerate(self.root_probabilities):
            edges=np.r_[0,np.cumsum(p)];edges[-1]=1.
            k=np.minimum((x[:,j]*cfg.root_bins).astype(int),cfg.root_bins-1)
            u[:,j]=edges[k]+p[k]*(x[:,j]*cfg.root_bins-k)
            ld+=np.log(cfg.root_bins*p[k])
        theta=self._theta(x[:,:cfg.root_dim])
        for b,pairs in enumerate(self.pairs):
            if not len(pairs):continue
            a=cfg.root_dim+b*cfg.block_size+pairs[:,0];v=cfg.root_dim+b*cfg.block_size+pairs[:,1]
            u[:,v],pairld=reference.encode(x[:,a],x[:,v],theta[b,:,None]);ld+=pairld.sum(1)
        unit(u,cfg.dimension)
        return u,ld

    def sample_from_gaussian(self, z):
        z=gaussian(z,self.config.dimension)
        u=unit(ndtr(z),self.config.dimension) # saturation rejected before transformation
        x,ld=self.decode_uniform(u)
        return x,ld+(-.5*z*z-.5*np.log(2*np.pi)).sum(1)

    def encode_gaussian(self, observed):
        u,ld=self.encode_uniform(observed);z=gaussian(ndtri(u),self.config.dimension)
        return z,ld-(-.5*z*z-.5*np.log(2*np.pi)).sum(1)

    def log_prob(self, observed):
        return self.encode_uniform(observed)[1]

    def state_dict(self):
        return {'schema':1,'config':asdict(self.config),'root_probabilities':self.root_probabilities.tolist(),
          'pairs':[p.tolist() for p in self.pairs],'coefficients':self.coefficients.tolist(),
          'graph_failures':list(self.graph_failures),'constant_context':self.constant_context,'continuous_context':self.continuous_context,'diagnostics':self.diagnostics}

    def save(self,path):
        with Path(path).open('x') as f:json.dump(self.state_dict(),f,sort_keys=True,indent=2);f.write('\n')

    @classmethod
    def load(cls,path):
        state=json.loads(Path(path).read_text())
        if state.pop('schema')!=1:raise ValueError('unknown schema')
        state['config']=Config(**state['config'])
        return cls(**state)
