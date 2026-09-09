"""Observation-only matched cosine-copula estimator. No repository/PSC access.

The estimator takes only arrays and public hyperparameters. Fixture truth is in
run_checks.py, and is never an argument to fit(). The theorem concerns real-
arithmetic densities; floating-point sampling is checked, not called exact KL.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
from scipy.special import ndtr, ndtri

SQRT2 = math.sqrt(2.0)
TAU = 2.0 * math.pi

def feature(u: np.ndarray) -> np.ndarray:
    return SQRT2 * np.cos(TAU * np.asarray(u))

def primitive(v: np.ndarray) -> np.ndarray:
    return SQRT2 / TAU * np.sin(TAU * np.asarray(v))

def conditional_cdf(v, u, theta):
    return np.asarray(v) + np.asarray(theta) * feature(u) * primitive(v)

def conditional_pdf(v, u, theta):
    return 1.0 + np.asarray(theta) * feature(u) * feature(v)

def inverse_cdf(w, u, theta, bits: int = 48, linear_grid: bool = True):
    """Binary search dyadic CDF nodes; optionally invert their linear interpolant.

    linear_grid=True defines a genuine, continuous piecewise-linear inverse in
    real arithmetic, unlike returning a rounded bisection midpoint. No 2**bits
    table is allocated. bits<=48 leaves adequate float64 resolution for checks.
    """
    if not 1 <= bits <= 48:
        raise ValueError('bits must be in [1, 48]')
    w, u, theta = np.broadcast_arrays(np.asarray(w, float), np.asarray(u, float),
                                     np.asarray(theta, float))
    if np.any((w <= 0) | (w >= 1)) or np.any(np.abs(theta) >= .5):
        raise ValueError('open-unit source and |theta|<1/2 required')
    lo = np.zeros_like(w); hi = np.ones_like(w)
    for _ in range(bits):
        mid = (lo + hi) * .5
        below = conditional_cdf(mid, u, theta) < w
        lo = np.where(below, mid, lo)
        hi = np.where(below, hi, mid)
    if not linear_grid:
        return (lo + hi) * .5
    h = 2.0 ** -bits
    # Stable average derivative: do not difference two nearly equal CDF values.
    slope = 1.0 + theta * feature(u) * SQRT2 * np.sinc(h) * np.cos(TAU*(lo+.5*h))
    value = lo + (w - conditional_cdf(lo, u, theta)) / slope
    # No clipping: report numerical interval failures instead of hiding them.
    if np.any(value < lo-2e-14) or np.any(value > hi+2e-14):
        raise FloatingPointError('piecewise-linear inverse left its bracket')
    return value

def grid_pdf(v, u, theta, bits: int = 32):
    h = 2.0 ** -bits
    j = np.minimum(np.floor(np.asarray(v)/h), 2.0**bits-1)
    avg = SQRT2*np.sinc(h)*np.cos(TAU*(j+.5)*h)
    return 1.0 + np.asarray(theta)*feature(u)*avg

def grid_cdf(v, u, theta, bits: int = 32):
    h = 2.0 ** -bits
    lo = np.minimum(np.floor(np.asarray(v)/h), 2.0**bits-1)*h
    return conditional_cdf(lo,u,theta)+(np.asarray(v)-lo)*grid_pdf(v,u,theta,bits)

def root_inverse(w: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    w = np.asarray(w, float); probabilities = np.asarray(probabilities, float)
    n,c = w.shape; k = probabilities.shape[1]
    if probabilities.shape[0] != c or np.any(probabilities <= 0):
        raise ValueError('positive root probabilities, one row per coordinate')
    out=np.empty_like(w)
    for j in range(c):
        cum=np.r_[0.,np.cumsum(probabilities[j])]; cum[-1]=1.
        b=np.minimum(np.searchsorted(cum,w[:,j],side='right')-1,k-1)
        out[:,j]=(b+(w[:,j]-cum[b])/probabilities[j,b])/k
    return out

def root_cdf(x: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    x=np.asarray(x,float); k=probabilities.shape[1]
    b=np.minimum((x*k).astype(int),k-1)
    cum=np.c_[np.zeros(len(probabilities)),np.cumsum(probabilities,axis=1)]
    j=np.arange(x.shape[1])[None,:]
    return cum[j,b]+probabilities[j,b]*(x*k-b)

def root_logpdf(x: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    k=probabilities.shape[1]; b=np.minimum((np.asarray(x)*k).astype(int),k-1)
    return np.log(k*probabilities[np.arange(x.shape[1])[None,:],b]).sum(axis=1)

@dataclass
class FittedModel:
    root_prob: np.ndarray
    pairings: list[np.ndarray]
    theta: np.ndarray
    structural_pass: list[bool]
    group_size: int
    n_structure: int
    a: float
    kappa: float
    score_diagnostics: list[dict]

    @property
    def root_dim(self): return len(self.root_prob)
    @property
    def dim(self): return self.root_dim+len(self.pairings)*self.group_size
    @property
    def bins(self): return self.theta.shape[1]

    def context(self,c1):
        b=np.minimum((np.asarray(c1)*self.bins).astype(int),self.bins-1)
        return self.theta[:,b].T

    def sample(self,z: np.ndarray,bits: int=32):
        z=np.asarray(z,float)
        if z.ndim!=2 or z.shape[1]!=self.dim: raise ValueError('wrong Gaussian shape')
        u=ndtr(z)
        if np.any((u<=0)|(u>=1)): raise FloatingPointError('Gaussian CDF saturated')
        out=u.copy(); out[:,:self.root_dim]=root_inverse(u[:,:self.root_dim],self.root_prob)
        th=self.context(out[:,0])
        for b,pairs in enumerate(self.pairings):
            i,j=pairs.T
            out[:,j]=inverse_cdf(u[:,j],u[:,i],th[:,b,None],bits,True)
        return out

    def encode_uniform(self,x: np.ndarray,bits: int=32):
        x=np.asarray(x,float); u=x.copy()
        u[:,:self.root_dim]=root_cdf(x[:,:self.root_dim],self.root_prob)
        th=self.context(x[:,0])
        for b,pairs in enumerate(self.pairings):
            i,j=pairs.T
            u[:,j]=grid_cdf(x[:,j],x[:,i],th[:,b,None],bits)
        return u

    def logpdf(self,x: np.ndarray,grid_bits: int|None=None):
        x=np.asarray(x,float)
        if x.ndim!=2 or x.shape[1]!=self.dim or np.any((x<0)|(x>1)):
            raise ValueError('observed arrays must have the declared shape and unit support')
        ans=root_logpdf(x[:,:self.root_dim],self.root_prob); th=self.context(x[:,0])
        for b,pairs in enumerate(self.pairings):
            i,j=pairs.T
            q=(conditional_pdf(x[:,j],x[:,i],th[:,b,None]) if grid_bits is None else
               grid_pdf(x[:,j],x[:,i],th[:,b,None],grid_bits))
            ans+=np.log(q).sum(axis=1)
        return ans


def fit(observations: np.ndarray, *, root_dim: int, groups: int,
        root_bins: int=8, context_bins: int=32, n_structure: int=2000,
        a: float=.35, kappa: float=.45) -> FittedModel:
    """Learn root masses, arbitrary within-group matching, and nonlinear functions.

    No teacher variables, covariance, pair identities, phases, or latent banks
    are accepted. Known C1 context location and the basis are public assumptions.
    Failed degree-one matching gates cause an explicit product fallback per group.
    """
    x=np.asarray(observations,float)
    if x.ndim!=2 or np.any(~np.isfinite(x)) or np.any((x<0)|(x>=1)):
        raise ValueError('finite 2-D unit-cube observations required')
    n,d=x.shape
    if not 0<n_structure<n: raise ValueError('both independent array splits must be nonempty')
    if (d-root_dim)%(2*groups): raise ValueError('groups must have even equal size')
    if context_bins%root_bins: raise ValueError('context grid must refine root grid')
    if not 0<a<=kappa<.5: raise ValueError('need 0<a<=kappa<1/2')
    size=(d-root_dim)//groups
    counts=np.vstack([np.bincount((x[:,j]*root_bins).astype(int),minlength=root_bins)
                      for j in range(root_dim)])
    root_prob=(counts+1.0)/(n+root_bins)
    bins=np.minimum((x[n_structure:,0]*context_bins).astype(int),context_bins-1)
    nc=np.bincount(bins,minlength=context_bins)
    pairings=[]; th=np.zeros((groups,context_bins)); passed=[]; diagnostics=[]
    for g in range(groups):
        start=root_dim+g*size
        z=feature(x[:n_structure,start:start+size])
        score=z.T@z/n_structure; np.fill_diagonal(score,0.)
        mask=np.abs(score)>a/2; degree=mask.sum(axis=1)
        good=bool(np.all(degree==1)); passed.append(good)
        if good:
            i,j=np.where(np.triu(mask,1)); pairs=np.c_[i+start,j+start]
            t=(feature(x[n_structure:,pairs[:,0]])*feature(x[n_structure:,pairs[:,1]])).mean(axis=1)
            sums=np.bincount(bins,weights=t,minlength=context_bins)
            th[g]=np.clip(np.divide(sums,nc,out=np.zeros_like(sums),where=nc>0),-kappa,kappa)
        else:
            pairs=np.arange(start,start+size).reshape(-1,2)
        diagnostics.append({'degree_one_nodes':int(np.sum(degree==1)),
                            'zero_degree_nodes':int(np.sum(degree==0)),
                            'maximum_degree':int(degree.max()),
                            'maximum_abs_score':float(np.abs(score).max()),
                            'minimum_context_count':int(nc.min()),
                            'maximum_context_count':int(nc.max())})
        pairings.append(pairs)
    return FittedModel(root_prob,pairings,th,passed,size,n_structure,a,kappa,diagnostics)


def entropy_F(theta, terms: int=160):
    """Integral p_theta log p_theta by its absolutely convergent even series."""
    t=np.asarray(theta,float); ans=np.zeros_like(t)
    for k in range(1,terms+1):
        moment=float(math.comb(2*k,k))**2/(4.0**k)
        ans+=moment*t**(2*k)/((2*k)*(2*k-1))
    return ans

def entropy_F_prime(theta, terms: int=160):
    t=np.asarray(theta,float); ans=np.zeros_like(t)
    for k in range(1,terms+1):
        moment=float(math.comb(2*k,k))**2/(4.0**k)
        ans+=moment*t**(2*k-1)/(2*k-1)
    return ans

def pair_kl(theta, eta):
    theta,eta=np.broadcast_arrays(theta,eta)
    return entropy_F(theta)-entropy_F(eta)-(theta-eta)*entropy_F_prime(eta)


def bound(*, N=4000,n_structure=2000,C=192,G=4,m=360,k0=8,K=32,
          kappa=.45,a=.35,L=.5,root_context_density_lower=.5,grid_bits=32):
    n=N-n_structure; M=G*m; E=G*(2*m)*(2*m-1)//2
    pmin=root_context_density_lower/K
    delta=min(1.,2*E*math.exp(-n_structure*a*a/(8*(1+(2+kappa)*a/6))))
    v=L*L/(12*K*K)
    A=K/(n+1)*(1+3/((n+2)*pmin))
    B=1/(2*(1-4*kappa*kappa))
    pieces={'root_estimation':C*(k0-1)/(N+1),
            'context_approximation':B*M*v,
            'context_and_strength_estimation':B*(G+M*v)*A,
            'empty_context_cells':B*M*kappa*kappa*math.exp(-n*pmin),
            'unknown_matching_estimation':delta*M*math.log((1+2*kappa)/(1-2*kappa)),
            'continuous_grid_density_approximation':M*2*math.pi*kappa/(2.**grid_bits*(1-2*kappa))}
    pred=M*v+(G+M*v)*A+M*kappa*kappa*math.exp(-n*pmin)+8*M*kappa*kappa*delta
    return {'assumptions':{'N':N,'n_structure':n_structure,'n_context':n,'C':C,'G':G,'m':m,
                           'k0':k0,'K':K,'kappa':kappa,'a':a,'L':L,'context_density_lower':root_context_density_lower},
            'pieces':pieces,'joint_KL_upper':sum(pieces.values()),'matching_error_probability_upper':delta,
            'conditional_product_KL_lower':M*a*a/2,'candidate_edges':E,
            'pooled_function_squared_error_upper':pred,
            'masked_feature_MSE_gain_per_coordinate_lower':a*a-pred/M,
            'scope':'Expected over independent fitting arrays; continuous real-arithmetic grid density. Not a native-image or floating-point atomic-law KL theorem.'}
