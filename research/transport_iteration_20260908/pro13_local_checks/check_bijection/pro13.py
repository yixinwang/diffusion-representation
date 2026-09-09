"""Pro13: declared tent-copula law and observed-only packed-bit estimator.
Standalone fabricated data only. No repository/native/PSC access in this code.
Floating arrays are not asserted to have continuous-density KL.
"""
from __future__ import annotations
import math
import sys
import time
from typing import Any
import numpy as np
from scipy.special import ndtr, ndtri_exp, log_ndtr

A = math.sqrt(1.5)
KAPPA = .45
KNOTS = np.array([0., .125, .375, .625, .875, 1.])
VALUES = A*np.array([1., 1., -1., -1., 1., 1.])
PRIMITIVES = A*np.array([0., .125, .125, -.125, -.125, 0.])
SLOPES = np.diff(VALUES)/np.diff(KNOTS)


def _unit(x: Any, dtype=np.float64) -> np.ndarray:
    x = np.asarray(x, dtype=dtype)
    if not np.all(np.isfinite(x)) or np.any((x < 0) | (x > 1)):
        raise ValueError('expected finite unit-interval values')
    return x


def psi(x: Any, dtype=np.float64) -> np.ndarray:
    x = _unit(x, dtype)
    r = np.minimum(x, dtype(1)-x)
    return dtype(A)*np.where(r <= dtype(.125), dtype(1),
                   np.where(r < dtype(.375), dtype(2)-dtype(8)*r, dtype(-1)))


def primitive(x: Any, dtype=np.float64) -> np.ndarray:
    x = _unit(x, dtype)
    k = KNOTS.astype(dtype); p = VALUES.astype(dtype)
    h = PRIMITIVES.astype(dtype); b = SLOPES.astype(dtype)
    j = np.searchsorted(k[1:-1], x, side='right')
    d = x-k[j]
    y = h[j]+p[j]*d+dtype(.5)*b[j]*d*d
    return np.where((x == 0) | (x == 1), dtype(0), y)


def cdf(x: Any, alpha: Any, dtype=np.float64) -> np.ndarray:
    x = _unit(x, dtype)
    alpha = np.asarray(alpha, dtype=dtype)
    return x+alpha*primitive(x, dtype)


def icdf(q: Any, alpha: Any, dtype=np.float64, *, diagnostics=False):
    """Four internal comparisons; one stable quadratic or affine inverse.
    Exact endpoints supported for auditing, not a claim of an open-cube float map.
    Tiny segment overshoots are explicitly counted and bounded, not hidden.
    """
    q, alpha = np.broadcast_arrays(_unit(q, dtype), np.asarray(alpha, dtype=dtype))
    if not np.all(np.isfinite(alpha)) or np.any(np.abs(alpha) > dtype(KAPPA*A)*(1+8*np.finfo(dtype).eps)):
        raise ValueError('alpha exceeds declared positive-density class')
    k = KNOTS.astype(dtype); p = VALUES.astype(dtype)
    h = PRIMITIVES.astype(dtype); b = SLOPES.astype(dtype)
    boundaries = k[1:-1]+alpha[..., None]*h[1:-1]
    j = np.sum(q[..., None] >= boundaries, axis=-1)
    delta = q-(k[j]+alpha*h[j])
    d0 = dtype(1)+alpha*p[j]; beta = alpha*b[j]
    disc = d0*d0+dtype(2)*beta*delta
    if np.any(disc <= 0) or not np.all(np.isfinite(disc)):
        raise FloatingPointError('nonpositive/nonfinite quadratic discriminant')
    xi = dtype(2)*delta/(d0+np.sqrt(disc))
    # Flat pieces are genuinely affine; alpha=0 also takes this branch.
    xi = np.where(beta == 0, delta/d0, xi)
    x = k[j]+xi
    excursion = np.maximum(k[j]-x, x-k[j+1])
    tol = 32*np.finfo(dtype).eps
    if np.any(excursion > tol) or not np.all(np.isfinite(x)):
        raise FloatingPointError('inverse left selected segment beyond rounding budget')
    count = int(np.sum(excursion > 0))
    x = np.minimum(np.maximum(x, k[j]), k[j+1])
    x = np.where(q == 0, dtype(0), np.where(q == 1, dtype(1), x))
    if diagnostics:
        return x, dict(min_discriminant=float(np.min(disc)),
                       segment_roundoff_clamps=count,
                       max_segment_excursion=float(max(0., np.max(excursion))),
                       max_cdf_residual=float(np.max(np.abs(cdf(x,alpha,dtype)-q))))
    return x


def gaussian_child_decode(parent_z: Any, noise_z: Any, theta: Any) -> np.ndarray:
    """Gaussianized child, retaining log tail probabilities; no probability clip.
    This is a tail-safe Gaussianized representation, not an open-cube output.
    """
    p, z, t = np.broadcast_arrays(np.asarray(parent_z,float), np.asarray(noise_z,float), np.asarray(theta,float))
    if not np.all(np.isfinite(p)) or not np.all(np.isfinite(z)) or not np.all(np.isfinite(t)) or np.any(abs(t)>KAPPA):
        raise ValueError('nonfinite input or invalid theta')
    alpha = t*psi(ndtr(-np.abs(p)))
    dtail = 1+alpha*A
    lq = log_ndtr(-np.abs(z))
    flat = lq <= np.log(dtail)-math.log(8)
    # Avoid exponential underflow even for inputs +/-1000 in the affine tails.
    lv = np.empty_like(lq)
    lv[flat] = lq[flat]-np.log(dtail[flat])
    lv[~flat] = np.log(icdf(np.exp(lq[~flat]),alpha[~flat]))
    lower = ndtri_exp(lv)
    return np.where(z >= 0, -lower, lower)


def gaussian_child_encode(parent_z: Any, child_z: Any, theta: Any) -> np.ndarray:
    p,y,t = np.broadcast_arrays(np.asarray(parent_z,float),np.asarray(child_z,float),np.asarray(theta,float))
    if not np.all(np.isfinite(p)) or not np.all(np.isfinite(y)) or not np.all(np.isfinite(t)) or np.any(abs(t)>KAPPA):
        raise ValueError('nonfinite input or invalid theta')
    alpha = t*psi(ndtr(-np.abs(p)))
    lv = log_ndtr(-np.abs(y))
    flat = lv <= -math.log(8)
    lq = np.empty_like(lv)
    lq[flat] = lv[flat]+np.log1p(alpha[flat]*A)
    lq[~flat] = np.log(cdf(np.exp(lv[~flat]),alpha[~flat]))
    lower = ndtri_exp(lq)
    return np.where(y >= 0, -lower, lower)


def packed_graph(block: np.ndarray, tau_num=159, tau_den=1000):
    """Observed sign-psi correlations, computed by exact integer Hamming counts.
    A block with any vertex degree != 1 falls back entirely to product.
    CPython int uses sys.int_info.bits_per_digit bits, NOT presumed uint64.
    """
    x = _unit(block)
    if x.ndim != 2 or len(x)<1 or x.shape[1] % 2:
        raise ValueError('nonempty [N, even d] block required')
    n,d = x.shape
    packed = np.packbits((x <= .25)|(x >= .75),axis=0,bitorder='little').T.copy()
    codes = [int.from_bytes(row.tobytes(), 'little') for row in packed]
    degrees = np.zeros(d,dtype=int); edges=[]
    for i in range(d):
        ci=codes[i]
        for j in range(i+1,d):
            cross = n-2*(ci^codes[j]).bit_count()
            if abs(cross)*tau_den >= tau_num*n:
                edges.append((i,j)); degrees[i]+=1; degrees[j]+=1
    accepted = bool(np.all(degrees==1))
    return (edges if accepted else []), dict(accepted=accepted,threshold_edges=len(edges),
        zero_degree=int(np.sum(degrees==0)),multiple_degree=int(np.sum(degrees>1)),
        packed_payload_bytes=int(packed.nbytes),python_integer_bytes=sum(sys.getsizeof(v) for v in codes))


def dense_graph(block: np.ndarray, threshold=.175):
    x = _unit(block); n,d=x.shape
    f=psi(x)
    gram=(f.T@f)/n
    mask=np.abs(gram)>=threshold; np.fill_diagonal(mask,False)
    deg=mask.sum(axis=1); accepted=bool(np.all(deg==1))
    edges=list(zip(*np.where(np.triu(mask,1))))
    edges=[(int(i),int(j)) for i,j in edges]
    return (edges if accepted else []),dict(accepted=accepted,threshold_edges=len(edges),
        zero_degree=int(np.sum(deg==0)),multiple_degree=int(np.sum(deg>1)))


def fit(observed: np.ndarray, *, c=192, groups=4, block_size=720, n_graph=2000,
        root_bins=8, context_bins=32, method='packed') -> dict:
    """Learner receives only observed complete arrays and public configuration."""
    x=_unit(observed)
    if x.ndim!=2 or x.shape[1]!=c+groups*block_size or not 0<n_graph<len(x):
        raise ValueError('shape/split mismatch')
    n=len(x); start=time.perf_counter()
    root_counts=np.zeros((c,root_bins),int)
    for j in range(c):
        root_counts[j]=np.bincount(np.minimum((x[:,j]*root_bins).astype(int),root_bins-1),minlength=root_bins)
    root=(root_counts+1)/(n+root_bins)
    root_time=time.perf_counter()-start
    s=time.perf_counter(); pairs=[]; diagnostics=[]
    for g in range(groups):
        b=x[:n_graph,c+g*block_size:c+(g+1)*block_size]
        if method == 'packed':
            e,di=packed_graph(b)
        elif method == 'packed_c':
            from packed_kernel import compiled_graph
            e,di=compiled_graph(b)
        elif method == 'dense':
            e,di=dense_graph(b)
        else:
            raise ValueError('unknown graph method')
        pairs.append(e); diagnostics.append(di)
    graph_time=time.perf_counter()-s
    s=time.perf_counter()
    bins=np.minimum((x[n_graph:,0]*context_bins).astype(int),context_bins-1)
    counts=np.bincount(bins,minlength=context_bins)
    theta=np.zeros((groups,context_bins))
    for g,e in enumerate(pairs):
        if e:
            indices=np.array(e)+c+g*block_size
            y=(psi(x[n_graph:,indices[:,0]])*psi(x[n_graph:,indices[:,1]])).mean(axis=1)
            sums=np.bincount(bins,weights=y,minlength=context_bins)
            theta[g]=np.clip(np.divide(sums,counts,out=np.zeros(context_bins),where=counts>0),-KAPPA,KAPPA)
    reg_time=time.perf_counter()-s
    return dict(c=c,groups=groups,block_size=block_size,root_bins=root_bins,context_bins=context_bins,
        n_fit=n,n_graph=n_graph,method=method,root_probabilities=root.tolist(),
        pairs=pairs,theta=theta.tolist(),graph_diagnostics=diagnostics,
        timings=dict(root_seconds=root_time,graph_seconds=graph_time,regression_seconds=reg_time,
                     total_fit_seconds=root_time+graph_time+reg_time))


def bounds() -> dict:
    G,d,m,N,n,K,k0,c,L,kappa,a=4,720,360,4000,2000,32,8,192,.5,.45,.35
    M=G*m; H=G*d*(d-1)//2; b=1.5; tau=.159
    signal=27/32*a
    delta_null=2*(H-M)*math.exp(-n*tau*tau/2)
    delta_miss=M*math.exp(-n*(signal-tau)**2/2)
    delta=delta_null+delta_miss
    B=1/(2*(1-(kappa*b)**2))
    bias=L*L/(12*K*K)
    rt=bias+2*K/(n+1)*(1/m+bias)+kappa*kappa*math.exp(-n*.5/K)
    root=c*math.log((N+k0)/(N+1))
    J=M*math.log((1+b*kappa)/(1-b*kappa))
    t=a/2; center=b+kappa
    delta_dense=2*H*math.exp(-n*t*t/(2*(1+center*t/3)))
    return dict(signal=signal,tau=tau,delta_null=delta_null,delta_miss=delta_miss,
        delta_graph=delta,B_kappa=B,bias=bias,theta_risk=rt,root_KL=root,
        conditional_KL=M*B*rt,bad_graph_KL=J*delta,joint_KL=root+M*B*rt+J*delta,
        dense_graph_delta=delta_dense,dense_joint_KL=root+M*B*rt+J*delta_dense,
        fixed_chart_product_floor=M*a*a/2,masked_psi_over_A_excess_MSE=(2/3)*rt+4*kappa*kappa*delta,
        bad_graph_envelope=J,full_gram_FMAs=G*n*d*d,
        unordered_pairs=H,uint64_xor_popcount_words=H*math.ceil(n/64),
        python_bits_per_digit=sys.int_info.bits_per_digit,
        python_digit_upper_iterations=H*math.ceil(n/sys.int_info.bits_per_digit),
        packed_all_groups_bytes=G*d*math.ceil(n/8))


def copula_entropy(theta, terms=96):
    """h(theta)=int (1+theta z)log(1+theta z); symmetric-moment series.
    All terms nonnegative. Tail bounded by rmax^(2T+2)/((2T+2)(2T+1)(1-rmax^2)).
    """
    t=np.asarray(theta,float); out=np.zeros_like(t)
    for r in range(1,terms+1):
        out+=(1.5*t)**(2*r)*((r+1)/(2*r+1))**2/((2*r)*(2*r-1))
    return out


def copula_cross_log(theta, eta, terms=96):
    """int (1+theta z) log(1+eta z), including eta=0."""
    t,e=np.broadcast_arrays(np.asarray(theta,float),np.asarray(eta,float)); out=np.zeros_like(t)
    for r in range(1,terms+1):
        f=((r+1)/(2*r+1))**2
        out+=f*(-(1.5*e)**(2*r)/(2*r)+(1.5*t)*(1.5*e)**(2*r-1)/(2*r-1))
    return out


def decode(state: dict, gaussian: np.ndarray) -> np.ndarray:
    """Full D Gaussian-prior to unit-cube map. Saturation is a recorded failure.
    For extreme Gaussianized scalar-tail audits use gaussian_child_decode.
    """
    z=np.asarray(gaussian,float); c=state['c']; G=state['groups']; d=state['block_size']
    if z.ndim!=2 or z.shape[1]!=c+G*d or not np.all(np.isfinite(z)):
        raise ValueError('Gaussian array shape/finite check failed')
    q=ndtr(z)
    if np.any((q<=0)|(q>=1)):
        raise FloatingPointError('unit-cube Gaussian-CDF saturation; no clipping/resampling')
    x=q.copy(); rp=np.asarray(state['root_probabilities']); k=state['root_bins']
    for j in range(c):
        cum=np.r_[0.,np.cumsum(rp[j])]
        cell=np.searchsorted(cum[1:-1],q[:,j],side='right')
        x[:,j]=(cell+(q[:,j]-cum[cell])/rp[j,cell])/k
    K=state['context_bins']; cb=np.minimum((x[:,0]*K).astype(int),K-1)
    th=np.asarray(state['theta'])
    for g,edges in enumerate(state['pairs']):
        if edges:
            ij=np.asarray(edges,int)+c+g*d
            x[:,ij[:,1]]=icdf(q[:,ij[:,1]],th[g,cb,None]*psi(x[:,ij[:,0]]))
    if np.any((x<=0)|(x>=1)) or not np.all(np.isfinite(x)):
        raise FloatingPointError('unit-cube output boundary; no clipping/resampling')
    return x


def log_prob(state:dict, observed:np.ndarray) -> np.ndarray:
    x=_unit(observed); n=len(x); out=np.zeros(n); c=state['c']; k=state['root_bins']
    if x.ndim!=2 or x.shape[1]!=c+state['groups']*state['block_size']:
        raise ValueError('shape mismatch')
    rp=np.asarray(state['root_probabilities']); cb=np.minimum((x[:,0]*state['context_bins']).astype(int),state['context_bins']-1)
    th=np.asarray(state['theta'])
    for j in range(c):
        cells=np.minimum((x[:,j]*k).astype(int),k-1)
        out+=np.log(k*rp[j,cells])
    for g,edges in enumerate(state['pairs']):
        if edges:
            ij=np.asarray(edges)+c+g*state['block_size']
            out+=np.log1p(th[g,cb,None]*psi(x[:,ij[:,0]])*psi(x[:,ij[:,1]])).sum(axis=1)
    return out


def encode(state:dict, observed:np.ndarray) -> np.ndarray:
    """Inverse of the full learned unit-cube map, rejecting rounded boundaries."""
    from scipy.special import ndtri
    x=_unit(observed); c=state['c'];G=state['groups'];d=state['block_size'];k=state['root_bins']
    if x.ndim!=2 or x.shape[1]!=c+G*d or np.any((x<=0)|(x>=1)):
        raise ValueError('open-cube array of learned dimension required')
    q=x.copy();rp=np.asarray(state['root_probabilities'])
    for j in range(c):
        cell=np.minimum((x[:,j]*k).astype(int),k-1)
        cum=np.r_[0.,np.cumsum(rp[j])]
        q[:,j]=cum[cell]+rp[j,cell]*(k*x[:,j]-cell)
    K=state['context_bins']; cb=np.minimum((x[:,0]*K).astype(int),K-1);th=np.asarray(state['theta'])
    for g,edges in enumerate(state['pairs']):
        if edges:
            ij=np.asarray(edges)+c+g*d
            q[:,ij[:,1]]=cdf(x[:,ij[:,1]],th[g,cb,None]*psi(x[:,ij[:,0]]))
    if np.any((q<=0)|(q>=1)) or not np.all(np.isfinite(q)):
        raise FloatingPointError('encoded probability boundary; no clipping')
    return ndtri(q)
