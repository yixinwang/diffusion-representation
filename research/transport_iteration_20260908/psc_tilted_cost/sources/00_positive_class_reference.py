"""Iteration-8 exact non-Gaussian innovation reference and fair plug-in FM.

Standalone mathematical fixture. No native dataset, repository access, GPU,
external weights, or scheduler. All arrays have D=3072 actual coordinates.
Both generators share fitted coarse law, learned global summary, complete source,
and outer inverse Haar/sigmoid. FM is given the exact Gaussian-reference
population field of the fitted law; it caches the summary once, then uses Heun.
The comparator is this specified algorithm, not all FM parameterizations/solvers.

Requires Python 3, numpy, scipy. Run with OPENBLAS_NUM_THREADS=1.
"""
from __future__ import annotations
import argparse, hashlib, json, math, platform, time
from pathlib import Path
import numpy as np
import scipy
from scipy.special import ndtr, ndtri_exp, log_ndtr, expit

D, COARSE, DETAIL, FEATURES = 3072, 192, 2880, 16
NFES = (4,8,16,32,64)


def quantile_transport(z, e):
    """F_e^{-1}(Phi(z)), equivalently Phi^{-1}(G_e^{-1}(Phi(z))); |e|<1."""
    z,e=np.broadcast_arrays(np.asarray(z,dtype=np.float64),np.asarray(e,dtype=np.float64))
    if not np.isfinite(z).all() or not np.isfinite(e).all() or np.any(np.abs(e)>=1):
        raise ValueError('finite source and |e|<1 required')
    lp=log_ndtr(-np.abs(z)); p=np.exp(lp)
    # Evaluate only the selected tail branch, rather than computing two inverse
    # normal CDFs and discarding one. Algebra and output law are unchanged.
    sign=np.where(z<=0,-1.,1.)
    base=1+sign*e
    denom=base+np.sqrt(base*base-4*sign*e*p)
    return -sign*ndtri_exp(np.log(2.)+lp-np.log(denom))


def inverse_transport(r,e):
    r,e=np.broadcast_arrays(np.asarray(r,dtype=np.float64),np.asarray(e,dtype=np.float64))
    lp=log_ndtr(-np.abs(r)); p=np.exp(lp)
    lo=ndtri_exp(lp+np.log(1-e+e*p))
    hi=-ndtri_exp(lp+np.log(1+e-e*p))
    return np.where(r<=0,lo,hi)


def field(y,t,e):
    a=1-t; s2=a*a+t*t; d=np.sqrt(2*a*a+t*t); k=t/d
    value=k*y
    return 2*e*a/(s2*d)*np.exp(-value*value/2)/np.sqrt(2*np.pi)/(1+e*(2*ndtr(value)-1))


def heun(z,e,nfe):
    if nfe not in NFES:raise ValueError('fixed actual-call grid required')
    y=np.array(z,dtype=np.float64,copy=True);h=2/nfe
    for i in range(nfe//2):
        v=field(y,i*h,e);y+=h/2*(v+field(y+h*v,(i+1)*h,e))
    return y


def orthogonal_dictionary():
    a=np.random.default_rng(77180).normal(size=(COARSE,FEATURES))
    q,_=np.linalg.qr(a)
    return q.T


def haar_inverse_one(c,d):
    lh,hl,hh=np.split(d,3,axis=1)
    out=np.empty((len(c),3,2*c.shape[2],2*c.shape[3]),dtype=np.float64)
    out[:,:,0::2,0::2]=(c+lh+hl+hh)/2
    out[:,:,0::2,1::2]=(c-lh+hl-hh)/2
    out[:,:,1::2,0::2]=(c+lh-hl-hh)/2
    out[:,:,1::2,1::2]=(c-lh-hl+hh)/2
    return out


def outer_decode(c,r):
    x=haar_inverse_one(c.reshape(-1,3,8,8),r[:,:576].reshape(-1,9,8,8))
    x=haar_inverse_one(x,r[:,576:].reshape(-1,9,16,16))
    return expit(x)


def haar_forward_one(x):
    a,b,c,d=x[:,:,0::2,0::2],x[:,:,0::2,1::2],x[:,:,1::2,0::2],x[:,:,1::2,1::2]
    return (a+b+c+d)/2,np.concatenate(((a-b+c-d)/2,(a+b-c-d)/2,(a-b-c+d)/2),axis=1)


def outer_encode(x):
    if np.any((x<=0)|(x>=1)):raise ValueError('non-interior floating pixel input')
    logit=np.log(x)-np.log1p(-x)
    c,d2=haar_forward_one(logit);c,d1=haar_forward_one(c)
    return c.reshape(len(x),-1),np.concatenate((d1.reshape(len(x),-1),d2.reshape(len(x),-1)),axis=1)


def generate(z,gamma,index,dictionary,kind='exact',nfe=None):
    if z.ndim!=2 or z.shape[1]!=D:raise ValueError('complete D-dimensional source required')
    c=quantile_transport(z[:,:COARSE],gamma)
    # The generated source coarse U is available exactly: no inverse recomputation
    # and no observed context at generation. Both arms cache this same summary.
    e=.5+.1*(2*ndtr(z[:,:COARSE]@dictionary[index])-1)
    r=(quantile_transport(z[:,COARSE:],e[:,None]) if kind=='exact'
       else heun(z[:,COARSE:],e[:,None],nfe))
    return outer_decode(c,r)


def fit(root_images,head_images,dictionary):
    """Unknown gamma and index learned only from observed image arrays."""
    c0,_=outer_encode(root_images)
    gamma=.5 if np.mean(2*ndtr(c0)-1)>=0 else -.5
    c,r=outer_encode(head_images);u=inverse_transport(c,gamma)
    psi=2*ndtr(r)-1;features=2*ndtr(u@dictionary.T)-1
    # Common phi(r) term is identical for every summary candidate.
    scores=np.array([np.log1p((.5+.1*features[:,j,None])*psi).sum() for j in range(FEATURES)])
    index=int(scores.argmax())
    return gamma,index,scores


def learning_failure_bound(nroot=32,nhead=256):
    root=math.exp(-nroot*COARSE/72)
    affinity=1-9/16*(1-math.exp(-DETAIL*.1**2*.5**2/(24*1.6)))
    summary=(FEATURES-1)*affinity**nhead
    return {'root_error_bound':root,'summary_affinity_bound':affinity,
            'summary_error_bound':summary,'total_failure_bound':root+summary}


def run(seed=77181,repeats=9):
    rng=np.random.default_rng(seed);dictionary=orthogonal_dictionary()
    truth_gamma,truth_index=.5,7
    start=time.perf_counter()
    root=generate(rng.normal(size=(32,D)),truth_gamma,truth_index,dictionary)
    head=generate(rng.normal(size=(256,D)),truth_gamma,truth_index,dictionary)
    data_seconds=time.perf_counter()-start
    start=time.perf_counter();gamma,index,scores=fit(root,head,dictionary);fit_seconds=time.perf_counter()-start
    z=rng.normal(size=(32,D));x=generate(z,gamma,index,dictionary)
    c,r=outer_encode(x);u=inverse_transport(c,gamma)
    e=.5+.1*(2*ndtr(u@dictionary[index])-1)
    recovered=np.concatenate((u,inverse_transport(r,e[:,None])),axis=1)
    tail=np.array([-50.,-30.,-12.,0.,12.,30.,50.])
    tail_error=max(float(np.max(np.abs(inverse_transport(quantile_transport(tail,e0),e0)-tail))) for e0 in (-.5,.4,.6))
    copy=generate(z,gamma,index,dictionary)
    labels=['exact']+[f'fm_{n}' for n in NFES]
    timings={}
    for batch in (1,64):
        source=rng.normal(size=(batch,D));times={k:[] for k in labels}
        def sample(label):
            return generate(source,gamma,index,dictionary) if label=='exact' else generate(source,gamma,index,dictionary,'fm',int(label[3:]))
        for label in labels:
            for _ in range(3):sample(label)
        for _ in range(repeats):
            for label in rng.permutation(labels):
                t=time.perf_counter();out=sample(label);elapsed=time.perf_counter()-t
                if not np.isfinite(out).all():raise FloatingPointError('nonfinite output')
                times[label].append(elapsed)
        med={k:float(np.median(v)) for k,v in times.items()}
        timings[str(batch)]={'median_seconds':med,'raw_seconds':times,
                            'fm_over_exact':{k:med[k]/med['exact'] for k in labels if k!='exact'}}
    return {'scope':'single local CPU synthetic mathematical fixture; NOT native image/video or neural-model evidence',
            'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,
            'dimensions':{'D':D,'coarse':COARSE,'residual':DETAIL},
            'training':{'root_arrays':32,'summary_arrays':256,'data_generation_seconds':data_seconds,
                        'common_fit_seconds':fit_seconds,'fitted_gamma':gamma,'fitted_index':index,
                        'correct_selection':gamma==truth_gamma and index==truth_index,
                        'summary_score_gap':float(np.sort(scores)[-1]-np.sort(scores)[-2])},
            'finite_learning_bound':learning_failure_bound(),
            'numerical':{'full_source_roundtrip_max':float(np.max(np.abs(recovered-z))),
                         'head_tail_roundtrip_max_on_abs50':tail_error,'exact_copy_bitwise':bool(np.array_equal(x,copy)),
                         'orthogonal_dictionary_error':float(np.max(np.abs(dictionary@dictionary.T-np.eye(FEATURES))))},
            'timing':timings,
            'cost_caveats':'same fitted model and root/summary cache for both; all D output transform charged. Float64 CPU only. Common training cost equal. No GPU/native quality or memory advantage inferred.'}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path);p.add_argument('--seed',type=int,default=77181);p.add_argument('--repeats',type=int,default=9);a=p.parse_args()
    result=run(a.seed,a.repeats);text=json.dumps(result,indent=2);print(text)
    if a.output:a.output.write_text(text+'\n')
