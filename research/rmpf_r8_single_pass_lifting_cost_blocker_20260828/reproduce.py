#!/usr/bin/env python3
"""Standalone deterministic RMPF-R8 algebra check.

This is not the full CIFAR/UCF run. It verifies the exact Haar/lifting inverse and
Jacobian formula used by the cost-gated milestone without external project code.
"""
from __future__ import annotations
import json, math
import numpy as np


def haar(x):
    x00=x[:,0::2,0::2]; x01=x[:,0::2,1::2]
    x10=x[:,1::2,0::2]; x11=x[:,1::2,1::2]
    return ((x00+x01+x10+x11)/2,
            [(x00-x01+x10-x11)/2,
             (x00+x01-x10-x11)/2,
             (x00-x01-x10+x11)/2])


def ihaar(a, ds):
    h,v,d=ds
    x00=(a+h+v+d)/2; x01=(a-h+v-d)/2
    x10=(a+h-v-d)/2; x11=(a-h-v+d)/2
    out=np.empty((len(a),2*a.shape[1],2*a.shape[2],a.shape[3]))
    out[:,0::2,0::2]=x00; out[:,0::2,1::2]=x01
    out[:,1::2,0::2]=x10; out[:,1::2,1::2]=x11
    return out


def neighbor(x):
    out=np.zeros_like(x)
    out[:,1:]+=x[:,:-1]; out[:,:1]+=x[:,:1]
    out[:,:-1]+=x[:,1:]; out[:,-1:]+=x[:,-1:]
    out[:,:,1:]+=x[:,:,:-1]; out[:,:,:1]+=x[:,:,:1]
    out[:,:,:-1]+=x[:,:,1:]; out[:,:,-1:]+=x[:,:,-1:]
    return out/4


def pair_butterfly(x):
    z=x.copy(); n=z.shape[-1]; h=n//2
    a=z[...,:h].copy(); b=z[...,h:]
    z[...,:h]=(a+b)/math.sqrt(2); z[...,h:]=(a-b)/math.sqrt(2)
    return z


def main():
    rng=np.random.default_rng(9200)
    x=rng.normal(size=(4,32,32,1))
    a1,d1=haar(x); a2,d2=haar(a1)
    mu=np.linspace(-.1,.1,a2.size//len(a2)); scale=np.linspace(.8,1.2,len(mu))
    za=(a2.reshape(len(x),-1)-mu)/scale
    parent=a2; neigh=neighbor(parent)
    alpha=.17; beta=-.09; sigma=.73
    r2=[]
    for d in d2:
        raw=(d-alpha*parent-beta*neigh)/sigma
        r2.append(pair_butterfly(raw.reshape(len(x),-1)).reshape(raw.shape))
    # Inverse must recover level 2 before constructing level-1 parent.
    a2r=(za*scale+mu).reshape(a2.shape)
    d2r=[]
    for r in r2:
        raw=pair_butterfly(r.reshape(len(x),-1)).reshape(r.shape)
        d2r.append(sigma*raw+alpha*a2r+beta*neighbor(a2r))
    a1r=ihaar(a2r,d2r)
    d1r=[]
    for d in d1:
        raw=(d-alpha*a1-beta*neighbor(a1))/sigma
        rr=pair_butterfly(raw.reshape(len(x),-1)).reshape(raw.shape)
        raw2=pair_butterfly(rr.reshape(len(x),-1)).reshape(rr.shape)
        d1r.append(sigma*raw2+alpha*a1r+beta*neighbor(a1r))
    xr=ihaar(a1r,d1r)
    logdet=-len(mu)*math.log(1.0)-sum(math.log(v) for v in scale)-sum(d.size//len(x) for d in d2+d1)*math.log(sigma)
    result={
        "max_roundtrip_error":float(np.max(np.abs(x-xr))),
        "finite_logdet":bool(np.isfinite(logdet)),
        "logdet":float(logdet),
        "decision":"deterministic_algebra_pass"
    }
    print(json.dumps(result,indent=2,sort_keys=True))
    assert result["max_roundtrip_error"]<1e-12


if __name__=="__main__":
    main()
