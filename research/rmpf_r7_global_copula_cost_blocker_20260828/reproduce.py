#!/usr/bin/env python3
"""Standalone RMPF-R7 hidden-copula reproduction (NumPy/SciPy only)."""
from __future__ import annotations
import json, math
from dataclasses import dataclass
import numpy as np
from scipy import stats
from scipy.fft import dct, idct
from scipy.linalg import hadamard
from scipy.special import ndtr, ndtri

LOG2PI = math.log(2.0 * math.pi)


def normal_logpdf(x):
    x=np.asarray(x,float); return -.5*(np.sum(x*x,axis=1)+x.shape[1]*LOG2PI)

def nll(x, ld): return float(-np.mean(normal_logpdf(x)+ld)/x.shape[1])
def state(c): return (np.prod(np.where(c>=0.,1,-1),axis=1)>0).astype(np.int64)
def ci(x):
    x=np.asarray(x,float); m=float(x.mean()); q=float(stats.t.ppf(.975,len(x)-1)); se=float(x.std(ddof=1)/math.sqrt(len(x)))
    return {'mean':m,'lower95':m-q*se,'upper95':m+q*se,'values':x.tolist()}

@dataclass
class BlockCopula:
    indices: np.ndarray
    means: np.ndarray
    scales: np.ndarray
    active: np.ndarray
    a_dim: int
    b_dim: int
    def __post_init__(self):
        self.indices=np.asarray(self.indices,np.int64); self.means=np.asarray(self.means,float); self.scales=np.asarray(self.scales,float); self.active=np.asarray(self.active,bool)
        self.H=hadamard(len(self.indices),dtype=float)/math.sqrt(len(self.indices))
    @property
    def rank(self): return len(self.indices)
    @property
    def m(self): return self.rank//2
    def transform(self,x):
        out=np.asarray(x,float).copy(); b=out[:,self.a_dim:]; h=dct(b,type=2,norm='ortho',axis=1); p=h[:,self.indices]@self.H.T; s=state(p[:,:self.m]); y=p.copy(); ld=np.zeros(len(out))
        for j in range(self.m):
            mu=self.means[j,s] if self.active[j] else 0.; sg=self.scales[j,s] if self.active[j] else 1.; y[:,self.m+j]=(p[:,self.m+j]-mu)/sg; ld-=np.log(sg)
        h[:,self.indices]=y@self.H; out[:,self.a_dim:]=idct(h,type=2,norm='ortho',axis=1); return out,ld
    def inverse(self,z):
        out=np.asarray(z,float).copy(); b=out[:,self.a_dim:]; h=dct(b,type=2,norm='ortho',axis=1); y=h[:,self.indices]@self.H.T; s=state(y[:,:self.m]); p=y.copy(); ld=np.zeros(len(out))
        for j in range(self.m):
            mu=self.means[j,s] if self.active[j] else 0.; sg=self.scales[j,s] if self.active[j] else 1.; p[:,self.m+j]=y[:,self.m+j]*sg+mu; ld+=np.log(sg)
        h[:,self.indices]=p@self.H; out[:,self.a_dim:]=idct(h,type=2,norm='ortho',axis=1); return out,ld
    def copy(self): return BlockCopula(self.indices.copy(),self.means.copy(),self.scales.copy(),self.active.copy(),self.a_dim,self.b_dim)

def fit_two(v,s):
    mu=np.zeros(2); sg=np.ones(2)
    for c in (0,1):
        z=v[s==c]
        if len(z)>=2: mu[c]=z.mean(); sg[c]=np.clip(z.std(ddof=1),.25,4.)
    return mu,sg

def fit_block(x,a_dim,rank,seed):
    b=x[:,a_dim:]; h=dct(b,type=2,norm='ortho',axis=1); idx=np.arange(rank); H=hadamard(rank,dtype=float)/math.sqrt(rank); p=h[:,idx]@H.T; m=rank//2
    rng=np.random.default_rng(seed); perm=rng.permutation(len(x)); fit,cal=perm[:len(x)//2],perm[len(x)//2:]; sf=state(p[fit,:m]); sc=state(p[cal,:m])
    means=np.zeros((m,2)); scales=np.ones((m,2)); active=np.zeros(m,bool)
    for j in range(m):
        mu,sg=fit_two(p[fit,m+j],sf); cand=np.mean(.5*(((p[cal,m+j]-mu[sc])/sg[sc])**2+2*np.log(sg[sc])+LOG2PI)); ident=np.mean(.5*(p[cal,m+j]**2+LOG2PI)); gain=ident-cand
        if min(np.sum(sf==0),np.sum(sf==1))>=64 and gain>.001:
            means[j],scales[j]=fit_two(p[:,m+j],state(p[:,:m])); active[j]=True
    return BlockCopula(idx,means,scales,active,a_dim,b.shape[1])

def strict_knots(v,lo,hi,w=1e-3):
    v=np.asarray(v,float).copy(); v[0],v[-1]=lo,hi
    for i in range(1,len(v)): v[i]=max(v[i],v[i-1]+w)
    if v[-1]>hi: v=lo+(v-lo)*((hi-lo)/(v[-1]-lo))
    v[0],v[-1]=lo,hi
    for i in range(len(v)-2,-1,-1): v[i]=min(v[i],v[i+1]-w)
    v[0],v[-1]=lo,hi; return v

@dataclass
class RQS:
    x:np.ndarray; y:np.ndarray; d:np.ndarray; B:float
    @classmethod
    def fit(cls,samples,bins=24,B=8.):
        s=np.asarray(samples,float); lo,hi=-B,B; pl=np.mean(s<=lo); ph=np.mean(s<=hi); inside=ph-pl
        if len(s)<8 or inside<.5:
            k=np.linspace(lo,hi,bins+1); return cls(k,k.copy(),np.ones_like(k),B)
        p=pl+inside*np.linspace(0,1,bins+1); x=strict_knots(np.clip(np.quantile(s,np.clip(p,0,1)),lo,hi),lo,hi); np0,np1=ndtr(lo),ndtr(hi); y=ndtri(np0+(np1-np0)*np.linspace(0,1,bins+1)); y[0],y[-1]=lo,hi; y=strict_knots(y,lo,hi); delta=np.diff(y)/np.diff(x); d=np.ones(len(x)); d[1:-1]=np.maximum(2*delta[:-1]*delta[1:]/np.maximum(delta[:-1]+delta[1:],1e-12),1e-3); return cls(x,y,d,B)
    def forward(self,v):
        v=np.asarray(v,float); out=v.copy(); ld=np.zeros_like(v); mask=(v>-self.B)&(v<self.B)
        if not mask.any(): return out,ld
        x=v[mask]; k=np.clip(np.searchsorted(self.x,x,side='right')-1,0,len(self.x)-2); x0,x1=self.x[k],self.x[k+1]; y0,y1=self.y[k],self.y[k+1]; d0,d1=self.d[k],self.d[k+1]; w=x1-x0; h=y1-y0; delta=h/w; th=np.clip((x-x0)/w,0,1); om=1-th; q=d0+d1-2*delta; den=delta+q*th*om; num=delta*th*th+d0*th*om; out[mask]=y0+h*num/den; der=delta*delta*(d1*th*th+2*delta*th*om+d0*om*om)/(den*den); ld[mask]=np.log(der); return out,ld

def r6(train,test,layer):
    a=layer.a_dim; htr=dct(train[:,a:],type=2,norm='ortho',axis=1); hte=dct(test[:,a:],type=2,norm='ortho',axis=1); H=layer.H; ptr=htr[:,layer.indices]@H.T; pte=hte[:,layer.indices]@H.T; y=np.empty_like(pte); ld=np.zeros(len(test))
    for j in range(layer.rank): y[:,j],d=RQS.fit(ptr[:,j]).forward(pte[:,j]); ld+=d
    hte[:,layer.indices]=y@H; out=test.copy(); out[:,a:]=idct(hte,type=2,norm='ortho',axis=1); return out,ld,pte,y

def gap(original,transformed):
    m=original.shape[1]//2; s=state(original[:,:m]); return float(np.mean([abs(np.mean(transformed[s==1,m+j])-np.mean(transformed[s==0,m+j])) for j in range(m)]))

def one(seed):
    rng=np.random.default_rng(seed); D,a,b,r=32,8,24,8; means=np.tile([[-3.,3.]],(r//2,1)); scales=np.tile([[.75,1.25]],(r//2,1)); teacher=BlockCopula(np.arange(r),means,scales,np.ones(r//2,bool),a,b)
    train0=rng.normal(size=(50000,D)); train,_=teacher.inverse(train0); test0=rng.normal(size=(30000,D)); test,_=teacher.inverse(test0); layer=fit_block(train,a,r,seed+10); z,ld=layer.transform(test); back,ild=layer.inverse(z[:128]); copy=layer.copy(); zc,ldc=copy.transform(test); rz,rld,po,pr=r6(train,test,layer)
    h=dct(test[:,a:],type=2,norm='ortho',axis=1); p=h[:,layer.indices]@layer.H.T; hz=dct(z[:,a:],type=2,norm='ortho',axis=1); py=hz[:,layer.indices]@layer.H.T
    x=test[:1]; eps=1e-6; J=np.empty((D,D))
    for j in range(D):
        xp=x.copy(); xm=x.copy(); xp[0,j]+=eps; xm[0,j]-=eps; yp,_=layer.transform(xp); ym,_=layer.transform(xm); J[:,j]=(yp[0]-ym[0])/(2*eps)
    sign,fd=np.linalg.slogdet(J); return {'seed':seed,'identity_minus_r7':nll(test,np.zeros(len(test)))-nll(z,ld),'r6_minus_r7':nll(rz,rld)-nll(z,ld),'r7_gap':gap(p,py),'r6_gap':gap(po,pr),'active':int(layer.active.sum()),'roundtrip':float(np.max(np.abs(back-test[:128]))),'logdet_cancel':float(np.max(np.abs(ld[:128]+ild))),'fd_logdet_error':abs(float(fd)-float(layer.transform(x)[1][0])),'jacobian_sign':float(sign),'copy_mismatch':float(max(np.max(np.abs(z-zc)),np.max(np.abs(ld-ldc))))}

def main():
    rows=[one(s) for s in range(9500,9505)]; keys=['identity_minus_r7','r6_minus_r7','r7_gap','r6_gap']; verdict={'contrasts':{k:ci([r[k] for r in rows]) for k in keys},'max_roundtrip':max(r['roundtrip'] for r in rows),'max_fd_logdet_error':max(r['fd_logdet_error'] for r in rows),'max_copy_mismatch':max(r['copy_mismatch'] for r in rows),'confirmation_opened':False}; verdict['known_truth_pass']=bool(verdict['contrasts']['identity_minus_r7']['mean']>=.03 and verdict['contrasts']['r6_minus_r7']['lower95']>.01 and max(r['r7_gap'] for r in rows)<=.05 and min(r['r6_gap'] for r in rows)>=.20 and min(r['active'] for r in rows)==4 and verdict['max_roundtrip']<1e-10 and verdict['max_fd_logdet_error']<1e-6 and verdict['max_copy_mismatch']==0)
    print(json.dumps({'rows':rows,'verdict':verdict},indent=2,sort_keys=True))
if __name__=='__main__': main()
