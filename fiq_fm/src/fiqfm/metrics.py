from __future__ import annotations
import math
import numpy as np
import torch
from scipy.linalg import sqrtm
from scipy.stats import ttest_rel
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors


def arr(x): return x.detach().cpu().numpy() if isinstance(x,torch.Tensor) else np.asarray(x)

def paired_subsample(x,y,n,seed):
    x,y=arr(x).astype(float),arr(y).astype(float); n=min(n,len(x),len(y)); rng=np.random.default_rng(seed)
    return x[rng.choice(len(x),n,False)],y[rng.choice(len(y),n,False)]

def sliced_w2(x,y,n_proj=256,max_n=2500,seed=0):
    x,y=paired_subsample(x,y,max_n,seed); rng=np.random.default_rng(seed+1); p=rng.normal(size=(x.shape[1],n_proj)); p/=np.linalg.norm(p,axis=0,keepdims=True)+1e-12
    return float(np.sqrt(np.mean((np.sort(x@p,0)-np.sort(y@p,0))**2)))

def mmd2(x,y,max_n=1000,seed=0):
    x,y=paired_subsample(x,y,max_n,seed); z=np.r_[x,y]; rng=np.random.default_rng(seed+2); probe=z[rng.choice(len(z),min(500,len(z)),False)]
    d=pairwise_distances(probe,squared=True); med=np.median(d[d>0]) if np.any(d>0) else 1.; gamma=1/max(2*med,1e-12)
    kxx=np.exp(-gamma*pairwise_distances(x,squared=True)); kyy=np.exp(-gamma*pairwise_distances(y,squared=True)); kxy=np.exp(-gamma*pairwise_distances(x,y,squared=True)); np.fill_diagonal(kxx,0); np.fill_diagonal(kyy,0)
    n=len(x); return float(max(kxx.sum()/(n*(n-1))+kyy.sum()/(n*(n-1))-2*kxy.mean(),0))
def energy(x,y,max_n=1000,seed=0):
    x,y=paired_subsample(x,y,max_n,seed); return float(max(2*pairwise_distances(x,y).mean()-pairwise_distances(x).mean()-pairwise_distances(y).mean(),0))
def cov_error(x,y):
    x,y=arr(x),arr(y); a,b=np.cov(x,rowvar=False),np.cov(y,rowvar=False); return float(np.linalg.norm(a-b,'fro')/max(np.linalg.norm(a,'fro'),1e-12))
def mean_error(x,y):
    x,y=arr(x),arr(y); return float(np.linalg.norm(x.mean(0)-y.mean(0))/(math.sqrt(x.shape[1])*max(np.sqrt(x.var(0).mean()),1e-12)))
def frechet(x,y,ridge=1e-6):
    x,y=arr(x).astype(float),arr(y).astype(float); mx,my=x.mean(0),y.mean(0); cx=np.cov(x,rowvar=False)+ridge*np.eye(x.shape[1]); cy=np.cov(y,rowvar=False)+ridge*np.eye(x.shape[1]); s=sqrtm(cx@cy); s=s.real if np.iscomplexobj(s) else s
    return float(max(np.sum((mx-my)**2)+np.trace(cx+cy-2*s),0))
def knn_pr(real,gen,k=5,max_n=1500,seed=0):
    r,g=paired_subsample(real,gen,max_n,seed); nr=NearestNeighbors(n_neighbors=k+1).fit(r); rd,_=nr.kneighbors(r); dg,ig=nr.kneighbors(g,n_neighbors=1); precision=np.mean(dg[:,0]<=rd[ig[:,0],-1]); ng=NearestNeighbors(n_neighbors=k+1).fit(g); gd,_=ng.kneighbors(g); dr,ir=ng.kneighbors(r,n_neighbors=1); recall=np.mean(dr[:,0]<=gd[ir[:,0],-1]); return float(precision),float(recall)
def sample_metrics(real,gen,seed=0): return {"sliced_w2":sliced_w2(real,gen,seed=seed),"mmd2":mmd2(real,gen,seed=seed),"energy":energy(real,gen,seed=seed),"cov_error":cov_error(real,gen),"mean_error":mean_error(real,gen)}
def paired_stats(candidate,baseline,lower=True,seed=0):
    a,b=np.asarray(candidate,float),np.asarray(baseline,float); imp=b-a if lower else a-b; rng=np.random.default_rng(seed); boot=imp[rng.integers(0,len(a),(20000,len(a)))].mean(1); out={"candidate_mean":float(a.mean()),"baseline_mean":float(b.mean()),"mean_improvement":float(imp.mean()),"relative_improvement":float(imp.mean()/max(abs(b.mean()),1e-12)),"ci_low":float(np.quantile(boot,.025)),"ci_high":float(np.quantile(boot,.975))}
    if len(a)>=2:
        t=ttest_rel(b,a) if lower else ttest_rel(a,b); out["paired_t_pvalue_one_sided"]=float(t.pvalue/2 if t.statistic>0 else 1-t.pvalue/2)
    return out
