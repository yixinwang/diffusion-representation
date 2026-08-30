#!/usr/bin/env python3
"""Self-contained reproduction of the RMPF-R18 oracle-headroom table."""
from __future__ import annotations
import json, math
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

D=16; HALF=4; RHO=0.72
SEEDS=range(10030,10035)
GAMMAS=(0.0,0.5,1.0,1.5,2.0,3.0)
MASSES=(0.125,0.25,0.5,1.0)
TRUE_VECTOR=np.array([4.,-3.,2.2,-1.6,.85,-.45,.22,-.10])
TRUE_VECTOR/=np.linalg.norm(TRUE_VECTOR)
TRUE_MATRICES=np.array([
 [[1.15,.40,-.20,.10],[-.35,.95,.25,-.15],[.20,-.10,.85,.45],[-.05,.30,-.40,1.05]],
 [[.90,-.45,.35,.10],[.30,1.10,-.20,.25],[-.40,.15,1.00,-.30],[.20,-.35,.25,.95]],
])

def unit(x):
 x=np.asarray(x,float); r=np.linalg.norm(x,axis=1); u=np.zeros_like(x); m=r>1e-14; u[m]=x[m]/r[m,None]; return u,r

def mobius(u,a):
 a2=np.sum(a*a,axis=1); dot=np.sum(a*u,axis=1); den=1+a2+2*dot
 out=((1-a2)[:,None]*u+2*(1+dot)[:,None]*a)/den[:,None]
 return out,3*(np.log1p(-a2)-np.log(den))

def ball(c,mats,state):
 v,_=unit(c); q=np.einsum('nij,nj->ni',mats[state],v); q2=np.sum(q*q,axis=1)
 return RHO*q/np.sqrt(1+q2)[:,None]

def joint_forward(x,mats,state):
 if not np.any(mats): return x.copy(),np.zeros(len(x))
 c,t=x[:,:4],x[:,4:]; a=ball(c,mats,state); u,r=unit(t); o=t.copy(); ld=np.zeros(len(x)); m=r>1e-14
 o[m],ld[m]=mobius(u[m],a[m]); o[m]*=r[m,None]
 return np.c_[c,o],ld

def joint_inverse(x,mats,state):
 if not np.any(mats): return x.copy(),np.zeros(len(x))
 c,t=x[:,:4],x[:,4:]; a=ball(c,mats,state); u,r=unit(t); o=t.copy(); ld=np.zeros(len(x)); m=r>1e-14
 o[m],ld[m]=mobius(u[m],-a[m]); o[m]*=r[m,None]
 return np.c_[c,o],ld

def objective(flat,c,t):
 m=flat.reshape(4,4); _,ld=joint_forward(np.c_[c,t],m[None],np.zeros(len(c),int))
 return -float(ld.mean())+1e-5*float(np.mean(m*m))

def fit_matrix(c,t):
 vc,_=unit(c); ut,_=unit(t); x0=(-.35*(ut.T@vc/len(vc))).ravel()
 r=minimize(objective,x0,args=(c,t),method='L-BFGS-B',options={'maxiter':180,'ftol':1e-12,'gtol':1e-7,'maxls':30})
 return r.x.reshape(4,4)

def source_state(g,true_state):
 mean=g.mean(0); vec=g[true_state==1].mean(0)-g[true_state==0].mean(0); vec/=np.linalg.norm(vec)
 p=(g-mean)@vec; thr=.5*(p[true_state==0].mean()+p[true_state==1].mean()); return mean,vec,thr

def state(g,mean,vec,thr): return (((g-mean)@vec)>thr).astype(int)

def context_map(g,mean,perm,signs): return (g[:,perm]-mean[perm])*signs

def source_context(mean,vec):
 p=np.argsort(-np.abs(vec),kind='stable'); s=np.sign(vec[p]); s[s==0]=1; return mean.copy(),p,s

def fit_direct(c,t,s):
 shapes=[]; rec=[]; scores={g:0. for g in GAMMAS}
 for k in range(2):
  idx=np.flatnonzero(s==k); fi=idx[(np.arange(len(idx))&1)==0]; va=idx[(np.arange(len(idx))&1)==1]
  m=fit_matrix(c[fi],t[fi]); bic=16*math.log(max(2,len(va)))/(2*max(1,len(va))); gains={}
  for g in GAMMAS:
   _,ld=joint_forward(np.c_[c[va],t[va]],(g*m)[None],np.zeros(len(va),int)); gains[g]=float(ld.mean()); scores[g]+=len(va)*max(gains[g]-bic,0)
  shapes.append(m); rec.append((idx,bic,gains))
 gamma=min(g for g,v in scores.items() if v==max(scores.values()))
 out=np.zeros((2,4,4))
 for k,(idx,bic,gains) in enumerate(rec):
  if gamma>0 and gains[gamma]>bic: out[k]=gamma*fit_matrix(c[idx],t[idx])
 return out,gamma

def energy(gen,ref): return float(cdist(gen,ref).mean()-.5*cdist(gen,gen).mean())

def generate(base,mean,vec,thr,mats,context):
 g,f,t=base[:,:8],base[:,8:12],base[:,12:]; s=state(g,mean,vec,thr); p,_=joint_inverse(np.c_[context(g),t],mats,s); return np.c_[g,f,p[:,4:]]

def strength(g,mean,vec,thr,mats,context):
 s=state(g,mean,vec,thr); return np.linalg.norm(ball(context(g),mats,s),axis=1)

def interval(x):
 x=np.asarray(x); mu=x.mean(); se=x.std(ddof=1)/math.sqrt(len(x)); h=2.7764451051977987*se; return mu,mu-h,mu+h

def run_seed(seed):
 rng=np.random.default_rng(seed); nt,ne=16000,1024
 trb=rng.normal(size=(nt,D)); teb=rng.normal(size=(4096,D))
 tm=np.zeros(8); tp=np.argsort(-np.abs(TRUE_VECTOR),kind='stable'); ts=np.sign(TRUE_VECTOR[tp]); ts[ts==0]=1
 trs=((trb[:,:8]@TRUE_VECTOR)>0).astype(int); tes=((teb[:,:8]@TRUE_VECTOR)>0).astype(int)
 tc=lambda g: context_map(g,tm,tp,ts)[:,:4]
 trp,_=joint_inverse(np.c_[tc(trb[:,:8]),trb[:,12:]],TRUE_MATRICES,trs)
 tep,_=joint_inverse(np.c_[tc(teb[:,:8]),teb[:,12:]],TRUE_MATRICES,tes)
 tr=np.c_[trb[:,:12],trp[:,4:]]; te=np.c_[teb[:,:12],tep[:,4:]]
 mean,vec,thr=source_state(tr[:,:8],trs); cm,perm,sgn=source_context(mean,vec); ctx=lambda g: context_map(g,cm,perm,sgn)[:,:4]
 st=state(tr[:,:8],mean,vec,thr); mats,gamma=fit_direct(ctx(tr[:,:8]),tr[:,12:],st)
 # Preserve the authoritative RNG stream before generated identities.
 rng.permutation(8); rng.choice(np.array([-1.,1.]),size=8); rng.permutation(nt)
 base=rng.normal(size=(ne,D)); ref=te[:ne]
 htr=strength(tr[:,:8],mean,vec,thr,mats,ctx); href=strength(ref[:,:8],mean,vec,thr,mats,ctx); hgen=strength(base[:,:8],mean,vec,thr,mats,ctx)
 direct=generate(base,mean,vec,thr,mats,ctx); zero=base.copy(); oracle=generate(base,tm,TRUE_VECTOR,0.,TRUE_MATRICES,tc)
 rows=[]
 for q in MASSES:
  tau=-np.inf if q==1 else float(np.quantile(htr,1-q,method='higher')); mr=href>=tau; mg=hgen>=tau
  rows.append({'seed':seed,'tail_mass':q,'threshold':tau,'reference_count':int(mr.sum()),'generated_count':int(mg.sum()),
   'oracle_gain':energy(zero[mg],ref[mr])-energy(oracle[mg],ref[mr]),
   'r16_gain':energy(zero[mg],ref[mr])-energy(direct[mg],ref[mr]),'gamma':gamma})
 return rows

rows=[r for s in SEEDS for r in run_seed(s)]; df=pd.DataFrame(rows); out=[]
for q in MASSES:
 d=df[df.tail_mass==q]; om,ol,ou=interval(d.oracle_gain); rm,rl,ru=interval(d.r16_gain)
 out.append({'tail_mass':q,'oracle_energy_gain':om,'oracle_lower':ol,'oracle_upper':ou,'r16_energy_gain':rm,'r16_lower':rl,'r16_upper':ru,'min_reference_count':int(d.reference_count.min()),'min_generated_count':int(d.generated_count.min())})
result=pd.DataFrame(out); print(result.to_csv(index=False)); print(json.dumps({'selected_tail_mass':None,'coupling_family_retired':True,'real_development_opened':False,'confirmation_opened':False},indent=2))
