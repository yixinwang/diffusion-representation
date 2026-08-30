#!/usr/bin/env python3
"""Compact NumPy/SciPy reproduction of the R15 conditioner algebra and aligned NLL gap."""
import json, math
import numpy as np
from scipy.optimize import minimize

RANK=8; HALF=4; RHO=.72
TRUE_V=np.array([4,-3,2.2,-1.6,.85,-.45,.22,-.1],float); TRUE_V/=np.linalg.norm(TRUE_V)
TRUE_A=np.array([[[1.15,.4,-.2,.1],[-.35,.95,.25,-.15],[.2,-.1,.85,.45],[-.05,.3,-.4,1.05]],[[.9,-.45,.35,.1],[.3,1.1,-.2,.25],[-.4,.15,1,-.3],[.2,-.35,.25,.95]]])

def unit(x):
 r=np.linalg.norm(x,axis=1); u=np.zeros_like(x); m=r>1e-14; u[m]=x[m]/r[m,None]; return u,r

def mobius(u,a):
 aa=np.sum(a*a,axis=1); au=np.sum(a*u,axis=1); den=1+aa+2*au
 y=((1-aa)[:,None]*u+2*(1+au)[:,None]*a)/den[:,None]
 return y,3*(np.log1p(-aa)-np.log(den))

def ball(c,A,s):
 v,_=unit(c); q=np.einsum('nij,nj->ni',A[s],v); return RHO*q/np.sqrt(1+np.sum(q*q,axis=1))[:,None]

def forward(x,A,s):
 c,t=x[:,:4],x[:,4:]; a=ball(c,A,s); u,r=unit(t); y,ld=mobius(u,a); return np.c_[c,r[:,None]*y],ld

def inverse(x,A,s):
 c,t=x[:,:4],x[:,4:]; a=ball(c,A,s); u,r=unit(t); y,ld=mobius(u,-a); return np.c_[c,r[:,None]*y],ld

def ctx(g,mean,v):
 p=np.argsort(-np.abs(v),kind='stable'); d=np.sign(v[p]); d[d==0]=1; return (g[:,p]-mean[p])*d,p,d

def obj(flat,c,t):
 _,ld=forward(np.c_[c,t],flat.reshape(1,4,4),np.zeros(len(c),int)); return -ld.mean()+2e-4*np.mean(flat*flat)

def fit(c,t,s):
 A=np.zeros((2,4,4))
 for k in (0,1):
  i=np.where(s==k)[0]; r=minimize(obj,np.zeros(16),args=(c[i],t[i]),method='L-BFGS-B'); A[k]=r.x.reshape(4,4)
 return A

rng=np.random.default_rng(10030); n=16000
base=rng.normal(size=(n,16)); state=(base[:,:8]@TRUE_V>0).astype(int)
true_c,_,_=ctx(base[:,:8],np.zeros(8),TRUE_V)
data_t,_=inverse(np.c_[true_c[:,:4],base[:,12:]],TRUE_A,state)
data=np.c_[base[:,:12],data_t[:,4:]]
mean=data[:,:8].mean(0); v=data[:,:8][state==1].mean(0)-data[:,:8][state==0].mean(0); v/=np.linalg.norm(v)
c,p,d=ctx(data[:,:8],mean,v); A=fit(c[:,:4],data[:,12:],state)
z,ld=forward(np.c_[c[:,:4],data[:,12:]],A,state)
nll_model=np.mean(.5*(np.sum(data[:,:12]**2,axis=1)+np.sum(z[:,4:]**2,axis=1))-ld+.5*16*math.log(2*math.pi))/16
nll_identity=np.mean(.5*np.sum(data**2,axis=1)+.5*16*math.log(2*math.pi))/16
back,ild=inverse(z,A,state)
print(json.dumps({'nll_identity_minus_r15':nll_identity-nll_model,'roundtrip':float(np.max(np.abs(back-np.c_[c[:,:4],data[:,12:]]))),'logdet_cancel':float(np.max(np.abs(ld+ild))),'perm':p.tolist(),'signs':d.tolist()},indent=2))