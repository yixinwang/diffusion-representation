from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from .core import random_orthogonal


@dataclass
class Split:
    x_train: torch.Tensor
    x_val: torch.Tensor
    x_test: torch.Tensor
    y_train: torch.Tensor | None = None
    y_val: torch.Tensor | None = None
    y_test: torch.Tensor | None = None
    c_train: torch.Tensor | None = None
    c_val: torch.Tensor | None = None
    c_test: torch.Tensor | None = None


class ExactQuotientDistribution:
    """Diffeomorphic active transport plus conditionally correlated 2x2 fiber blocks."""
    def __init__(self,D=18,d=2,seed=701):
        if d!=2 or (D-d)%2: raise ValueError("requires d=2 and even residual dimension")
        self.D,self.d,self.rdim=D,d,D-d; self.q=random_orthogonal(D,seed)
        b=self.rdim//2
        self.s1=torch.linspace(.28,.42,b)
        self.s2=torch.linspace(.36,.50,b)
        self.rho=torch.linspace(.24,.48,b)
        ang=torch.linspace(.15,1.35,b)
        self.dir=torch.stack([torch.cos(ang),torch.sin(ang)],1)

    def active_map(self,z0):
        z1=1.75*z0[:,0]+.35*torch.tanh(z0[:,1])+.18*z0[:,0]**3/(1+z0[:,0]**2)
        z2=1.25*z0[:,1]+.25*torch.tanh(z0[:,0])+.12*z0[:,1]**3/(1+z0[:,1]**2)
        return torch.stack([z1,z2],1)

    def factors(self,z):
        out=[]
        for j in range(self.rdim//2):
            rr=self.rho[j].to(z)*torch.tanh(z@self.dir[j].to(z))
            L=torch.zeros(len(z),2,2,device=z.device,dtype=z.dtype)
            L[:,0,0]=self.s1[j].to(z); L[:,1,0]=rr; L[:,1,1]=self.s2[j].to(z); out.append(L)
        return out

    def sample(self,n,seed):
        g=torch.Generator().manual_seed(seed); y0=torch.randn(n,self.D,generator=g); z=self.active_map(y0[:,:2]); rb=[]
        for j,L in enumerate(self.factors(z)):
            rb.append((L@y0[:,2+2*j:2+2*j+2,None]).squeeze(-1))
        r=torch.cat(rb,1); y=torch.cat([z,r],1); x=y@self.q.T
        return {"x":x,"z":z,"r":r,"y":y}

    def diagonal_kl_gap(self,z):
        ans=torch.zeros(len(z),device=z.device)
        for L in self.factors(z):
            S=L@L.transpose(-1,-2); ans += .5*torch.log((S[:,0,0]*S[:,1,1])/torch.linalg.det(S))
        return ans


def synthetic_splits(seed=0,n_train=5000,n_val=1500,n_test=2500,D=18,d=2):
    dist=ExactQuotientDistribution(D,d,701+seed); tr=dist.sample(n_train,seed+1); va=dist.sample(n_val,seed+2); te=dist.sample(n_test,seed+3)
    return dist,Split(tr["x"],va["x"],te["x"]),{"train":tr,"val":va,"test":te}


@dataclass
class Standardizer:
    mean: torch.Tensor
    scale: torch.Tensor
    def transform(self,x): return (x-self.mean)/self.scale
    def inverse(self,x): return x*self.scale+self.mean


def _oh(y,k=10): return torch.nn.functional.one_hot(torch.as_tensor(y,dtype=torch.long),k).float()


def digits_splits(seed=0,train_fraction=.6,val_fraction=.2):
    ds=load_digits(); X=ds.data.astype(np.float32); y=ds.target.astype(np.int64); idx=np.arange(len(X))
    tr,hold=train_test_split(idx,train_size=train_fraction,random_state=seed,stratify=y)
    va,te=train_test_split(hold,train_size=val_fraction/(1-train_fraction),random_state=seed+1,stratify=y[hold])
    def deq(ii,s):
        rng=np.random.default_rng(s); return torch.tensor(np.clip(X[ii]+rng.uniform(-.5,.5,X[ii].shape),-.5,16.5)/16.,dtype=torch.float32)
    xtr,xv,xt=deq(tr,seed+11),deq(va,seed+12),deq(te,seed+13)
    mean=xtr.mean(0,keepdim=True); scale=xtr.std(0,unbiased=True,keepdim=True).clamp_min(.05); st=Standardizer(mean,scale)
    split=Split(st.transform(xtr),st.transform(xv),st.transform(xt),
                torch.tensor(y[tr]),torch.tensor(y[va]),torch.tensor(y[te]),_oh(y[tr]),_oh(y[va]),_oh(y[te]))
    return split,st,{"train_indices":tr.tolist(),"val_indices":va.tolist(),"test_indices":te.tolist()}
