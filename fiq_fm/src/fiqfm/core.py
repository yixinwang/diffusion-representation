from __future__ import annotations

import copy
import math
import random
import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
from torch import nn


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


@dataclass
class OrthogonalChart:
    q: torch.Tensor
    active_dim: int
    eigenvalues: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.q.ndim != 2 or self.q.shape[0] != self.q.shape[1]:
            raise ValueError("q must be square")
        if not 0 < self.active_dim < self.q.shape[0]:
            raise ValueError("active_dim must lie in {1,...,D-1}")
        eye = torch.eye(self.q.shape[0], dtype=self.q.dtype, device=self.q.device)
        if torch.linalg.norm(self.q.T @ self.q - eye).item() > 1e-3:
            raise ValueError("q is not orthogonal")

    @property
    def ambient_dim(self) -> int:
        return int(self.q.shape[0])

    @property
    def residual_dim(self) -> int:
        return self.ambient_dim - self.active_dim

    def transform(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        y = x @ self.q.to(x)
        return y[..., : self.active_dim], y[..., self.active_dim :]

    def transform_full(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.q.to(x)

    def inverse(self, z: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        return torch.cat([z, r], dim=-1) @ self.q.to(z).T


def _orient_columns(q: torch.Tensor) -> torch.Tensor:
    q = q.clone()
    for j in range(q.shape[1]):
        i = int(torch.argmax(q[:, j].abs()).item())
        if q[i, j] < 0:
            q[:, j] *= -1
    return q


def random_orthogonal(dim: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(dim, dim, generator=g, dtype=torch.float64)
    q, r = torch.linalg.qr(a)
    s = torch.sign(torch.diag(r)); s[s == 0] = 1
    return _orient_columns(q * s.unsqueeze(0)).float()


def fit_flow_moment_chart(
    x1: torch.Tensor,
    active_dim: int,
    *,
    seed: int = 0,
    n_pairs: int = 20_000,
    t: float = 0.75,
) -> tuple[OrthogonalChart, dict[str, Any]]:
    """Recover a static orthogonal quotient from ordinary rectified-flow labels.

    For centered X1, X0~N(0,I), Xt=(1-t)X0+tX1 and V=X1-X0,
    E sym(V Xt^T)=t Cov(X1)-(1-t)I, so its eigenvectors equal PCA's.
    """
    if x1.ndim != 2:
        raise ValueError("x1 must be [n,D]")
    x = x1.detach().double().cpu()
    x = x - x.mean(0, keepdim=True)
    n, d = x.shape
    if not 0 < active_dim < d:
        raise ValueError("invalid active_dim")
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, n, (n_pairs,), generator=g)
    data = x[idx]
    x0 = torch.randn(n_pairs, d, generator=g, dtype=torch.float64)
    xt = (1 - t) * x0 + t * data
    v = data - x0
    m = (v.T @ xt) / n_pairs
    m = 0.5 * (m + m.T)
    vals, vecs = torch.linalg.eigh(m)
    order = torch.argsort(vals, descending=True)
    vals, vecs = vals[order], _orient_columns(vecs[:, order])

    cov = x.T @ x / max(n - 1, 1)
    cvals, cvecs = torch.linalg.eigh(cov)
    corder = torch.argsort(cvals, descending=True)
    cvals, cvecs = cvals[corder], cvecs[:, corder]
    sv = torch.linalg.svdvals(vecs[:, :active_dim].T @ cvecs[:, :active_dim]).clamp(0, 1)
    sine = torch.sqrt((1 - sv.square()).clamp_min(0)).max().item()
    diagnostics = {
        "n_pairs": n_pairs,
        "t": t,
        "principal_angle_sine_vs_pca": float(sine),
        "active_moment_eigengap": float((vals[active_dim - 1] - vals[active_dim]).item()),
        "moment_eigenvalues": vals[: min(12, d)].tolist(),
        "covariance_eigenvalues": cvals[: min(12, d)].tolist(),
    }
    return OrthogonalChart(vecs.float(), active_dim, vals.float()), diagnostics


def subspace_sine(q1: torch.Tensor, q2: torch.Tensor, d: int) -> float:
    sv = torch.linalg.svdvals(q1[:, :d].double().T @ q2[:, :d].double()).clamp(0, 1)
    return float(torch.sqrt((1 - sv.square()).clamp_min(0)).max().item())


class TimeEmbedding(nn.Module):
    def __init__(self, n_freq: int = 6):
        super().__init__()
        self.register_buffer("freq", 2.0 ** torch.arange(n_freq), persistent=False)

    @property
    def dim(self) -> int:
        return 1 + 2 * self.freq.numel()

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 1:
            t = t[:, None]
        a = 2 * torch.pi * t * self.freq[None, :]
        return torch.cat([t, torch.sin(a), torch.cos(a)], -1)


class VectorField(nn.Module):
    def __init__(self, dim: int, hidden: int = 96, depth: int = 2, condition_dim: int = 0):
        super().__init__()
        self.dim = dim
        self.condition_dim = condition_dim
        self.time = TimeEmbedding()
        inp = dim + condition_dim + self.time.dim
        layers: list[nn.Module] = []
        for _ in range(depth):
            layers += [nn.Linear(inp, hidden), nn.SiLU()]
            inp = hidden
        layers += [nn.Linear(inp, dim)]
        self.net = nn.Sequential(*layers)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor, condition: torch.Tensor | None = None) -> torch.Tensor:
        parts = [x, self.time(t)]
        if self.condition_dim:
            if condition is None:
                raise ValueError("condition required")
            parts.append(condition)
        return self.net(torch.cat(parts, -1))


@dataclass
class TrainResult:
    best_step: int
    best_validation: float
    final_train: float
    history: list[dict[str, float]]


def _fixed_flow_val(model: VectorField, x: torch.Tensor, c: torch.Tensor | None, seed: int) -> float:
    model.eval()
    g = torch.Generator(device=x.device).manual_seed(seed)
    n = min(2048, len(x))
    idx = torch.randperm(len(x), generator=g, device=x.device)[:n]
    x1 = x[idx]; cc = None if c is None else c[idx]
    x0 = torch.randn(x1.shape, generator=g, device=x.device, dtype=x.dtype)
    t = torch.rand(n, 1, generator=g, device=x.device, dtype=x.dtype)
    xt = (1 - t) * x0 + t * x1
    with torch.no_grad():
        return float(((model(xt, t, cc) - (x1 - x0)) ** 2).mean().item())


def train_flow(
    model: VectorField,
    x_train: torch.Tensor,
    x_val: torch.Tensor,
    *,
    c_train: torch.Tensor | None = None,
    c_val: torch.Tensor | None = None,
    steps: int = 800,
    batch_size: int = 256,
    lr: float = 2e-3,
    seed: int = 0,
    eval_every: int = 50,
    patience: int = 10,
) -> tuple[VectorField, TrainResult]:
    seed_all(seed)
    dev = next(model.parameters()).device
    x_train, x_val = x_train.to(dev), x_val.to(dev)
    c_train = None if c_train is None else c_train.to(dev)
    c_val = None if c_val is None else c_val.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, steps))
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters(): p.requires_grad_(False)
    g = torch.Generator(device=dev).manual_seed(seed + 19)
    best, best_step, state, stale = float("inf"), 0, None, 0
    hist: list[dict[str, float]] = []
    final = float("nan")
    for step in range(1, steps + 1):
        model.train()
        idx = torch.randint(0, len(x_train), (batch_size,), generator=g, device=dev)
        x1 = x_train[idx]; cc = None if c_train is None else c_train[idx]
        x0 = torch.randn(x1.shape, generator=g, device=dev, dtype=x1.dtype)
        t = torch.rand(batch_size, 1, generator=g, device=dev, dtype=x1.dtype)
        xt = (1 - t) * x0 + t * x1
        loss = ((model(xt, t, cc) - (x1 - x0)) ** 2).mean()
        if not torch.isfinite(loss): raise FloatingPointError("non-finite flow loss")
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step(); sched.step(); final = float(loss.item())
        with torch.no_grad():
            for pe, p in zip(ema.parameters(), model.parameters()): pe.mul_(0.995).add_(p, alpha=.005)
        if step == 1 or step % eval_every == 0 or step == steps:
            val = _fixed_flow_val(ema, x_val, c_val, seed + 100_003)
            hist.append({"step": step, "train": final, "validation": val})
            if val < best - 1e-7:
                best, best_step = val, step
                state = {k: v.detach().cpu().clone() for k, v in ema.state_dict().items()}; stale = 0
            else: stale += 1
            if stale >= patience: break
    if state is None: state = {k: v.detach().cpu().clone() for k, v in ema.state_dict().items()}
    model.load_state_dict(state); model.eval()
    return model, TrainResult(best_step, best, final, hist)


@torch.no_grad()
def sample_flow(
    model: VectorField,
    n: int,
    *,
    condition: torch.Tensor | None = None,
    ode_steps: int = 20,
    method: str = "heun",
    seed: int = 0,
) -> torch.Tensor:
    dev = next(model.parameters()).device
    g = torch.Generator(device=dev).manual_seed(seed)
    x = torch.randn(n, model.dim, generator=g, device=dev)
    c = None if condition is None else condition.to(dev)
    dt = 1.0 / ode_steps
    model.eval()
    for k in range(ode_steps):
        t0 = torch.full((n,1), k*dt, device=dev)
        v0 = model(x, t0, c)
        if method == "euler": x = x + dt*v0
        elif method == "heun":
            xp = x + dt*v0
            t1 = torch.full((n,1), (k+1)*dt, device=dev)
            x = x + .5*dt*(v0 + model(xp, t1, c))
        else: raise ValueError(method)
    return x.cpu()


class BlockGaussianFiber(nn.Module):
    """One-shot conditional block-affine stochastic fiber."""
    def __init__(self, active_dim: int, residual_dim: int, hidden: int = 96, depth: int = 2,
                 condition_dim: int = 0, block_size: int = 2, min_scale: float = 1e-3):
        super().__init__()
        self.active_dim, self.residual_dim = active_dim, residual_dim
        self.condition_dim, self.block_size, self.min_scale = condition_dim, block_size, min_scale
        self.blocks: list[tuple[int,int]] = []
        s=0
        while s<residual_dim:
            e=min(residual_dim,s+block_size); self.blocks.append((s,e)); s=e
        self.n_chol=sum((e-s)*(e-s+1)//2 for s,e in self.blocks)
        inp=active_dim+condition_dim; layers=[]
        for _ in range(depth): layers += [nn.Linear(inp,hidden),nn.SiLU()]; inp=hidden
        self.trunk=nn.Sequential(*layers)
        self.mean=nn.Linear(inp,residual_dim); self.raw=nn.Linear(inp,self.n_chol)
        for m in self.modules():
            if isinstance(m,nn.Linear): nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)
        nn.init.zeros_(self.mean.weight); nn.init.zeros_(self.mean.bias)
        nn.init.zeros_(self.raw.weight); nn.init.zeros_(self.raw.bias)

    def _context(self,z,c):
        if self.condition_dim:
            if c is None: raise ValueError("condition required")
            z=torch.cat([z,c],-1)
        return self.trunk(z)

    def params(self,z,c=None):
        h=self._context(z,c); mu=self.mean(h); raw=self.raw(h); cursor=0; factors=[]
        for s,e in self.blocks:
            b=e-s; num=b*(b+1)//2; vals=raw[:,cursor:cursor+num]; cursor+=num
            L=torch.zeros(len(z),b,b,device=z.device,dtype=z.dtype)
            ii=torch.tril_indices(b,b,device=z.device); L[:,ii[0],ii[1]]=vals
            d=torch.arange(b,device=z.device)
            L[:,d,d]=torch.nn.functional.softplus(L[:,d,d])+self.min_scale
            mask=torch.tril(torch.ones(b,b,device=z.device,dtype=torch.bool),diagonal=-1)
            L[:,mask]=2.0*torch.tanh(L[:,mask])
            factors.append(L)
        return mu,factors

    def nll(self,r,z,c=None,reduction="mean"):
        mu,fac=self.params(z,c); centered=r-mu
        nll=torch.zeros(len(r),device=r.device,dtype=r.dtype)
        log2pi=math.log(2*math.pi)
        for L,(s,e) in zip(fac,self.blocks):
            y=torch.linalg.solve_triangular(L,centered[:,s:e,None],upper=False).squeeze(-1)
            nll += .5*(y.square().sum(-1)+(e-s)*log2pi)+torch.log(torch.diagonal(L,dim1=-2,dim2=-1)).sum(-1)
        if reduction=="none": return nll
        return nll.mean() if reduction=="mean" else nll.sum()

    @torch.no_grad()
    def sample(self,z,c=None,seed=0):
        mu,fac=self.params(z,c); g=torch.Generator(device=z.device).manual_seed(seed); out=[]
        for L,(s,e) in zip(fac,self.blocks):
            eps=torch.randn(len(z),e-s,1,generator=g,device=z.device,dtype=z.dtype)
            out.append((L@eps).squeeze(-1))
        return (mu+torch.cat(out,-1)).cpu()

    @torch.no_grad()
    def conditional_mean(self,z,c=None): return self.mean(self._context(z,c))


def evaluate_fiber(model: BlockGaussianFiber,z,r,c=None,batch=2048)->float:
    dev=next(model.parameters()).device; vals=[]; model.eval()
    with torch.no_grad():
        for s in range(0,len(z),batch):
            e=min(len(z),s+batch); cc=None if c is None else c[s:e].to(dev)
            vals.append(model.nll(r[s:e].to(dev),z[s:e].to(dev),cc,"none").cpu())
    return float(torch.cat(vals).mean().item())


def train_fiber(model: BlockGaussianFiber,ztr,rtr,zv,rv,*,ctr=None,cv=None,steps=800,batch_size=256,
                lr=2e-3,seed=0,eval_every=50,patience=10):
    seed_all(seed); dev=next(model.parameters()).device
    ztr,rtr,zv,rv=ztr.to(dev),rtr.to(dev),zv.to(dev),rv.to(dev)
    ctr=None if ctr is None else ctr.to(dev); cv=None if cv is None else cv.to(dev)
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-5); sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=max(1,steps))
    g=torch.Generator(device=dev).manual_seed(seed+29); best=float("inf"); state=None; best_step=0; stale=0; hist=[]; final=float("nan")
    for step in range(1,steps+1):
        model.train(); idx=torch.randint(0,len(ztr),(batch_size,),generator=g,device=dev); cc=None if ctr is None else ctr[idx]
        loss=model.nll(rtr[idx],ztr[idx],cc)
        if not torch.isfinite(loss): raise FloatingPointError("non-finite fiber loss")
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5); opt.step(); sched.step(); final=float(loss.item())
        if step==1 or step%eval_every==0 or step==steps:
            val=evaluate_fiber(model,zv,rv,cv); hist.append({"step":step,"train":final,"validation":val})
            if val<best-1e-7: best,best_step=val,step; state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; stale=0
            else: stale+=1
            if stale>=patience: break
    if state is None: state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    model.load_state_dict(state); model.eval(); return model,TrainResult(best_step,best,final,hist)


class Autoencoder(nn.Module):
    def __init__(self,ambient_dim:int,latent_dim:int,hidden:int=96,depth:int=2,condition_dim:int=0):
        super().__init__(); self.ambient_dim=ambient_dim; self.latent_dim=latent_dim; self.condition_dim=condition_dim
        inp=ambient_dim+condition_dim; enc=[]
        for _ in range(depth): enc += [nn.Linear(inp,hidden),nn.SiLU()]; inp=hidden
        self.enc=nn.Sequential(*enc); self.zmean=nn.Linear(inp,latent_dim); self.zlogvar=nn.Linear(inp,latent_dim)
        inp=latent_dim+condition_dim; dec=[]
        for _ in range(depth): dec += [nn.Linear(inp,hidden),nn.SiLU()]; inp=hidden
        dec += [nn.Linear(inp,ambient_dim)]; self.dec=nn.Sequential(*dec)
        for m in self.modules():
            if isinstance(m,nn.Linear): nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)

    def _cat(self,x,c): return torch.cat([x,c],-1) if self.condition_dim else x
    def encode(self,x,c=None):
        h=self.enc(self._cat(x,c)); return self.zmean(h),self.zlogvar(h).clamp(-12,8)
    def decode(self,z,c=None): return self.dec(self._cat(z,c))
    def forward(self,x,c=None,stochastic=True,g=None):
        m,lv=self.encode(x,c); eps=torch.randn(m.shape,generator=g,device=m.device,dtype=m.dtype) if stochastic else torch.zeros_like(m)
        z=m+torch.exp(.5*lv)*eps; return self.decode(z,c),m,lv,z


def _ae_eval(model,x,c,beta):
    dev=next(model.parameters()).device; model.eval()
    with torch.no_grad():
        rec,m,lv,_=model(x.to(dev),None if c is None else c.to(dev),False)
        mse=((rec-x.to(dev))**2).sum(-1).mean(); kl=.5*(m.square()+lv.exp()-lv-1).sum(-1).mean()
    return float((mse+beta*kl).item()),float(mse.item()),float(kl.item())


def train_autoencoder(model:Autoencoder,xtr,xv,*,ctr=None,cv=None,beta=.01,steps=800,batch_size=256,lr=2e-3,
                      seed=0,eval_every=50,patience=10):
    seed_all(seed); dev=next(model.parameters()).device; xtr=xtr.to(dev); xv=xv.to(dev); ctr=None if ctr is None else ctr.to(dev); cv=None if cv is None else cv.to(dev)
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-5); sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=max(1,steps)); g=torch.Generator(device=dev).manual_seed(seed+37)
    best=float("inf"); state=None; best_step=0; stale=0; hist=[]; final=float("nan")
    for step in range(1,steps+1):
        idx=torch.randint(0,len(xtr),(batch_size,),generator=g,device=dev); c=None if ctr is None else ctr[idx]
        rec,m,lv,_=model(xtr[idx],c,stochastic=beta>0,g=g); mse=((rec-xtr[idx])**2).sum(-1).mean(); kl=.5*(m.square()+lv.exp()-lv-1).sum(-1).mean(); warm=min(1.,step/max(1,.2*steps)); loss=mse+beta*warm*kl
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5); opt.step(); sched.step(); final=float(loss.item())
        if step==1 or step%eval_every==0 or step==steps:
            val,vm,vk=_ae_eval(model,xv,cv,beta); hist.append({"step":step,"train":final,"validation":val,"mse":vm,"kl":vk})
            if val<best-1e-7: best,best_step=val,step; state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; stale=0
            else: stale+=1
            if stale>=patience: break
    if state is None: state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    model.load_state_dict(state); model.eval(); return model,TrainResult(best_step,best,final,hist)


@torch.no_grad()
def encode(model:Autoencoder,x,c=None,batch=2048):
    dev=next(model.parameters()).device; out=[]
    for s in range(0,len(x),batch):
        e=min(len(x),s+batch); cc=None if c is None else c[s:e].to(dev); out.append(model.encode(x[s:e].to(dev),cc)[0].cpu())
    return torch.cat(out)


def parameter_matched_width(dim:int,condition_dim:int,target:int,depth:int=2,maximum:int=512)->int:
    best=8
    for h in range(8,maximum+1):
        if count_parameters(VectorField(dim,h,depth,condition_dim))<=target: best=h
        else: break
    return best


def time_call(fn,repeats=3):
    fn(); vals=[]
    for _ in range(repeats):
        t=time.perf_counter(); fn(); vals.append(time.perf_counter()-t)
    return float(np.median(vals))


def result_dict(x: Any) -> dict[str, Any]:
    return asdict(x) if hasattr(x,"__dataclass_fields__") else x


def _nonlinear_features(z: torch.Tensor, n_random: int = 48, seed: int = 0) -> torch.Tensor:
    z = z.detach().double().cpu()
    z = (z - z.mean(0, keepdim=True)) / z.std(0, unbiased=True, keepdim=True).clamp_min(1e-6)
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(z.shape[1], n_random, generator=g, dtype=torch.float64)
    w = w / torch.linalg.norm(w, dim=0, keepdim=True).clamp_min(1e-12)
    proj = z @ w
    return torch.cat([torch.ones(len(z), 1, dtype=z.dtype), z, z.square(), torch.tanh(proj), torch.sin(proj)], 1)


def fiber_dependence_scores(z_train,r_train,z_val,r_val,*,seed=0,n_random=48,ridge=1e-2):
    """Cross-fitted validation R2 for conditional covariance edges."""
    ptr=_nonlinear_features(z_train,n_random,seed); pva=_nonlinear_features(z_val,n_random,seed)
    mu=ptr[:,1:].mean(0,keepdim=True); sd=ptr[:,1:].std(0,unbiased=True,keepdim=True).clamp_min(1e-6)
    ptr=torch.cat([ptr[:,:1],(ptr[:,1:]-mu)/sd],1); pva=torch.cat([pva[:,:1],(pva[:,1:]-mu)/sd],1)
    m=r_train.shape[1]; pairs=[(i,j) for i in range(m) for j in range(i+1,m)]
    ytr=torch.stack([r_train[:,i].double().cpu()*r_train[:,j].double().cpu() for i,j in pairs],1)
    yva=torch.stack([r_val[:,i].double().cpu()*r_val[:,j].double().cpu() for i,j in pairs],1)
    gram=ptr.T@ptr+ridge*torch.eye(ptr.shape[1],dtype=ptr.dtype); coef=torch.linalg.solve(gram,ptr.T@ytr); pred=pva@coef; base=ytr.mean(0,keepdim=True)
    score=(1-(yva-pred).square().mean(0)/(yva-base).square().mean(0).clamp_min(1e-12)).clamp_min(0)
    out=torch.zeros(m,m,dtype=torch.float64)
    for k,(i,j) in enumerate(pairs): out[i,j]=out[j,i]=score[k]
    return out.float()


def residual_block_permutation(scores: torch.Tensor, block_size: int = 2) -> list[int]:
    import networkx as nx
    m=scores.shape[0]
    if block_size<1: raise ValueError("block_size must be positive")
    if block_size==1: return list(range(m))
    unused=set(range(m)); groups=[]
    if block_size==2:
        g=nx.Graph(); g.add_nodes_from(range(m))
        for i in range(m):
            for j in range(i+1,m): g.add_edge(i,j,weight=float(scores[i,j]))
        matching=nx.algorithms.matching.max_weight_matching(g,maxcardinality=True,weight='weight')
        groups=[sorted(list(e)) for e in matching]; used={i for gr in groups for i in gr}; groups += [[i] for i in range(m) if i not in used]
    else:
        while unused:
            i=max(unused,key=lambda a: float(scores[a,list(unused)].max()) if len(unused)>1 else -1.); group=[i]; unused.remove(i)
            while unused and len(group)<block_size:
                j=max(unused,key=lambda a: float(scores[a,group].sum())); group.append(j); unused.remove(j)
            groups.append(group)
    groups.sort(key=lambda gr:(-sum(float(scores[i,j]) for ii,i in enumerate(gr) for j in gr[ii+1:]),min(gr)))
    return [i for gr in groups for i in gr]


def refine_fiber_gauge(chart: OrthogonalChart,x_train: torch.Tensor,x_val: torch.Tensor,*,block_size=2,seed=0):
    ztr,rtr=chart.transform(x_train); zv,rv=chart.transform(x_val); scores=fiber_dependence_scores(ztr,rtr,zv,rv,seed=seed); perm=residual_block_permutation(scores,block_size)
    q=torch.cat([chart.q[:,:chart.active_dim],chart.q[:,chart.active_dim:][:,perm]],1); new=OrthogonalChart(q,chart.active_dim,chart.eigenvalues)
    blocks=[perm[i:i+block_size] for i in range(0,len(perm),block_size)]; within=sum(float(scores[i,j]) for gr in blocks for ii,i in enumerate(gr) for j in gr[ii+1:]); total=float(scores.triu(1).sum())
    return new,{"residual_permutation":perm,"within_block_score":within,"total_edge_score":total,"fraction_score_within_blocks":within/max(total,1e-12),"block_size":block_size}
