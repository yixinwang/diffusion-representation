#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,math,os,sys
from dataclasses import asdict,dataclass
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from torch import nn

ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT/'src'))
from fiqfm.core import (Autoencoder,BlockGaussianFiber,VectorField,count_parameters,encode,evaluate_fiber,
    fit_flow_moment_chart,refine_fiber_gauge,parameter_matched_width,result_dict,sample_flow,time_call,train_autoencoder,train_fiber,train_flow)
from fiqfm.data import digits_splits
from fiqfm.metrics import frechet,knn_pr,paired_stats,sample_metrics

@dataclass
class Config:
    seeds:tuple[int,...]=(0,1,2,3,4); d:int=16; hidden:int=96; train_steps:int=500; batch:int=256
    ode_steps:int=20; n_pairs:int=20000; block:int=4; device:str='cpu'; timing_repeats:int=2

def dump(p,o): p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(o,indent=2,sort_keys=True,default=lambda x:x.item() if hasattr(x,'item') else str(x))+'\n')

def copy_decoder(ae,dec):
    src=[m for m in ae.dec.modules() if isinstance(m,nn.Linear)]; dst=[m for m in dec.trunk.modules() if isinstance(m,nn.Linear)]+[dec.mean]
    if len(src)==len(dst) and all(a.weight.shape==b.weight.shape for a,b in zip(src,dst)):
        with torch.no_grad():
            for a,b in zip(src,dst): b.weight.copy_(a.weight); b.bias.copy_(a.bias)

def fit_flow(xtr,xv,dim,cfg,seed,ctr,cv,hidden=None):
    m=VectorField(dim,hidden or cfg.hidden,2,10).to(cfg.device)
    return train_flow(m,xtr,xv,c_train=ctr,c_val=cv,steps=cfg.train_steps,batch_size=cfg.batch,seed=seed,eval_every=max(10,cfg.train_steps//10),patience=8)

def fit_fiber(ztr,rtr,zv,rv,block,cfg,seed,ctr,cv,initializer=None):
    m=BlockGaussianFiber(ztr.shape[1],rtr.shape[1],cfg.hidden,2,condition_dim=10,block_size=block).to(cfg.device)
    if initializer: initializer(m)
    return train_fiber(m,ztr,rtr,zv,rv,ctr=ctr,cv=cv,steps=cfg.train_steps,batch_size=cfg.batch,seed=seed,eval_every=max(10,cfg.train_steps//10),patience=8)

class EvalNet(nn.Module):
    def __init__(self):
        super().__init__(); self.f=nn.Sequential(nn.Linear(64,96),nn.SiLU(),nn.Linear(96,32),nn.SiLU()); self.h=nn.Linear(32,10)
    def features(self,x): return self.f(x)
    def forward(self,x): return self.h(self.features(x))

def train_evaluator(split,seed,device):
    torch.manual_seed(seed); m=EvalNet().to(device); opt=torch.optim.AdamW(m.parameters(),lr=2e-3,weight_decay=1e-4); g=torch.Generator(device=device).manual_seed(seed+91); best=-1.; state=None; stale=0
    xtr,ytr=split.x_train.to(device),split.y_train.to(device); xv,yv=split.x_val.to(device),split.y_val.to(device)
    for step in range(1,1001):
        idx=torch.randint(0,len(xtr),(256,),generator=g,device=device); loss=nn.functional.cross_entropy(m(xtr[idx]),ytr[idx],label_smoothing=.02); opt.zero_grad(); loss.backward(); opt.step()
        if step%50==0:
            with torch.no_grad(): acc=float((m(xv).argmax(1)==yv).float().mean())
            if acc>best+1e-6: best=acc; state={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}; stale=0
            else: stale+=1
            if stale>=8: break
    m.load_state_dict(state); m.eval(); return m,best
@torch.no_grad()
def features(m,x): return m.features(x.to(next(m.parameters()).device)).cpu()
@torch.no_grad()
def label_acc(m,x,y): return float((m(x.to(next(m.parameters()).device)).argmax(1).cpu()==y).float().mean())

def linear_probe(ztr,ytr,zv,yv,zt,yt):
    best=None
    for C in [.03,.1,.3,1,3,10]:
        lr=LogisticRegression(C=C,max_iter=2000,solver='lbfgs').fit(ztr.numpy(),ytr.numpy()); acc=lr.score(zv.numpy(),yv.numpy())
        if best is None or acc>best[0]: best=(acc,lr,C)
    return {'val_accuracy':float(best[0]),'test_accuracy':float(best[1].score(zt.numpy(),yt.numpy())),'C':best[2]}

def class_frechet(real_f,gen_f,y):
    vals=[]
    for k in range(10): vals.append(frechet(real_f[y==k],gen_f[y==k]))
    return float(np.mean(vals)),vals

def latent_baseline(beta,block,split,cfg,seed):
    ae=Autoencoder(64,cfg.d,cfg.hidden,2,10).to(cfg.device)
    ae,aeh=train_autoencoder(ae,split.x_train,split.x_val,ctr=split.c_train,cv=split.c_val,beta=beta,steps=cfg.train_steps,batch_size=cfg.batch,seed=seed+1,eval_every=max(10,cfg.train_steps//10),patience=8)
    ztr,zv,zt=encode(ae,split.x_train,split.c_train),encode(ae,split.x_val,split.c_val),encode(ae,split.x_test,split.c_test)
    flow,fh=fit_flow(ztr,zv,cfg.d,cfg,seed+2,split.c_train,split.c_val)
    dec=BlockGaussianFiber(cfg.d,64,cfg.hidden,2,condition_dim=10,block_size=block).to(cfg.device); copy_decoder(ae,dec)
    dec,dh=train_fiber(dec,ztr,split.x_train,zv,split.x_val,ctr=split.c_train,cv=split.c_val,steps=cfg.train_steps,batch_size=cfg.batch,seed=seed+3,eval_every=max(10,cfg.train_steps//10),patience=8)
    zg=sample_flow(flow,len(split.x_test),condition=split.c_test,ode_steps=cfg.ode_steps,seed=seed+4).to(cfg.device); xg=dec.sample(zg,split.c_test.to(cfg.device),seed+5)
    with torch.no_grad(): mean=dec.conditional_mean(zt.to(cfg.device),split.c_test.to(cfg.device)).cpu(); recon=float(((mean-split.x_test)**2).mean())
    return xg,{'autoencoder':result_dict(aeh),'flow':result_dict(fh),'decoder':result_dict(dh),'test_decoder_nll':evaluate_fiber(dec,zt,split.x_test,split.c_test),'z_mean_reconstruction_mse':recon,'linear_probe':linear_probe(ztr,split.y_train,zv,split.y_val,zt,split.y_test),'beta':beta}, {'training_total':count_parameters(ae)+count_parameters(flow)+count_parameters(dec),'generation_total':count_parameters(flow)+count_parameters(dec)}

def run_seed(seed,cfg,out):
    split,standardizer,metadata=digits_splits(seed); evaluator,evaluator_val=train_evaluator(split,seed+500,cfg.device)
    chart,cd=fit_flow_moment_chart(split.x_train,cfg.d,seed=seed+10,n_pairs=cfg.n_pairs); chart,gd=refine_fiber_gauge(chart,split.x_train,split.x_val,block_size=cfg.block,seed=seed+11); ztr,rtr=chart.transform(split.x_train); zv,rv=chart.transform(split.x_val); zt,rt=chart.transform(split.x_test)
    zflow,zfh=fit_flow(ztr,zv,cfg.d,cfg,seed+20,split.c_train,split.c_val)
    block,bh=fit_fiber(ztr,rtr,zv,rv,cfg.block,cfg,seed+30,split.c_train,split.c_val); diag,dh=fit_fiber(ztr,rtr,zv,rv,1,cfg,seed+40,split.c_train,split.c_val)
    zg=sample_flow(zflow,len(split.x_test),condition=split.c_test,ode_steps=cfg.ode_steps,seed=seed+50).to(cfg.device); fiq=chart.inverse(zg.cpu(),block.sample(zg,split.c_test.to(cfg.device),seed+51)); fiqd=chart.inverse(zg.cpu(),diag.sample(zg,split.c_test.to(cfg.device),seed+52))
    with torch.no_grad(): rmean=block.conditional_mean(zt.to(cfg.device),split.c_test.to(cfg.device)).cpu(); fiq_recon=float(((chart.inverse(zt,rmean)-split.x_test)**2).mean())
    budget=count_parameters(zflow)+count_parameters(block); full_hidden=parameter_matched_width(64,10,budget,2); full=VectorField(64,full_hidden,2,10).to(cfg.device); full,fullh=train_flow(full,split.x_train,split.x_val,c_train=split.c_train,c_val=split.c_val,steps=cfg.train_steps,batch_size=cfg.batch,seed=seed+60,eval_every=max(10,cfg.train_steps//10),patience=8); fullg=sample_flow(full,len(split.x_test),condition=split.c_test,ode_steps=cfg.ode_steps,seed=seed+61)
    vaeg,vaeh,vaep=latent_baseline(1e-2,1,split,cfg,seed+100); raeg,raeh,raep=latent_baseline(0.,cfg.block,split,cfg,seed+200)
    samples={'fiq_fm_block':fiq,'fiq_fm_diagonal_ablation':fiqd,'full_fm_param_matched':fullg,'vae_lfm_diagonal':vaeg,'rae_lfm_block':raeg}; realf=features(evaluator,split.x_test); metrics={}
    for i,(name,x) in enumerate(samples.items()):
        f=features(evaluator,x); p,r=knn_pr(realf,f,seed=seed+1000+i); cf,cfl=class_frechet(realf,f,split.y_test); metrics[name]={**sample_metrics(split.x_test,x,seed+2000+i),'feature_frechet':frechet(realf,f),'class_feature_frechet':cf,'precision':p,'recall':r,'requested_label_accuracy':label_acc(evaluator,x,split.y_test)}
    timing_n=len(split.x_test)
    def gf():
        zz=sample_flow(zflow,timing_n,condition=split.c_test,ode_steps=cfg.ode_steps,seed=seed+700).to(cfg.device); return chart.inverse(zz.cpu(),block.sample(zz,split.c_test.to(cfg.device),seed+701))
    def gfull(): return sample_flow(full,timing_n,condition=split.c_test,ode_steps=cfg.ode_steps,seed=seed+702)
    res={'seed':seed,'data':{'split_sizes':[len(split.x_train),len(split.x_val),len(split.x_test)],'metadata':metadata},'evaluator':{'validation_accuracy':evaluator_val,'test_accuracy_real':label_acc(evaluator,split.x_test,split.y_test)},'chart':{**cd,'fiber_gauge':gd},'fiber_test_nll':{'block':evaluate_fiber(block,zt,rt,split.c_test),'diagonal':evaluate_fiber(diag,zt,rt,split.c_test)},'representation':{'fiq':{'z_mean_reconstruction_mse':fiq_recon,'linear_probe':linear_probe(ztr,split.y_train,zv,split.y_val,zt,split.y_test)},'vae':vaeh['linear_probe'],'rae':raeh['linear_probe']},'metrics':metrics,'parameters':{'fiq_fm_block':{'generation_total':budget},'full_fm_param_matched':{'hidden':full_hidden,'generation_total':count_parameters(full)},'vae_lfm_diagonal':vaep,'rae_lfm_block':raep},'timing_seconds':{'n_samples':timing_n,'fiq_fm_block':time_call(gf,cfg.timing_repeats),'full_fm_param_matched':time_call(gfull,cfg.timing_repeats)},'training':{'fiq_flow':result_dict(zfh),'fiq_block':result_dict(bh),'fiq_diag':result_dict(dh),'full':result_dict(fullh),'vae':vaeh,'rae':raeh}}
    dump(out/f'seed_{seed}.json',res)
    fig,axs=plt.subplots(6,11,figsize=(11,6),constrained_layout=True)
    order=[('real',split.x_test),*samples.items()]
    for row,(name,x) in enumerate(order):
        raw=standardizer.inverse(x).reshape(-1,8,8).clamp(0,1)
        for col in range(10): axs[row,col].imshow(raw[col].numpy(),cmap='gray',vmin=0,vmax=1); axs[row,col].axis('off')
        axs[row,10].text(.05,.5,name,fontsize=8); axs[row,10].axis('off')
    fig.savefig(out/f'seed_{seed}_samples.png',dpi=170); plt.close(fig); return res

def aggregate(results,out):
    methods=list(results[0]['metrics']); mets=list(results[0]['metrics'][methods[0]]); rows=[]; s={'n_seeds':len(results),'methods':{},'paired_against':{}}
    for m in methods:
        s['methods'][m]={}
        for k in mets:
            v=[r['metrics'][m][k] for r in results]; s['methods'][m][k]={'mean':float(np.mean(v)),'se':float(np.std(v,ddof=1)/math.sqrt(len(v))) if len(v)>1 else 0.,'values':v}; rows += [{'seed':r['seed'],'method':m,'metric':k,'value':r['metrics'][m][k]} for r in results]
    for b in methods[1:]: s['paired_against'][b]={k:paired_stats([r['metrics']['fiq_fm_block'][k] for r in results],[r['metrics'][b][k] for r in results],lower=(k!='precision' and k!='recall' and k!='requested_label_accuracy'),seed=23) for k in mets}
    s['representation']={m:{'linear_probe_test_mean':float(np.mean([r['representation'][m]['linear_probe']['test_accuracy'] for r in results]))} for m in ['fiq','vae','rae']}
    dump(out/'summary.json',s); pd.DataFrame(rows).to_csv(out/'metrics_long.csv',index=False)
    fig,ax=plt.subplots(figsize=(9,4),constrained_layout=True); vals=[s['methods'][m]['class_feature_frechet']['mean'] for m in methods]; err=[s['methods'][m]['class_feature_frechet']['se'] for m in methods]; ax.bar(range(len(methods)),vals,yerr=err,capsize=3); ax.set_xticks(range(len(methods)),methods,rotation=25,ha='right'); ax.set_ylabel('class-conditional feature Fréchet'); fig.savefig(out/'summary_feature_frechet.png',dpi=170); plt.close(fig); return s

def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,default=ROOT/'results/sklearn_digits'); p.add_argument('--seeds',type=int,nargs='+',default=[0,1,2,3,4]); p.add_argument('--smoke',action='store_true'); p.add_argument('--device',default='cpu'); a=p.parse_args(); torch.set_num_threads(min(4,os.cpu_count() or 1)); cfg=Config(tuple(a.seeds),device=a.device)
    if a.smoke: cfg=Config(tuple(a.seeds),d=12,hidden=32,train_steps=35,batch=128,ode_steps=4,n_pairs=2500,block=4,device=a.device,timing_repeats=1)
    a.output.mkdir(parents=True,exist_ok=True); dump(a.output/'config.json',asdict(cfg)); rs=[]
    for seed in cfg.seeds: print(f'[digits] seed {seed}',flush=True); rs.append(run_seed(seed,cfg,a.output))
    ss=aggregate(rs,a.output); print(json.dumps({m:ss['methods'][m]['class_feature_frechet']['mean'] for m in ss['methods']},indent=2))
if __name__=='__main__': main()
