#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math, os, sys
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT/'src'))
from fiqfm.core import (Autoencoder,BlockGaussianFiber,VectorField,count_parameters,encode,evaluate_fiber,
    fit_flow_moment_chart,refine_fiber_gauge,parameter_matched_width,result_dict,sample_flow,subspace_sine,time_call,
    train_autoencoder,train_fiber,train_flow)
from fiqfm.data import synthetic_splits
from fiqfm.metrics import paired_stats,sample_metrics


@dataclass
class Config:
    seeds: tuple[int,...]=(0,1,2,3,4)
    D:int=18; d:int=2; n_train:int=5000; n_val:int=1500; n_test:int=2500
    hidden:int=80; train_steps:int=550; batch:int=384; ode_steps:int=20; sample_n:int=2500
    n_pairs:int=30000; device:str='cpu'; timing_repeats:int=2


def dump(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,sort_keys=True,default=lambda x: x.item() if hasattr(x,'item') else str(x))+'\n')


def copy_decoder(ae,decoder):
    src=[m for m in ae.dec.modules() if isinstance(m,nn.Linear)]
    dst=[m for m in decoder.trunk.modules() if isinstance(m,nn.Linear)]+[decoder.mean]
    if len(src)==len(dst) and all(a.weight.shape==b.weight.shape for a,b in zip(src,dst)):
        with torch.no_grad():
            for a,b in zip(src,dst): b.weight.copy_(a.weight); b.bias.copy_(a.bias)


def fit_flow(xtr,xv,dim,cfg,seed,hidden=None):
    torch.manual_seed(seed)
    m=VectorField(dim,hidden or cfg.hidden,2).to(cfg.device)
    return train_flow(m,xtr,xv,steps=cfg.train_steps,batch_size=cfg.batch,seed=seed,eval_every=max(10,cfg.train_steps//10),patience=8)


def fit_fiber(ztr,rtr,zv,rv,block,cfg,seed,initializer=None):
    torch.manual_seed(seed)
    m=BlockGaussianFiber(ztr.shape[1],rtr.shape[1],cfg.hidden,2,block_size=block).to(cfg.device)
    if initializer: initializer(m)
    return train_fiber(m,ztr,rtr,zv,rv,steps=cfg.train_steps,batch_size=cfg.batch,seed=seed,eval_every=max(10,cfg.train_steps//10),patience=8)


def latent_baseline(name,beta,block,split,cfg,seed):
    torch.manual_seed(seed+1)
    ae=Autoencoder(cfg.D,cfg.d,cfg.hidden,2).to(cfg.device)
    ae,aeh=train_autoencoder(ae,split.x_train,split.x_val,beta=beta,steps=cfg.train_steps,batch_size=cfg.batch,
                             seed=seed+1,eval_every=max(10,cfg.train_steps//10),patience=8)
    ztr,zv,zte=encode(ae,split.x_train),encode(ae,split.x_val),encode(ae,split.x_test)
    flow,fh=fit_flow(ztr,zv,cfg.d,cfg,seed+2)
    torch.manual_seed(seed+3)
    dec=BlockGaussianFiber(cfg.d,cfg.D,cfg.hidden,2,block_size=block).to(cfg.device); copy_decoder(ae,dec)
    dec,dh=train_fiber(dec,ztr,split.x_train,zv,split.x_val,steps=cfg.train_steps,batch_size=cfg.batch,seed=seed+3,
                       eval_every=max(10,cfg.train_steps//10),patience=8)
    zg=sample_flow(flow,cfg.sample_n,ode_steps=cfg.ode_steps,seed=seed+4).to(cfg.device)
    xg=dec.sample(zg,seed=seed+5)
    details={"autoencoder":result_dict(aeh),"flow":result_dict(fh),"decoder":result_dict(dh),
             "test_decoder_nll":evaluate_fiber(dec,zte,split.x_test),"beta":beta}
    counts={"training_total":count_parameters(ae)+count_parameters(flow)+count_parameters(dec),
            "generation_total":count_parameters(flow)+count_parameters(dec)}
    return xg,details,counts


def run_seed(seed,cfg,out):
    dist,split,truth=synthetic_splits(seed,cfg.n_train,cfg.n_val,cfg.n_test,cfg.D,cfg.d)
    chart,cd=fit_flow_moment_chart(split.x_train,cfg.d,seed=seed+10,n_pairs=cfg.n_pairs)
    chart,gd=refine_fiber_gauge(chart,split.x_train,split.x_val,block_size=2,seed=seed+11)
    ztr,rtr=chart.transform(split.x_train); zv,rv=chart.transform(split.x_val); zte,rte=chart.transform(split.x_test)
    qerr=subspace_sine(chart.q,dist.q,cfg.d)

    zflow,zfh=fit_flow(ztr,zv,cfg.d,cfg,seed+20)
    block,bh=fit_fiber(ztr,rtr,zv,rv,2,cfg,seed+30)
    diag,dh=fit_fiber(ztr,rtr,zv,rv,1,cfg,seed+40)
    zg=sample_flow(zflow,cfg.sample_n,ode_steps=cfg.ode_steps,seed=seed+50).to(cfg.device)
    fiq=chart.inverse(zg.cpu(),block.sample(zg,seed+51)); fiqd=chart.inverse(zg.cpu(),diag.sample(zg,seed+52))

    fiq_budget=count_parameters(zflow)+count_parameters(block)
    full_hidden=parameter_matched_width(cfg.D,0,fiq_budget,2)
    full,fullh=fit_flow(split.x_train,split.x_val,cfg.D,cfg,seed+60,hidden=full_hidden)
    fullg=sample_flow(full,cfg.sample_n,ode_steps=cfg.ode_steps,seed=seed+61)

    vaeg,vaeh,vaep=latent_baseline('vae_lfm_diagonal',1e-2,1,split,cfg,seed+100)
    raeg,raeh,raep=latent_baseline('rae_lfm_block',0.0,2,split,cfg,seed+200)

    samples={'fiq_fm_block':fiq,'fiq_fm_diagonal_ablation':fiqd,'full_fm_param_matched':fullg,
             'vae_lfm_diagonal':vaeg,'rae_lfm_block':raeg}
    metric_seed=seed+301
    metrics={m:sample_metrics(split.x_test,x,metric_seed) for m,x in samples.items()}
    timing_n=min(1000,cfg.sample_n)
    def gen_fiq():
        z=sample_flow(zflow,timing_n,ode_steps=cfg.ode_steps,seed=seed+700).to(cfg.device)
        return chart.inverse(z.cpu(),block.sample(z,seed+701))
    def gen_full(): return sample_flow(full,timing_n,ode_steps=cfg.ode_steps,seed=seed+702)
    timings={'n_samples':timing_n,'fiq_fm_block':time_call(gen_fiq,cfg.timing_repeats),
             'full_fm_param_matched':time_call(gen_full,cfg.timing_repeats)}
    res={'seed':seed,'chart':{**cd,'fiber_gauge':gd,'active_subspace_sine_to_truth':qerr},
         'theoretical_diagonal_fiber_kl_gap':float(dist.diagonal_kl_gap(truth['test']['z']).mean()),
         'fiber_test_nll':{'block':evaluate_fiber(block,zte,rte),'diagonal':evaluate_fiber(diag,zte,rte)},
         'metrics':metrics,'parameters':{
             'fiq_fm_block':{'latent_flow':count_parameters(zflow),'fiber':count_parameters(block),'generation_total':fiq_budget},
             'full_fm_param_matched':{'hidden':full_hidden,'generation_total':count_parameters(full)},
             'vae_lfm_diagonal':vaep,'rae_lfm_block':raep},
         'timing_seconds':timings,'training':{'fiq_flow':result_dict(zfh),'fiq_block':result_dict(bh),'fiq_diag':result_dict(dh),
             'full':result_dict(fullh),'vae':vaeh,'rae':raeh}}
    dump(out/f'seed_{seed}.json',res)
    fig,axs=plt.subplots(2,3,figsize=(10,6),constrained_layout=True)
    for ax,(name,x) in zip(axs.ravel(),[('held-out target',split.x_test),*samples.items()]):
        y=x@dist.q; ax.scatter(y[:1000,0],y[:1000,1],s=3,alpha=.3); ax.set_title(name); ax.set_xlabel('active 1'); ax.set_ylabel('active 2')
    fig.savefig(out/f'seed_{seed}_active.png',dpi=150); plt.close(fig)
    return res


def aggregate(results,out):
    methods=list(results[0]['metrics']); mets=list(results[0]['metrics'][methods[0]]); rows=[]; summary={'n_seeds':len(results),'methods':{},'paired_against':{}}
    for m in methods:
        summary['methods'][m]={}
        for k in mets:
            vals=[r['metrics'][m][k] for r in results]; summary['methods'][m][k]={'mean':float(np.mean(vals)),'se':float(np.std(vals,ddof=1)/math.sqrt(len(vals))) if len(vals)>1 else 0.,'values':vals}; rows += [{'seed':r['seed'],'method':m,'metric':k,'value':r['metrics'][m][k]} for r in results]
    for b in methods[1:]:
        summary['paired_against'][b]={k:paired_stats([r['metrics']['fiq_fm_block'][k] for r in results],[r['metrics'][b][k] for r in results],True,17) for k in mets}
    summary['chart_sine']=[r['chart']['active_subspace_sine_to_truth'] for r in results]
    summary['learned_fiber_nll_advantage']=[r['fiber_test_nll']['diagonal']-r['fiber_test_nll']['block'] for r in results]
    summary['theory_fiber_kl_gap']=[r['theoretical_diagonal_fiber_kl_gap'] for r in results]
    dump(out/'summary.json',summary); pd.DataFrame(rows).to_csv(out/'metrics_long.csv',index=False)
    fig,ax=plt.subplots(figsize=(9,4),constrained_layout=True); means=[summary['methods'][m]['sliced_w2']['mean'] for m in methods]; se=[summary['methods'][m]['sliced_w2']['se'] for m in methods]; ax.bar(range(len(methods)),means,yerr=se,capsize=3); ax.set_xticks(range(len(methods)),methods,rotation=25,ha='right'); ax.set_ylabel('held-out sliced W2'); fig.savefig(out/'summary_sliced_w2.png',dpi=170); plt.close(fig)
    return summary


def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,default=ROOT/'results/synthetic_exact'); p.add_argument('--seeds',type=int,nargs='+',default=[0,1,2,3,4]); p.add_argument('--smoke',action='store_true'); p.add_argument('--device',default='cpu'); a=p.parse_args()
    torch.set_num_threads(min(4,os.cpu_count() or 1))
    cfg=Config(seeds=tuple(a.seeds),device=a.device)
    if a.smoke: cfg=Config(seeds=tuple(a.seeds),D=18,d=2,n_train=1000,n_val=300,n_test=500,hidden=32,train_steps=40,batch=128,ode_steps=4,sample_n=300,n_pairs=3000,device=a.device,timing_repeats=1)
    a.output.mkdir(parents=True,exist_ok=True); dump(a.output/'config.json',asdict(cfg)); results=[]
    for s in cfg.seeds: print(f'[synthetic] seed {s}',flush=True); results.append(run_seed(s,cfg,a.output))
    summary=aggregate(results,a.output); print(json.dumps({m:summary['methods'][m]['sliced_w2']['mean'] for m in summary['methods']},indent=2))
if __name__=='__main__': main()
