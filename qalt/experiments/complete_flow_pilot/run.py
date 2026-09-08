"""Capped observation-only synthetic pilot; no real-image success claim.

All learners see only the same training arrays. Hidden teacher innovations,
coefficients and inversion are never passed into a learner. Audit arrays are
created after all fits. Metrics are exploratory finite-array comparisons.
"""
from __future__ import annotations
import argparse,hashlib,json,math,os,platform,subprocess,time
from pathlib import Path
import numpy as np
import torch
from qalt.multiscale_flow import MultiscaleSplineFlow,CopiedStochasticLatentDecoder,haar_split,haar_merge
from qalt.flow_matching import FullTensorFlowMatching,HierarchicalFlowMatching


def world(n,channels,size,seed,kind):
    """Smooth nonlinear non-Gaussian law, hidden from training procedures.

Fixed Haar is available to all algorithms. Coarse sinh innovations and detail
location/scale depend nonlinearly on observed coarse values. The distant
world adds paired distant-detail dependence not explained by coarse values.
This is synthetic structure, not an empirical claim about images or videos.
"""
    rng=np.random.default_rng(seed)
    coarse=torch.from_numpy(rng.normal(size=(n,channels,size//4,size//4)).astype('float32'))
    coarse=.7*torch.sinh(.7*coarse)
    for level in range(2):
        noise=torch.from_numpy(rng.normal(size=(n,3*channels,*coarse.shape[2:])).astype('float32'))
        noise=.6*torch.sinh(.6*noise)
        if kind=='distant':
            # Triangular shear: second half depends on first half, and remains
            # invertible for any real source. No circular simultaneous update.
            width=noise.shape[-1];half=width//2
            noise[...,half:]=noise[...,half:]+.8*(noise[...,:width-half].square()-.2)
        context=coarse.repeat(1,3,1,1)
        detail=.3*torch.sin(1.5*context)+(.25+.2*torch.sigmoid(context))*noise
        coarse=haar_merge(coarse,detail)
    return coarse


def sync(device):
    if device.type=='cuda':torch.cuda.synchronize(device)


def distances(x,y):
    x=x.flatten(1).double();y=y.flatten(1).double()
    return torch.cdist(x,y)/math.sqrt(x.shape[1])


def metrics(real,fake):
    n=len(fake);m=len(real)
    xx=distances(fake,fake);yy=distances(real,real);xy=distances(fake,real)
    energy=float(xy.mean()-xx.sum()/(2*n*(n-1)))
    # Fixed unit scales; no audit-selected kernel bandwidth.
    result={'energy_score':energy,'mean_rms_gap':float((real.mean(0)-fake.mean(0)).square().mean().sqrt()),'second_moment_gap':float((real.square().mean(0)-fake.square().mean(0)).abs().mean())}
    for scale in [.1,.3,1.0]:
        kxx=torch.exp(-xx.square()/(2*scale**2));kyy=torch.exp(-yy.square()/(2*scale**2));kxy=torch.exp(-xy.square()/(2*scale**2))
        result[f'mmd2_unbiased_scale_{scale}']=float((kxx.sum()-n)/(n*(n-1))+(kyy.sum()-m)/(m*(m-1))-2*kxy.mean())
    return result


def fit(model,train,seed,steps,seconds,batch,device,kind):
    sync(device);start=time.perf_counter()
    opt=torch.optim.Adam(model.parameters(),lr=1e-3)
    index_generator=torch.Generator(device=device).manual_seed(seed+400)
    generator=torch.Generator(device=device).manual_seed(seed+401)
    losses=[];count=0
    for step in range(steps):
        if step and time.perf_counter()-start>=seconds:break
        idx=torch.randint(len(train),(batch,),generator=index_generator,device=device)
        x=train[idx];opt.zero_grad(set_to_none=True)
        if kind=='spline':loss=-model.log_prob(x).mean()/model.dimension
        else:loss=model.training_loss(x,generator=generator)
        if not torch.isfinite(loss):raise FloatingPointError('nonfinite training objective')
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),10.0);opt.step()
        losses.append(float(loss.detach()));count+=1
    sync(device)
    return {'seconds':time.perf_counter()-start,'steps':count,'training_objective_first':losses[0],'training_objective_last':losses[-1],'objective':'nats_per_coordinate' if kind=='spline' else 'velocity_mse_per_coordinate','parameters':sum(p.numel() for p in model.parameters())}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);parser.add_argument('--seed',type=int,default=3100)
    parser.add_argument('--steps',type=int,default=500);parser.add_argument('--seconds',type=float,default=120)
    parser.add_argument('--size',type=int,default=8);parser.add_argument('--channels',type=int,default=1)
    parser.add_argument('--train-size',type=int,default=2048);parser.add_argument('--audit-size',type=int,default=256)
    parser.add_argument('--batch',type=int,default=64);parser.add_argument('--width',type=int,default=24)
    parser.add_argument('--world',choices=['local','distant'],default='local');parser.add_argument('--device',default='cpu')
    args=parser.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    if args.size%4 or args.size<8:raise ValueError('size must be divisible by four and at least eight')
    torch.set_num_threads(1);device=torch.device(args.device)
    repo=Path(__file__).resolve().parents[3]
    source_files=['qalt/src/qalt/spline.py','qalt/src/qalt/multiscale_flow.py','qalt/src/qalt/flow_matching.py','qalt/experiments/complete_flow_pilot/run.py','qalt/experiments/complete_flow_pilot/PROTOCOL.md']
    manifest={'config':vars(args),'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),'source_hashes':{p:hashlib.sha256((repo/p).read_bytes()).hexdigest() for p in source_files},'torch':torch.__version__,'python':platform.python_version(),'slurm_job_id':os.environ.get('SLURM_JOB_ID'),'real_data_accessed':False,'status':'synthetic_development_only','quality_advantage_established':False,'cost_advantage_established':False,'representation_advantage_established':False}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    started=time.perf_counter();train=world(args.train_size,args.channels,args.size,args.seed,args.world).to(device)
    manifest['shared_train_hash']=hashlib.sha256(train.cpu().numpy().tobytes()).hexdigest()
    models={};training={}
    constructors={'spline':lambda:MultiscaleSplineFlow(args.channels,args.size,2,4,2,args.width,8),'full_fm':lambda:FullTensorFlowMatching(args.channels,args.size,2,args.width),'hierarchical_fm':lambda:HierarchicalFlowMatching(args.channels,args.size,2,args.width)}
    for name,constructor in constructors.items():
        torch.manual_seed(args.seed+500);sync(device);init_start=time.perf_counter();model=constructor().to(device);sync(device)
        init_seconds=time.perf_counter()-init_start
        if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
        training[name]=fit(model,train,args.seed,args.steps,max(.001,args.seconds-init_seconds),args.batch,device,name)
        training[name]['initialization_seconds']=init_seconds
        training[name]['peak_training_allocated_bytes']=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None
        model.zero_grad(set_to_none=True);model.eval();model.cpu();models[name]=model
        torch.save(model.state_dict(),out/f'{name}.pt')
        print(name,training[name],flush=True)
    # A new independent synthetic audit draw only after every fitted arm freezes.
    real=world(args.audit_size,args.channels,args.size,args.seed+100000,args.world)
    generator=torch.Generator(device=device).manual_seed(args.seed+200000)
    z=torch.randn(args.audit_size,args.channels*args.size**2,generator=generator,device=device)
    evaluations={};sample_arrays={}
    with torch.no_grad():
        for name,model in models.items():
            model.to(device)
            step_grid=[None] if name=='spline' else [8,32]
            for nfe in step_grid:
                label=name if nfe is None else f'{name}_heun_{nfe}'
                # All inference grids are reported; audit metrics choose no arm.
                sync(device);begin=time.perf_counter()
                generated=model.sample_from_gaussian(z) if nfe is None else model.sample_from_gaussian(z,steps=nfe)
                sync(device);elapsed=time.perf_counter()-begin
                if not torch.isfinite(generated).all():raise FloatingPointError('nonfinite generated values')
                cpu=generated.cpu();evaluations[label]=metrics(real,cpu);evaluations[label]['generation_seconds']=elapsed
                evaluations[label]['velocity_evaluations']=0 if nfe is None else 2*nfe*(1 if name=='full_fm' else 3)
                sample_arrays[label]=cpu.numpy()
            model.cpu()
        models['spline'].to(device)
        copy=CopiedStochasticLatentDecoder(models['spline'])
        copy_samples=copy.sample_from_gaussian(z).cpu().numpy()
        exact_copy=bool(np.array_equal(copy_samples,sample_arrays['spline']))
        if not exact_copy:raise AssertionError('copied stochastic latent model must tie exactly')
        encoded,ild=models['spline'].encode(torch.from_numpy(sample_arrays['spline']).to(device))
        _,ld=models['spline'].decode(z)
        roundtrip=float((encoded-z).abs().max());logdet=float((ild+ld).abs().max())
        if roundtrip>1e-3 or logdet>1e-2:raise AssertionError(f'float32 inversion failure: source={roundtrip}, logdet={logdet}')
    np.savez_compressed(out/'generated_samples.npz',**sample_arrays)
    manifest.update({'all_numerical_checks_pass':True,'source_roundtrip_tolerance':1e-3,'logdet_cancellation_tolerance':1e-2,'training':training,'evaluation':evaluations,'exact_stochastic_latent_copy':exact_copy,'copy_training_cost':'inherits full spline training cost; no free decoder','source_roundtrip_max':roundtrip,'logdet_cancellation_max':logdet,'runtime_seconds':time.perf_counter()-started,'audit_units':'independent synthetic full tensors; no site-iid assumption','inference_timing':'single full batch, no speed superiority inference; compilation/cache and variance unresolved'})
    (out/'summary.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest,indent=2),flush=True)


if __name__=='__main__':main()
