"""Seven-arm response factorial: frozen development protocol, no speed guarantee."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import signal
import subprocess
import sys
import time
import traceback
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'qalt/experiments/observed_flow_pilot'))
sys.path.insert(0,str(ROOT/'qalt/experiments/perceptual_appendix'))
import run_shared as utility
import evaluator
import metrics
from qalt.cached_global_innovation import CachedGlobalInnovationDecoder
from qalt.innovation_response import InnovationResponseFlow, InnovationResponseDecoder
from qalt.dense_global_conditional_spline import DenseGlobalConditionalSplineDecoder
from qalt.observed_flow_data import load_observed_flow_data

SEEDS=(78201,78202,78203)
ARMS=('P_frozen','P_joint','I_frozen','I_joint','RQS_frozen','RQS_joint','S42')
FAMILIES=('P','I','RQS')
DATA_ROOT=Path('/ocean/datasets/community/cifar/cifar-10/cifar-10-batches-py')
BATCH=32
DIMENSION=3072
LR=1e-3


def sources():
    files=list((ROOT/'qalt/src/qalt').rglob('*.py'))+list((ROOT/'qalt/tests').glob('*.py'))
    files += [ROOT/'qalt/data/observed_manifest_v1.json']
    for directory in ('innovation_response_pilot_v2','perceptual_appendix'):
        files += [p for p in (ROOT/f'qalt/experiments/{directory}').iterdir() if p.suffix in ('.py','.md','.slurm','.json')]
    files += [ROOT/'qalt/experiments/observed_flow_pilot/run_shared.py',ROOT/'qalt/experiments/observed_flow_pilot/SHARED_PROTOCOL.md']
    return sorted(set(files))


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(2**20),b''):h.update(chunk)
    return h.hexdigest()


QUALIFICATION = Path(__file__).with_name('qualification.json')
QUALIFICATION_RECEIPT = Path(__file__).with_name('qualification_receipt.json')
KERNEL_PATH = ROOT/'qalt/src/qalt/reflected_dense_spline.py'


def validate_qualification():
    """Fail closed before canonical access; root must populate reviewed GPU evidence."""
    q=json.loads(QUALIFICATION.read_text())
    if type(q.get('qualified')) is not bool or q['qualified'] is not True:
        raise ValueError('v2 GPU qualification is pending or invalid')
    def hex_value(value,length):
        return isinstance(value,str) and len(value)==length and all(c in '0123456789abcdef' for c in value)
    if type(q.get('schema')) is not int or q.get('schema')!=1 or q.get('backend')!='dense_reflected':raise ValueError('qualification schema/backend mismatch')
    if q.get('independent_review_verified') is not True or type(q.get('independent_review_verified')) is not bool:
        raise ValueError('independent GPU review required')
    if not hex_value(q.get('kernel_sha256'),64) or q['kernel_sha256']!=digest(KERNEL_PATH):
        raise ValueError('qualified kernel source mismatch')
    if not hex_value(q.get('diagnostic_source_commit'),40) or not hex_value(q.get('diagnostic_receipt_sha256'),64):
        raise ValueError('invalid diagnostic provenance hashes')
    if q.get('original_failure_job_id')!='45582364' or not isinstance(q.get('diagnostic_job_id'),str) or not q['diagnostic_job_id'].isdigit():
        raise ValueError('invalid diagnostic job provenance')
    if digest(QUALIFICATION_RECEIPT)!=q['diagnostic_receipt_sha256']:raise ValueError('diagnostic receipt byte mismatch')
    receipt=json.loads(QUALIFICATION_RECEIPT.read_text())
    for key in ('kernel_sha256','diagnostic_source_commit','diagnostic_job_id','original_failure_job_id'):
        if receipt.get(key)!=q[key]:raise ValueError('receipt provenance mismatch: '+key)
    if receipt.get('device')!='cuda':raise ValueError('GPU diagnostic required')
    for key in ('original_failure_reproduced','candidate_valid','candidate_gradients_finite','grad_enabled','state_unchanged'):
        if type(receipt.get(key)) is not bool or receipt[key] is not True:raise ValueError('unqualified diagnostic evidence: '+key)
    for key in ('training_performed','repair_evaluated'):
        if type(receipt.get(key)) is not bool or receipt[key] is not False:raise ValueError('diagnostic scope violation: '+key)
    return q


def source_guard(commit,out):
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    if len(commit)!=40 or commit!=head:raise ValueError('expected full commit must equal HEAD')
    hashes={}
    for p in sources():
        rel=p.relative_to(ROOT);raw=p.read_bytes()
        if raw!=subprocess.check_output(['git','show',f'{commit}:{rel}'],cwd=ROOT):raise ValueError(f'unfrozen source {rel}')
        dest=out/'sources'/rel;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(raw)
        hashes[str(rel)]=hashlib.sha256(raw).hexdigest()
    for name,module in list(sys.modules.items()):
        if name=='qalt' or name.startswith('qalt.'):
            path=Path(module.__file__).resolve()
            if path not in sources():raise ValueError(f'unguarded qalt import {name}')
    for module,relative in ((utility,'observed_flow_pilot/run_shared.py'),(evaluator,'perceptual_appendix/evaluator.py'),(metrics,'perceptual_appendix/metrics.py')):
        if Path(module.__file__).resolve()!=ROOT/f'qalt/experiments/{relative}':raise ValueError('utility import mismatch')
    utility.atomic_json(out/'source_identity.json',dict(commit=commit,sha256=hashes))


def payload(out):
    return {str(p.relative_to(out)):digest(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name not in ('status.json','failure.json')}


def checkpoint(path,model,optimizer=None,rng=None):
    temporary=path.with_suffix('.tmp')
    torch.save({'architecture':model_spec(model),'model':model.state_dict(),'optimizer':None if optimizer is None else optimizer.state_dict(),
                'rng':None if rng is None else rng.bit_generator.state},temporary)
    os.replace(temporary,path)


def finite_step(loss,parameters,optimizer):
    """Three optimizer host finite decisions; model-internal checks remain."""
    if loss.ndim!=0 or not bool(torch.isfinite(loss)):raise FloatingPointError('nonfinite/nonscalar loss')
    loss.backward()
    if any(p.grad is None for p in parameters):raise RuntimeError('missing expected parameter gradient')
    if not bool(torch.stack([torch.isfinite(p.grad).all() for p in parameters]).all()):raise FloatingPointError('nonfinite gradient; optimizer not stepped')
    optimizer.step()
    if not bool(torch.stack([torch.isfinite(p).all() for p in parameters]).all()):raise FloatingPointError('nonfinite updated parameter; failed state retained')


def train_until(model,optimizer,rng,loss_function,ids,deadline,out,name,state):
    params=[p for group in optimizer.param_groups for p in group['params']]
    report={'updates':0,'losses':[],'max_step_seconds':0.,'status':'running'}
    h=hashlib.sha256();start=time.perf_counter();state['phase']=name
    device=next(model.parameters()).device
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
    try:
        while time.perf_counter()<deadline:
            begin=time.perf_counter();index=rng.integers(0,len(ids),size=BATCH)
            h.update(np.asarray(ids[index],dtype='<i8').tobytes());optimizer.zero_grad(set_to_none=True)
            loss=loss_function(torch.as_tensor(index,device=next(model.parameters()).device))
            finite_step(loss,params,optimizer)
            utility.synchronize(next(model.parameters()).device)
            report['updates']+=1;report['losses'].append(float(loss.detach()))
            report['max_step_seconds']=max(report['max_step_seconds'],time.perf_counter()-begin)
            if report['updates']%10==0:utility.atomic_json(out/f'{name}_progress.json',report)
        if report['updates']==0:raise RuntimeError('zero-update stage; no budget extension')
        report['status']='completed'
    except BaseException:
        report['status']='failed'
        checkpoint(out/f'{name}_failed.pt',model,optimizer,rng)
        raise
    finally:
        report.update(elapsed_seconds=time.perf_counter()-start,deadline_overrun_seconds=max(0.,time.perf_counter()-deadline),record_order_sha256=h.hexdigest(),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None,
            peak_reserved_bytes=torch.cuda.max_memory_reserved(device) if device.type=='cuda' else None)
        utility.atomic_json(out/f'{name}_progress.json',report)
    return report


def analysis_parts(model,logits):
    mixed,first=model.pre_analysis(logits);code,second=model.analysis.encode(mixed)
    coarse,residual=model._split(code)
    return coarse,residual,first+second


def analysis_loss(model,x):
    c,r,ld=analysis_parts(model,x)
    return (.5*(c.square().flatten(1).sum(1)+r.square().flatten(1).sum(1)+DIMENSION*math.log(2*math.pi))-ld).mean()/DIMENSION


def root_loss(model,coarse):
    z,ld=model.coarse_decoder.encode(coarse,model._zero(coarse))
    return (.5*(z.square()+math.log(2*math.pi)).flatten(1).sum(1)-ld).mean()/model.latent_dimension


def freeze(model):
    model.eval()
    for p in model.parameters():p.requires_grad_(False);p.grad=None


def activate(model,which):
    freeze(model)
    parameters=list(model._analysis_parameters()) if which=='analysis' else list(getattr(model,which).parameters())
    if which=='analysis':model.unfreeze_analysis()
    for p in parameters:p.requires_grad_(True)
    model.train()
    return parameters


BASE_CONFIG=dict(channels=3,size=32,levels=2,pre_layers=2,
    analysis_coarse_layers=6,analysis_detail_layers=4,coarse_layers=4,residual_layers=4,
    width=32,bins=8,attention_heads=4,innovation_rank=16)
EXPECTED_COUNTS={'P':539120,'I':539120,'RQS':589712,'S42':544080}


def prepared_model():
    return InnovationResponseFlow(**BASE_CONFIG,response_mode='prefix')


def model_spec(model):
    decoder=model.residual_decoder
    if isinstance(decoder,InnovationResponseDecoder):
        modes={r.mode for r in decoder.responses}
        if len(modes)!=1 or not modes.issubset({'prefix','innovation'}):raise ValueError('mixed response modes')
        mode=next(iter(modes));family='P' if mode=='prefix' else 'I';backend=None
    elif isinstance(decoder,DenseGlobalConditionalSplineDecoder):
        if decoder.backend!='dense_reflected':raise ValueError('only dense_reflected registered')
        family='RQS';mode=None;backend=decoder.backend
    elif isinstance(decoder,CachedGlobalInnovationDecoder) and not decoder.use_mixer:
        family='S42';mode=None;backend=None
    else:raise ValueError('unregistered residual architecture')
    if model.parameter_counts['total']!=EXPECTED_COUNTS[family]:raise ValueError('registered parameter count mismatch')
    return {'family':family,'response_mode':mode,'backend':backend,'base_config':BASE_CONFIG,
        'scalar_width':42 if family=='S42' else None,'parameter_counts':model.parameter_counts}


def state_fingerprint(model):
    h=hashlib.sha256()
    for name,tensor in sorted(model.state_dict().items()):
        value=tensor.detach().cpu().contiguous()
        h.update(name.encode());h.update(str(value.dtype).encode());h.update(str(tuple(value.shape)).encode())
        h.update(value.numpy().tobytes())
    return h.hexdigest()


def optimizer_fingerprint(state_dict):
    h=hashlib.sha256()
    def visit(value):
        if isinstance(value,torch.Tensor):
            t=value.detach().cpu().contiguous();h.update(str(t.dtype).encode());h.update(str(tuple(t.shape)).encode());h.update(t.numpy().tobytes())
        elif isinstance(value,dict):
            for key in sorted(value,key=repr):h.update(repr(key).encode());visit(value[key])
        elif isinstance(value,(tuple,list)):
            h.update(type(value).__name__.encode())
            for item in value:visit(item)
        else:h.update(repr(value).encode())
    visit(state_dict)
    return h.hexdigest()


def make_family(template,family,seed):
    # P/I copy exactly the same shared response core/head tensors; mode differs.
    model=copy.deepcopy(template)
    if family in ('P','I'):
        for response in model.residual_decoder.responses:
            response.mode='prefix' if family=='P' else 'innovation'
    else:
        torch.manual_seed(seed+40)
        if torch.cuda.is_available():torch.cuda.manual_seed_all(seed+40)
        model.residual_decoder=(DenseGlobalConditionalSplineDecoder(45,3,8,layers=4,width=32,
            bins=8,attention_heads=4,backend='dense_reflected') if family=='RQS' else
            CachedGlobalInnovationDecoder(45,3,8,blocks=4,width=42,rank=16,bins=8,use_mixer=False))
    if family not in EXPECTED_COUNTS or model_spec(model)['family']!=family:raise ValueError('unknown family')
    freeze(model)
    return model


def restore_model(record):
    spec=record['architecture']
    if spec['base_config']!=BASE_CONFIG or spec['family'] not in EXPECTED_COUNTS:
        raise ValueError('checkpoint architecture not registered')
    model=make_family(prepared_model(),spec['family'],0)
    model.load_state_dict(record['model'],strict=True)
    if model_spec(model)!=spec:raise ValueError('checkpoint mode/backend/count mismatch')
    freeze(model)
    return model


def fork_training_state(model,optimizer,rng):
    fork=copy.deepcopy(model).cpu();opt=copy.deepcopy(optimizer.state_dict())
    for value in opt['state'].values():
        for key,tensor in value.items():
            if isinstance(tensor,torch.Tensor):value[key]=tensor.cpu()
    return fork,opt,copy.deepcopy(rng.bit_generator.state)


def fit_seed(seed,fit_logits,ids,out,common_setup,state,live):
    begin=time.perf_counter();out.mkdir();reports={};torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    model=prepared_model().cuda();live['model']=model
    params=activate(model,'analysis');optimizer=torch.optim.Adam(params,lr=LR)
    x=torch.as_tensor(fit_logits,dtype=torch.float32,device='cuda')
    rng=np.random.default_rng(seed+10)
    reports['analysis']=train_until(model,optimizer,rng,lambda i:analysis_loss(model,x[i]),ids,begin+90-common_setup,out,'analysis',state)
    checkpoint(out/'analysis.pt',model,optimizer,rng)
    analysis_charge=common_setup+time.perf_counter()-begin
    begin=time.perf_counter();model.freeze_analysis()
    # This cache belongs exclusively to the now-frozen shared A.
    with torch.no_grad():
        pairs=[analysis_parts(model,x[i:i+64]) for i in range(0,len(x),64)]
        if not bool(torch.stack([torch.isfinite(v).all() for triple in pairs for v in triple]).all()):
            torch.save(pairs,out/'failed_analysis_cache.pt')
            raise FloatingPointError('nonfinite coarse/residual/LD in complete fixed-A cache')
        coarse=torch.cat([v[0] for v in pairs]);residual=torch.cat([v[1] for v in pairs]);del pairs
    params=activate(model,'coarse_decoder');optimizer=torch.optim.Adam(params,lr=LR);rng=np.random.default_rng(seed+20)
    reports['root']=train_until(model,optimizer,rng,lambda i:root_loss(model,coarse[i]),ids,begin+90,out,'root',state)
    checkpoint(out/'shared.pt',model,optimizer,rng)
    freeze(model);template=copy.deepcopy(model).cpu()
    root_charge=time.perf_counter()-begin
    shared_charge=analysis_charge+root_charge
    # Each family pays its own real prefix, then forks model/Adam/RNG exactly.
    del optimizer,model
    arm_charges={};models={};initial_identity={};fork_identity={}
    for family in FAMILIES:
        begin=time.perf_counter();model=make_family(template,family,seed)
        initial_identity[family]={'tensor_sha256':state_fingerprint(model),'architecture':model_spec(model)}
        if family=='I' and initial_identity['I']['tensor_sha256']!=initial_identity['P']['tensor_sha256']:
            raise RuntimeError('P/I initial tensors differ')
        model=model.cuda();live['model']=model
        params=activate(model,'residual_decoder');optimizer=torch.optim.Adam(params,lr=LR);rng=np.random.default_rng(seed+30)
        residual_loss=lambda i:-model.residual_decoder.log_prob(residual[i],coarse[i]).mean()/model.residual_dimension
        prefix_name=family+'_prefix'
        reports[prefix_name]=train_until(model,optimizer,rng,residual_loss,ids,begin+90,out,prefix_name,state)
        checkpoint(out/(prefix_name+'.pt'),model,optimizer,rng)
        fork,fork_optimizer,fork_rng=fork_training_state(model,optimizer,rng)
        fork_identity[family]={'checkpoint_sha256':digest(out/(prefix_name+'.pt')),
            'tensor_sha256':state_fingerprint(fork),'rng':copy.deepcopy(fork_rng),
            'optimizer_sha256':optimizer_fingerprint(fork_optimizer),
            'optimizer_steps':[int(v['step'].item()) for v in fork_optimizer['state'].values()]}
        prefix_charge=time.perf_counter()-begin
        begin=time.perf_counter();name=family+'_frozen'
        reports[name]=train_until(model,optimizer,rng,residual_loss,ids,begin+180-prefix_charge,out,name,state)
        checkpoint(out/(name+'.pt'),model,optimizer,rng)
        freeze(model);models[name]=model.cpu();del optimizer
        arm_charges[name]=prefix_charge+time.perf_counter()-begin
        begin=time.perf_counter();joint=fork.cuda();live['model']=joint
        if state_fingerprint(joint)!=fork_identity[family]['tensor_sha256']:raise RuntimeError('fork tensor mutation')
        params=activate(joint,'residual_decoder');optimizer=torch.optim.Adam(params,lr=LR)
        optimizer.load_state_dict(fork_optimizer)
        if optimizer_fingerprint(optimizer.state_dict())!=fork_identity[family]['optimizer_sha256']:
            raise RuntimeError('decoder Adam moment/group restore mismatch')
        if [int(v['step'].item()) for v in optimizer.state.values()]!=fork_identity[family]['optimizer_steps']:
            raise RuntimeError('decoder Adam step restore mismatch')
        joint.unfreeze_analysis();optimizer.add_param_group({'params':list(joint._analysis_parameters()),'lr':LR})
        rng=np.random.default_rng();rng.bit_generator.state=copy.deepcopy(fork_rng)
        if rng.bit_generator.state!=fork_identity[family]['rng']:raise RuntimeError('fork RNG restore mismatch')
        name=family+'_joint'
        reports[name]=train_until(joint,optimizer,rng,lambda i:-joint.log_prob(x[i]).mean()/DIMENSION,
            ids,begin+180-prefix_charge,out,name,state)
        checkpoint(out/(name+'.pt'),joint,optimizer,rng)
        freeze(joint);models[name]=joint.cpu();del optimizer,fork_optimizer,model
        arm_charges[name]=prefix_charge+time.perf_counter()-begin
    begin=time.perf_counter();name='S42';arm=make_family(template,name,seed).cuda();live['model']=arm
    initial_identity[name]={'tensor_sha256':state_fingerprint(arm),'architecture':model_spec(arm)}
    params=activate(arm,'residual_decoder');optimizer=torch.optim.Adam(params,lr=LR);rng=np.random.default_rng(seed+30)
    reports[name]=train_until(arm,optimizer,rng,lambda i:-arm.residual_decoder.log_prob(residual[i],coarse[i]).mean()/arm.residual_dimension,
        ids,begin+180,out,name,state)
    checkpoint(out/(name+'.pt'),arm,optimizer,rng);freeze(arm);models[name]=arm.cpu();del optimizer
    arm_charges[name]=time.perf_counter()-begin
    if set(models)!=set(ARMS):raise RuntimeError('seven-arm inventory mismatch')
    summary={'stages':reports,'analysis_charged_seconds':analysis_charge,'root_charged_seconds':root_charge,
        'shared_charged_seconds':shared_charge,'decoder_charged_seconds':arm_charges,
        'standalone_fitting_seconds':{k:shared_charge+v for k,v in arm_charges.items()},
        'parameter_counts':{name:m.parameter_counts for name,m in models.items()},
        'initial_identity':initial_identity,'fork_identity':fork_identity,'scalar_width':42,
        'common_preparation_seconds':common_setup,'all_fits_frozen':True,
        'architecture':{name:model_spec(m) for name,m in models.items()},'backend':'dense_reflected_only_for_RQS'}
    utility.atomic_json(out/'fitting.json',summary)
    del x,coarse,residual
    return models,template,summary


@torch.no_grad()
def raw_generate(model,source,kind):
    if kind not in ARMS:raise ValueError('unknown evaluation arm')
    return model.decode(source)[0]


@torch.no_grad()
def numerical_gate(model,out,seed):
    z=torch.randn(8,DIMENSION,generator=torch.Generator().manual_seed(seed+99),dtype=torch.float32).cuda()
    y,ld=model.decode(z);back,ild=model.encode(y)
    # Copy control has no independent weights or noise, and preserves every source coordinate.
    saved=torch.load(out.parent/(out.name+'.pt'),map_location='cpu',weights_only=True)
    if saved['architecture']!=model_spec(model):raise ValueError('copy checkpoint architecture mismatch')
    clone=restore_model(saved).to(device=z.device,dtype=z.dtype)
    if hasattr(clone.residual_decoder,'prepare_inference'):clone.residual_decoder.prepare_inference()
    copied=clone.sample_from_gaussian(z)
    del clone
    np.savez(out/'numerical.npz',source=z.cpu().numpy(),logits=y.cpu().numpy(),recovered=back.cpu().numpy(),ld=ld.cpu().numpy(),inverse_ld=ild.cpu().numpy(),exact_copy=copied.cpu().numpy())
    finite=all(bool(torch.isfinite(t).all()) for t in (y,back,ld,ild,copied))
    checks={'finite':finite,'source_error':float((back-z).abs().max()),'ld_error':float((ld+ild).abs().max()),'exact_copy':bool(torch.equal(y,copied))}
    utility.atomic_json(out/'numerical.json',{k:(v if not isinstance(v,float) or math.isfinite(v) else str(v)) for k,v in checks.items()})
    if not finite or checks['source_error']>1e-3 or checks['ld_error']>1e-2 or not checks['exact_copy']:raise FloatingPointError('frozen numerical gate failed')


@torch.no_grad()
def likelihood_parts(model,logits,outer,kind):
    pieces=[]
    for first in range(0,len(logits),64):
        x=torch.as_tensor(logits[first:first+64],dtype=torch.float32,device='cuda')
        c,r,a_ld=analysis_parts(model,x)
        zc,c_ld=model.coarse_decoder.encode(c,model._zero(c))
        zr,r_ld=model.residual_decoder.encode(r,c)
        c_nll=.5*(zc.square()+math.log(2*math.pi)).flatten(1).sum(1)-c_ld
        r_nll=.5*(zr.square()+math.log(2*math.pi)).flatten(1).sum(1)-r_ld
        values=torch.stack((c_nll,r_nll,-a_ld),dim=1)
        if not bool(torch.isfinite(values).all()):raise FloatingPointError('nonfinite likelihood components')
        pieces.append(values.cpu().numpy().astype(np.float64))
    a=np.concatenate(pieces);return np.column_stack((a,-outer,a.sum(1)-outer))


def sample_metrics(generated,repair):
    fake=utility.descriptors(generated);real=utility.descriptors(repair)
    covariance=lambda a:float(a[:,2].mean()-a[:,0].mean()*a[:,1].mean())
    draws=generated.reshape(len(repair),2,-1);target=repair.reshape(len(repair),-1)
    norm=lambda a:np.linalg.norm(a,axis=1)/math.sqrt(DIMENSION)
    energy=.5*(norm(draws[:,0]-target)+norm(draws[:,1]-target)-norm(draws[:,0]-draws[:,1]))
    return {'energy_mean':float(energy.mean()),'covariance_error':abs(covariance(fake)-covariance(real)),
            'gradient_means':fake[:,3:5].mean(0).tolist(),'repair_gradient_means':real[:,3:5].mean(0).tolist()},energy,fake


def engineering_gates(reports):
    if set(reports)!=set(ARMS):raise ValueError('all seven arms required')
    s,m,j,r=(reports[k] for k in ('S42','I_frozen','I_joint','RQS_frozen'))
    real=np.asarray(j['repair_gradient_means']);grad=np.asarray(j['gradient_means'])
    gates={'I_frozen_residual_nll_better_S42':m['residual_nll']<s['residual_nll'],
      'I_frozen_covariance_error_half_S42':m['covariance_error']<=.5*s['covariance_error'],
      'I_joint_complete_nll_better_I_frozen':j['complete_nll']<m['complete_nll'],
      'I_joint_KID_better_S42':j['kid']<s['kid'],
      'I_joint_KID_better_RQS_frozen':j['kid']<r['kid'],
      'I_joint_gradient_within_10_percent':bool(np.all(np.abs(grad-real)<=.1*np.abs(real)))}
    for tail in ('frozen','joint'):
        candidate,control=reports['I_'+tail],reports['P_'+tail]
        for metric in ('complete_nll','kid'):
            gates[f'I_{tail}_{metric}_better_P_{tail}']=candidate[metric]<control[metric]
    for metric in ('complete_nll','kid'):
        gates[f'I_joint_{metric}_better_RQS_joint']=j[metric]<reports['RQS_joint'][metric]
    for arm in ('S42','I_frozen','RQS_frozen','RQS_joint','P_frozen','P_joint'):
        gates['I_joint_energy_not_worse_'+arm]=j['energy_mean']<=reports[arm]['energy_mean']+.0005
    references=[reports[k]['kid'] for k in ('RQS_frozen','RQS_joint')]
    gates['I_joint_KID_material_5pct_both_RQS']=bool(all(v>0 for v in references) and all(j['kid']<=.95*v for v in references))
    return gates


def evaluate_seed(seed,models,template,data,repair_logits,outer,out,extractor,live,state):
    source=torch.randn(2000,DIMENSION,generator=torch.Generator().manual_seed(seed+300))
    np.save(out/'common_gaussian.npy',source.numpy());reports={}
    utility.atomic_json(out/'pair_identity.json',{'source_sha256':digest(out/'common_gaussian.npy'),
        'repair_ids_sha256':hashlib.sha256(np.asarray(data.repair_ids,dtype='<i8').tobytes()).hexdigest(),
        'repair_values_sha256':hashlib.sha256(np.ascontiguousarray(data.repair).tobytes()).hexdigest(),
        'source_seed':seed+300,'dimension':DIMENSION,'pairs_per_image':1})
    state.update(seed=seed,arm='repair_reference',phase='repair_feature_extraction',chunk_start=None,completed_rows=0)
    real_features=evaluator.extract(extractor,data.repair,'cuda');np.save(out/'repair_features.npy',real_features)
    for name in ARMS:
        state.update(seed=seed,arm=name,phase='evaluation_setup',chunk_start=None,completed_rows=0)
        armout=out/name;armout.mkdir();model=models[name].cuda();live['model']=model
        if model_spec(model)['family']!=name.split('_')[0]:raise ValueError('evaluation architecture mismatch')
        if hasattr(model.residual_decoder,'prepare_inference'):model.residual_decoder.prepare_inference()
        numerical_gate(model,armout,seed)
        generated=np.lib.format.open_memmap(armout/'generated.npy',mode='w+',dtype=np.float64,shape=(2000,3,32,32))
        saved_logits=np.lib.format.open_memmap(armout/'logits.npy',mode='w+',dtype=np.float32,shape=(2000,3,32,32))
        start=time.perf_counter();completed=0;endpoint_counts={'zero':0,'one':0}
        utility.atomic_json(armout/'generation_progress.json',{'status':'running','completed_rows':0,'planned_rows':2000,'logits_written_rows':0})
        for first in range(0,2000,64):
            state.update(phase='generation',chunk_start=first,completed_rows=completed)
            utility.atomic_json(armout/'generation_progress.json',{'status':'writing_chunk','completed_rows':completed,'planned_rows':2000,'chunk_start':first,'logits_written_rows':completed})
            logits=raw_generate(model,source[first:first+64].cuda(),name)
            saved_logits[first:first+len(logits)]=logits.cpu().numpy();saved_logits.flush()
            utility.atomic_json(armout/'generation_progress.json',{'status':'logits_saved_pending_validation','completed_rows':completed,'planned_rows':2000,'chunk_start':first,'logits_written_rows':first+len(logits)})
            if not bool(torch.isfinite(logits).all()):raise FloatingPointError('nonfinite logits saved before sigmoid')
            values=logits.sigmoid();generated[first:first+len(logits)]=values.cpu().numpy().astype(np.float64);generated.flush()
            if not bool(torch.isfinite(values).all()):raise FloatingPointError('nonfinite generated pixels')
            endpoint_counts['zero']+=int((values==0).sum());endpoint_counts['one']+=int((values==1).sum())
            completed=first+len(logits);state['completed_rows']=completed
            utility.atomic_json(armout/'generation_progress.json',{'status':'completed' if completed==2000 else 'running','completed_rows':completed,'planned_rows':2000,'logits_written_rows':completed,'endpoint_counts':endpoint_counts})
        state.update(phase='repair_likelihood',chunk_start=None)
        seconds=time.perf_counter()-start
        parts=likelihood_parts(model,repair_logits,outer,name);np.save(armout/'repair_nll_components.npy',parts)
        measurements,energy,descriptors=sample_metrics(generated,data.repair)
        np.save(armout/'energy.npy',energy);np.save(armout/'descriptors.npy',descriptors)
        state['phase']='feature_extraction'
        features=evaluator.extract(extractor,generated,'cuda');np.save(armout/'features.npy',features)
        measurements.update(kid=metrics.polynomial_kid(real_features,features),prdc=metrics.prdc(real_features,features),sigmoid_endpoint_counts=endpoint_counts,complete_nll=float(parts[:,-1].mean()/DIMENSION),residual_nll=float(parts[:,1].mean()/2880),generation_seconds_including_save=seconds)
        reports[name]=measurements;utility.atomic_json(armout/'metrics.json',measurements)
        model.cpu();del generated,saved_logits
    gates=engineering_gates(reports)
    utility.atomic_json(out/'evaluation.json',{'arms':reports,'gates':gates,'all_gates_pass':all(gates.values()),'confirmatory':False})
    return gates


def write_fit_freeze_receipt(out,all_models):
    if set(all_models)!=set(SEEDS) or any(set(models)!=set(ARMS) for models,_ in all_models.values()):
        raise ValueError('all seeds and seven models must fit before repair access')
    checkpoints={}
    for seed,(models,_) in all_models.items():
        for arm,model in models.items():
            if model.training or any(p.requires_grad for p in model.parameters()):raise ValueError('unfrozen final model')
            path=out/f'seed_{seed}'/(arm+'.pt');checkpoints[f'{seed}/{arm}']={'sha256':digest(path),'architecture':model_spec(model)}
    utility.atomic_json(out/'ALL_FITS_FROZEN.json',{'before_any_repair_evaluation':True,'checkpoints':checkpoints,'seeds':SEEDS,'arms':ARMS})


def fail(out,state,live,error):
    record={'status':'failed','state':state,'error':repr(error),'traceback':traceback.format_exc()}
    try:
        if 'model' in live:checkpoint(out/'failed_model.pt',live['model'])
    except BaseException as second:record['checkpoint_error']=repr(second)
    utility.atomic_json(out/'failure.json',record)
    utility.atomic_json(out/'status.json',{**record,'payload_sha256':payload(out)})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expected-commit',required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--inception-source',type=Path,required=True);parser.add_argument('--weights',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False);state={};live={};study_start=time.perf_counter()
    try:
        source_guard(args.expected_commit,args.output)
        state['phase']='gpu_qualification'
        qualification=validate_qualification()
        utility.atomic_json(args.output/'qualification_verified.json',qualification)
        evaluator.validate_artifacts(args.inception_source,args.weights)
        if not torch.cuda.is_available():raise RuntimeError('CUDA mandatory; no CPU fallback')
        torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        def stop(signum,frame):raise TimeoutError('scheduler warning; preserve failed state')
        signal.signal(signal.SIGUSR1,stop);signal.signal(signal.SIGTERM,stop)
        subprocess.run([sys.executable,'-m','pytest','-q',str(ROOT/'qalt/tests/test_innovation_response_pilot_v2.py'),str(ROOT/'qalt/tests/test_innovation_response.py'),str(ROOT/'qalt/tests/test_dense_global_conditional_spline.py'),str(ROOT/'qalt/tests/test_reflected_dense_spline.py')],check=True,cwd=ROOT,stdout=(args.output/'preflight.txt').open('w'),stderr=subprocess.STDOUT)
        state['phase']='canonical_data';start=time.perf_counter();data=load_observed_flow_data(DATA_ROOT)
        if not data.ledger['canonical_dataset_verified'] or data.ledger['allow_noncanonical_fixture'] or data.fit.shape!=(4000,3,32,32) or data.repair.shape!=(1000,3,32,32):raise ValueError('strict canonical split required')
        utility.atomic_json(args.output/'data_ledger.json',data.ledger);np.savez(args.output/'record_ids.npz',fit=data.fit_ids,repair=data.repair_ids)
        fit_logits,fit_outer=utility.logit_inputs(data.fit);np.save(args.output/'fit_outer.npy',fit_outer)
        utility.atomic_json(args.output/'fitting_input_identity.json',{'observations_sha256':hashlib.sha256(np.ascontiguousarray(data.fit).tobytes()).hexdigest(),
            'logits_sha256':hashlib.sha256(fit_logits.numpy().tobytes()).hexdigest(),'shape':list(data.fit.shape),'fit_count':4000})
        common_setup=time.perf_counter()-start
        utility.atomic_json(args.output/'environment.json',{'torch':torch.__version__,'numpy':np.__version__,'python':platform.python_version(),'gpu':torch.cuda.get_device_name(),'host':platform.node(),'seeds':SEEDS,'eager_only':True,'repair_reused':True,'confirmatory':False,'source_sha':args.expected_commit,'inception_source_sha256':evaluator.SOURCE_SHA256,'inception_weights_sha256':evaluator.WEIGHT_SHA256,'slurm_job':os.environ.get('SLURM_JOB_ID'),'torch_threads':torch.get_num_threads(),'thread_environment':{k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')}})
        # Finish every seed's fits before any repair metric or feature extraction.
        all_models={}
        for seed in SEEDS:
            state['seed']=seed;models,template,reports=fit_seed(seed,fit_logits,data.fit_ids,args.output/f'seed_{seed}',common_setup,state,live)
            all_models[seed]=(models,template)
        write_fit_freeze_receipt(args.output,all_models)
        repair_logits,outer=utility.logit_inputs(data.repair)
        extractor=evaluator.make_extractor(args.inception_source,args.weights,'cuda');gates={}
        for seed,(models,template) in all_models.items():
            state.update(seed=seed,phase='evaluation');gates[seed]=evaluate_seed(seed,models,template,data,repair_logits,outer,args.output/f'seed_{seed}',extractor,live,state)
        utility.atomic_json(args.output/'status.json',{'status':'completed_development_only','all_fits_frozen':True,'seed_gates':gates,'all_gates_pass':all(all(v.values()) for v in gates.values()),'study_wall_seconds':time.perf_counter()-study_start,'cpu_peak_rss_native_units':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'payload_sha256':payload(args.output)})
    except BaseException as error:
        fail(args.output,state,live,error);raise


if __name__=='__main__':main()
