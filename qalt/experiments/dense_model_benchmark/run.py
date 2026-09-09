"""Frozen native-checkpoint backend equivalence and timing; no fitting/data load."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import random
import signal
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[3]
NATIVE=Path('/ocean/projects/mth250006p/ywang26/diffusion-results/20260908-native-shared-retry')
STATUS_SHA='5ace9f058b21482fa02d62b453d004e93b4114093e7a7b21528582dadeccb434'
NATIVE_COMMIT='341dbaf5022a7fe54bb0e89dac6e5d5c2df82277'
MODULES=('__init__','core','spline','dense_spline','multiscale_flow','flow_matching',
         'learned_latent_flow_matching','global_conditional_spline','dense_global_conditional_spline')
SOURCES=tuple('qalt/src/qalt/'+name+'.py' for name in MODULES)+(
    'qalt/experiments/observed_flow_pilot/run_shared.py',
    'qalt/experiments/dense_model_benchmark/run.py','qalt/experiments/dense_model_benchmark/PROTOCOL.md',
    'qalt/experiments/dense_model_benchmark/run.slurm','qalt/tests/test_dense_model_benchmark_protocol.py',
    'qalt/tests/test_dense_global_conditional_spline.py','qalt/tests/test_dense_spline.py')
ARMS=('reference','dense_eager','dense_compiled')
ORDER_SEED=2026091101


def digest(path):
    result=hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda:handle.read(1048576),b''):result.update(chunk)
    return result.hexdigest()


def write(path,value):
    temporary=path.with_suffix('.tmp')
    with temporary.open('w') as handle:
        json.dump(value,handle,indent=2,allow_nan=False);handle.write('\n');handle.flush();os.fsync(handle.fileno())
    temporary.replace(path)


def source_guard(commit,out):
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()!=commit:raise ValueError('expected full commit differs from HEAD')
    hashes={};snapshot=out/'sources';snapshot.mkdir()
    for i,relative in enumerate(SOURCES):
        content=(ROOT/relative).read_bytes()
        if content!=subprocess.check_output(['git','show',f'{commit}:{relative}'],cwd=ROOT):raise ValueError('dirty source: '+relative)
        hashes[relative]=hashlib.sha256(content).hexdigest();(snapshot/f'{i:02d}_{Path(relative).name}').write_bytes(content)
    write(out/'source_identity.json',{'commit':commit,'sha256':hashes})
    return hashes


def verify_native(path):
    if digest(path/'status.json')!=STATUS_SHA:raise ValueError('terminal native status hash differs')
    status=json.loads((path/'status.json').read_text())
    if status['status']!='completed_development_only' or not status['all_fits_frozen']:raise ValueError('native run is incomplete')
    names=('source_identity.json','shared_latest.pt','coupling_latest.pt','coupling/chunk_0000.npz')
    hashes={}
    for name in names:
        hashes[name]=digest(path/name)
        if hashes[name]!=status['payload_sha256'][name]:raise ValueError('native payload hash differs: '+name)
    identity=json.loads((path/'source_identity.json').read_text())
    if identity['commit']!=NATIVE_COMMIT:raise ValueError('native source commit differs')
    for name in MODULES:
        relative='qalt/src/qalt/'+name+'.py'
        if name not in ('dense_spline','dense_global_conditional_spline') and digest(ROOT/relative)!=identity['sha256'][relative]:raise ValueError('original architecture changed since native checkpoint')
    relative='qalt/experiments/observed_flow_pilot/run_shared.py'
    if digest(ROOT/relative)!=identity['sha256'][relative]:raise ValueError('shared generator changed since native checkpoint')
    return {'terminal_status_sha256':STATUS_SHA,'native_source_commit':NATIVE_COMMIT,'accessed_payload_sha256':hashes}


def error_stats(actual,reference,*,atol,rtol=0.):
    import numpy as np
    if actual.shape!=reference.shape or not np.isfinite(actual).all() or not np.isfinite(reference).all():raise FloatingPointError('shape or finite-value equivalence failure')
    error=np.abs(actual-reference);limit=atol+rtol*np.abs(reference)
    return {'maximum_absolute_error':float(error.max(initial=0)),
            'passed':bool(np.all(error<=limit)),'atol':atol,'rtol':rtol}


def run(args,state):
    out=args.output;state['phase']='source_guard';hashes=source_guard(args.expected_commit,out)
    state['phase']='native_payload_verification'
    native=verify_native(NATIVE);write(out/'native_identity.json',native)
    for variable in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
        if os.environ.get(variable)!='1':raise ValueError(variable+'=1 required')
    for key,leaf in (('TORCHINDUCTOR_CACHE_DIR','inductor-cache'),('TRITON_CACHE_DIR','triton-cache'),('TMPDIR','tmp')):
        folder=out/leaf;folder.mkdir();os.environ[key]=str(folder.resolve())
    os.environ['TORCHINDUCTOR_COMPILE_THREADS']='1'
    sys.path.insert(0,str(ROOT/'qalt/src'))
    import numpy as np
    import torch
    from qalt.learned_latent_flow_matching import LearnedLatentFlowMatching
    from qalt.global_conditional_spline import GlobalConditionalSplineDecoder
    from qalt.dense_global_conditional_spline import DenseGlobalConditionalSplineDecoder
    for name in MODULES:
        module=sys.modules.get('qalt' if name=='__init__' else 'qalt.'+name)
        if module is None or Path(module.__file__).resolve()!=ROOT/'qalt/src/qalt'/f'{name}.py':raise ValueError('imported module differs from closure')
    spec=importlib.util.spec_from_file_location('verified_shared_generator',ROOT/'qalt/experiments/observed_flow_pilot/run_shared.py')
    utility=importlib.util.module_from_spec(spec);spec.loader.exec_module(utility)
    if not torch.cuda.is_available():raise RuntimeError('CUDA required; no CPU timing fallback')
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    device=torch.device('cuda:0')
    write(out/'metadata.json',dict(source_commit=args.expected_commit,source_sha256=hashes,
        python=sys.version,numpy=np.__version__,torch=torch.__version__,cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(0),host=platform.node(),slurm_job=os.environ.get('SLURM_JOB_ID'),
        compiler='Inductor fullgraph static scalar spline kernel inside otherwise eager full pipeline',
        environment={k:os.environ.get(k) for k in ('CC','CXX','TORCHINDUCTOR_CACHE_DIR','TRITON_CACHE_DIR','TMPDIR','TORCHINDUCTOR_COMPILE_THREADS')},
        timing_order_seed=ORDER_SEED,warmups=10,repetitions=30,new_real_data_read=False,optimizer_steps=0))
    state['phase']='checkpoint_construction';start=time.perf_counter()
    analysis=LearnedLatentFlowMatching(channels=3,size=32,levels=2,pre_layers=2,coarse_layers=6,detail_layers=4,width=32,bins=8,attention_heads=4,unit_interval=False)
    del analysis.residual_velocity
    analysis.load_state_dict(torch.load(NATIVE/'shared_latest.pt',map_location='cpu',weights_only=True),strict=True)
    analysis.eval().requires_grad_(False)
    checkpoint=torch.load(NATIVE/'coupling_latest.pt',map_location='cpu',weights_only=True)
    models={name:(GlobalConditionalSplineDecoder(45,3,8,layers=4,width=32,bins=8,attention_heads=4) if name=='reference' else
        DenseGlobalConditionalSplineDecoder(45,3,8,layers=4,width=32,bins=8,attention_heads=4,backend='compiled' if name=='dense_compiled' else 'dense_eager')) for name in ARMS}
    for model in models.values():model.load_state_dict(checkpoint,strict=True);model.eval()
    with np.load(NATIVE/'coupling/chunk_0000.npz',allow_pickle=False) as bank:source=np.array(bank['gaussian'],copy=True)
    if source.shape!=(64,3072) or source.dtype!=np.float32 or not np.isfinite(source).all():raise ValueError('expected first64 float32 full Gaussian rows')
    np.save(out/'common_gaussian_source.npy',source)
    setup=[{'phase':'construction_and_checkpoint_loading','seconds':time.perf_counter()-start}]
    source=torch.from_numpy(source).to(device)
    def sync():torch.cuda.synchronize(device)
    def select(arm,pipeline):
        begin=time.perf_counter()
        for model in models.values():model.cpu();model.zero_grad(set_to_none=True)
        analysis.to(device if pipeline else 'cpu');models[arm].to(device)
        sync()
        return time.perf_counter()-begin
    def memory():return {'baseline_allocated_bytes':torch.cuda.memory_allocated(device),'baseline_reserved_bytes':torch.cuda.memory_reserved(device)}
    def peak(record):
        record.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(device),peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
        record['incremental_peak_allocated_bytes']=record['peak_allocated_bytes']-record['baseline_allocated_bytes']
    def pipeline(arm,batch):
        with torch.no_grad():return utility.generate(analysis,models[arm],source[:batch],kind='coupling',coarse_nfe=32)
    # First uses include compiled specialization and A/coarse/conditioner work.
    equivalence={}
    for batch in (1,64):
        reference=None
        for arm in ARMS:
            state.update(phase='pipeline_first_use',arm=arm,batch=batch)
            movement=select(arm,True);record=dict(arm=arm,case=f'pipeline_{batch}',movement_seconds=movement,**memory())
            torch.cuda.reset_peak_memory_stats(device);sync();start=time.perf_counter();state['active_call_started']=start
            value=pipeline(arm,batch);sync()
            record['first_use_seconds']=time.perf_counter()-start;peak(record);setup.append(record);write(out/'setup.json',setup)
            actual=value.cpu().numpy();del value
            np.save(out/f'pipeline_{batch}_{arm}.npy',actual)
            if reference is None:reference=actual
            equivalence[f'pipeline_{batch}_{arm}']=error_stats(actual,reference,atol=1e-4)
            write(out/'equivalence.json',equivalence)
            if not equivalence[f'pipeline_{batch}_{arm}']['passed']:raise FloatingPointError('full pipeline image equivalence gate failed')
    # Generate the backward inputs with the original checkpoint; no fit inputs.
    state['phase']='generated_backward_inputs';select('reference',True)
    with torch.no_grad():
        coarse=analysis.coarse_prior.sample_from_gaussian(source[:32,:192],steps=16)
        noise=source[:32,192:].reshape(32,45,8,8)
        residual,rld=models['reference'].decode(noise,coarse)
        encoded,ild=models['reference'].encode(residual,coarse)
        code=analysis._join_code(coarse,residual);logits,ald=analysis.decode_analysis(code);back,aild=analysis.encode_analysis(logits)
    np.savez(out/'generated_backward_inputs.npz',coarse=coarse.cpu().numpy(),residual=residual.cpu().numpy(),noise=noise.cpu().numpy(),code=code.cpu().numpy(),recovered_code=back.cpu().numpy())
    checks={'conditional_noise_roundtrip':float((encoded-noise).abs().max()),'conditional_logdet_cancellation':float((rld+ild).abs().max()),
        'analysis_code_roundtrip':float((back-code).abs().max()),'analysis_logdet_cancellation':float((ald+aild).abs().max()),
        'coarse_Heun_inverse_verified':False,'full_source_inverse_claim':False}
    write(out/'roundtrip_checks.json',checks)
    if not all(np.isfinite(checks[k]) and checks[k]<=limit for k,limit in (('conditional_noise_roundtrip',1e-3),('analysis_code_roundtrip',1e-3),('conditional_logdet_cancellation',1e-2),('analysis_logdet_cancellation',1e-2))):raise FloatingPointError('original conditional/source-code numerical gate failed')
    residual=residual.detach();coarse=coarse.detach()
    write(out/'persistent_input_memory.json',{'common_source_bytes':source.numel()*source.element_size(),
        'generated_residual_bytes':residual.numel()*residual.element_size(),'generated_coarse_bytes':coarse.numel()*coarse.element_size(),
        'scope':'these fixed GPU buffers remain in steady-case baselines; peaks are not pure generator storage'})
    del noise,rld,encoded,ild,code,logits,ald,back,aild
    def backward(arm):
        model=models[arm];model.zero_grad(set_to_none=True)
        z,ld=model.encode(residual,coarse)
        nll=.5*(z.square()+np.log(2*np.pi)).flatten(1).sum(1)-ld
        loss=nll.mean()/2880;loss.backward()
        return z,ld,nll,loss
    baseline=None
    for arm in ARMS:
        state.update(phase='backward_first_use_and_equivalence',arm=arm)
        movement=select(arm,False);record=dict(arm=arm,case='encode_nll_backward_32',movement_seconds=movement,**memory())
        torch.cuda.reset_peak_memory_stats(device);sync();start=time.perf_counter();state['active_call_started']=start
        z,ld,nll,loss=backward(arm);sync()
        record['first_use_seconds']=time.perf_counter()-start;peak(record);setup.append(record);write(out/'setup.json',setup)
        arrays={'encoded':z.detach().cpu().numpy(),'logdet':ld.detach().cpu().numpy(),'nll':nll.detach().cpu().numpy(),'loss':loss.detach().cpu().numpy()}
        for name,parameter in models[arm].named_parameters():
            if parameter.grad is None:raise AssertionError('missing parameter gradient: '+name)
            arrays['gradient::'+name]=parameter.grad.detach().cpu().numpy()
        np.savez(out/f'backward_{arm}.npz',**arrays)
        if baseline is None:baseline=arrays
        comparisons={key:error_stats(value,baseline[key],atol=1e-4 if key.startswith('gradient::') or key=='encoded' else 1e-2,
            rtol=1e-3 if key.startswith('gradient::') else 0) for key,value in arrays.items()}
        equivalence['backward_'+arm]=comparisons;write(out/'equivalence.json',equivalence)
        if not all(c['passed'] for c in comparisons.values()):raise FloatingPointError('backward value or parameter-gradient equivalence gate failed')
        del z,ld,nll,loss
    # No timing results are collected until all equivalence gates have passed.
    rows=[];orders=[];rng=random.Random(ORDER_SEED)
    for case in ('pipeline_1','pipeline_64','encode_nll_backward_32'):
        is_pipeline=case.startswith('pipeline');batch=int(case.split('_')[1]) if is_pipeline else 32
        call=lambda arm:pipeline(arm,batch) if is_pipeline else backward(arm)
        for arm in ARMS:
            state.update(phase='warmup',case=case,arm=arm);select(arm,is_pipeline);sync();start=time.perf_counter()
            for _ in range(10):call(arm)
            sync();setup.append({'case':case,'arm':arm,'warmup_count':10,'warmup_seconds':time.perf_counter()-start});write(out/'setup.json',setup)
        for repeat in range(30):
            order=list(ARMS);rng.shuffle(order);orders.append({'case':case,'repeat':repeat,'order':order})
            for arm in order:
                state.update(phase='steady_timing',case=case,arm=arm,repeat=repeat)
                movement=select(arm,is_pipeline);record=dict(case=case,arm=arm,repeat=repeat,movement_seconds=movement,**memory())
                torch.cuda.reset_peak_memory_stats(device);sync();begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                start=time.perf_counter();begin.record();result=call(arm);end.record();sync()
                record['wall_seconds']=time.perf_counter()-start;record['cuda_event_milliseconds']=begin.elapsed_time(end);peak(record)
                # Finite verification is separately charged, outside the sampled
                # backward duration; production training-loop checks are absent.
                check_start=time.perf_counter()
                values=(result,) if is_pipeline else result
                if any(not bool(torch.isfinite(v).all()) for v in values):raise FloatingPointError('nonfinite timed result')
                if not is_pipeline and any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in models[arm].parameters()):raise FloatingPointError('nonfinite timed parameter gradient')
                sync();record['post_call_finite_check_seconds']=time.perf_counter()-check_start
                rows.append(record);write(out/'timings.json',rows);write(out/'order.json',orders)
                del result,values
    summary={}
    for case in ('pipeline_1','pipeline_64','encode_nll_backward_32'):
        for arm in ARMS:
            selected=[r for r in rows if r['case']==case and r['arm']==arm]
            summary[case+'_'+arm]={'repetitions':len(selected),'median_wall_seconds':float(np.median([r['wall_seconds'] for r in selected])),
                'median_cuda_event_ms':float(np.median([r['cuda_event_milliseconds'] for r in selected])),
                'maximum_incremental_peak_bytes':max(r['incremental_peak_allocated_bytes'] for r in selected)}
    write(out/'summary.json',{'cases':summary,'setup':setup,'compiled_first_use_seconds':sum(r.get('first_use_seconds',0) for r in setup if r.get('arm')=='dense_compiled'),
        'scope':'frozen checkpoint implementation equivalence and timing; no quality comparison, fitting, optimizer or production training-loop benchmark',
        'amortization':'No break-even claim; first uses, all shapes, warmups, memory and inter-arm movement reported separately.'})


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--expected-commit',required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False);state={'phase':'startup'};started=time.perf_counter();status='failed'
    def stop(signum,frame):raise TimeoutError('benchmark deadline/termination warning')
    for sig in (signal.SIGTERM,signal.SIGUSR1,signal.SIGALRM):signal.signal(sig,stop)
    signal.alarm(29*60)
    try:run(args,state);status='completed_checkpoint_backend_benchmark'
    except BaseException as error:
        write(args.output/'FAILED.json',{'state':state,'error':repr(error),'traceback':traceback.format_exc(),
            'elapsed_seconds':time.perf_counter()-started,'active_or_last_first_use_elapsed_seconds':time.perf_counter()-state['active_call_started'] if 'active_call_started' in state else None});raise
    finally:
        signal.alarm(0)
        files={str(p.relative_to(args.output)):digest(p) for p in args.output.rglob('*') if p.is_file() and p.name!='COMPLETE.json' and not any(part in ('inductor-cache','triton-cache','tmp') for part in p.relative_to(args.output).parts)}
        write(args.output/'COMPLETE.json',{'status':status,'elapsed_seconds':time.perf_counter()-started,'payload_sha256':files,'excluded_reproducible_compiler_cache_directories':['inductor-cache','triton-cache','tmp']})


if __name__=='__main__':main()
