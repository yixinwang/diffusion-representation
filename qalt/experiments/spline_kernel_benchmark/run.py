#!/usr/bin/env python3
"""Frozen synthetic checked-call CUDA microbenchmark; no data/model training."""
import argparse
import hashlib
import importlib.metadata
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

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
SEED=20260909
ORDER_SEED=20260910
ARMS=('reference','dense_eager','dense_compiled')


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
    return h.hexdigest()


def write(path,value):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temporary.replace(path)


def source_guard(revision):
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()!=revision:
        raise ValueError('frozen revision mismatch')
    sources={}
    for path in [HERE/'run.py',HERE/'PROTOCOL.md',HERE/'run.slurm',
                 *(ROOT/'qalt/src/qalt'/p for p in ('__init__.py','core.py','spline.py','dense_spline.py'))]:
        rel=path.relative_to(ROOT).as_posix()
        if path.read_bytes()!=subprocess.check_output(['git','show',revision+':'+rel],cwd=ROOT):
            raise ValueError('frozen source differs: '+rel)
        sources[rel]=digest(path)
    return sources


def run(args,state):
    source_hashes=source_guard(args.expected_commit)
    # Explicit isolated caches, before importing torch/compiler. No shared cache mutation.
    for key,leaf in [('TORCHINDUCTOR_CACHE_DIR','inductor-cache'),('TMPDIR','tmp'),
                     ('TRITON_CACHE_DIR','triton-cache')]:
        path=args.output/leaf;path.mkdir();os.environ[key]=str(path.resolve())
    os.environ['TORCHINDUCTOR_COMPILE_THREADS']='1'
    sys.path.insert(0,str(ROOT/'qalt/src'))
    import numpy as np
    import torch
    from qalt.spline import rational_quadratic_spline as reference
    from qalt.dense_spline import dense_rational_quadratic_spline as dense
    from qalt.dense_spline import dense_spline_kernel
    for name in ('qalt','qalt.core','qalt.spline','qalt.dense_spline'):
        expected=ROOT/'qalt/src'/('qalt/__init__.py' if name=='qalt' else name.replace('.','/')+'.py')
        if Path(sys.modules[name].__file__).resolve()!=expected.resolve():raise ValueError('import origin mismatch')
    if not torch.cuda.is_available():raise RuntimeError('CUDA is mandatory; CPU timing forbidden')
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    device=torch.device('cuda:0')
    metadata={'source_commit':args.expected_commit,'source_sha256':source_hashes,
              'python':platform.python_version(),'numpy':np.__version__,'torch':torch.__version__,
              'triton':importlib.metadata.version('triton'),'cuda_build':torch.version.cuda,
              'device':torch.cuda.get_device_name(device),'device_capability':torch.cuda.get_device_capability(device),
              'host':platform.node(),'slurm_job':os.environ.get('SLURM_JOB_ID'),
              'compiler':'torch.compile backend=inductor fullgraph=True dynamic=False',
              'compiler_environment':{k:os.environ.get(k) for k in ('CC','CXX','TORCHINDUCTOR_COMPILE_THREADS')},
              'seed':SEED,'order_seed':ORDER_SEED,'real_data_read':False,'training_performed':False,
              'task_cache_paths':{k:os.environ[k] for k in ('TORCHINDUCTOR_CACHE_DIR','TRITON_CACHE_DIR','TMPDIR')},
              'timed_shape':[64,2880],'bins':8,'timed_dtype':'float32'}
    for name,command in [('driver',['nvidia-smi','--query-gpu=name,driver_version','--format=csv,noheader']),
                         ('host_c_compiler',['cc','--version'])]:
        try:metadata[name]=subprocess.check_output(command,text=True,stderr=subprocess.STDOUT,timeout=15).splitlines()[:2]
        except Exception as e:metadata[name]={'unavailable':repr(e)}
    write(args.output/'metadata.json',metadata)
    def tensors(shape,dtype,seed):
        g=torch.Generator(device='cpu').manual_seed(seed)
        values=[torch.randn(shape,generator=g,dtype=dtype)]
        values.extend(torch.randn((*shape,k),generator=g,dtype=dtype)*.2 for k in (8,8,7))
        return [v.to(device) for v in values]
    full=tensors((64,2880),torch.float32,SEED)
    small32=tensors((4,17),torch.float32,SEED+1)
    small64=tensors((4,17),torch.float64,SEED+2)
    np.savez(args.output/'fabricated_inputs.npz',**{f'{label}_{i}':v.cpu().numpy()
            for label,vs in [('full32',full),('small32',small32),('small64',small64)] for i,v in enumerate(vs)})
    # Validate static metadata once before compiling the kernel; every numerical
    # invocation below still checks its returned status exactly once.
    compiled=torch.compile(dense_spline_kernel,backend='inductor',fullgraph=True,dynamic=False)
    functions={'reference':reference,'dense_eager':dense,'dense_compiled':compiled}
    setup=[]
    def call(arm,values,inverse):
        if arm=='reference':return functions[arm](*values,inverse=inverse)
        y,ld,valid=functions[arm](*values,inverse=inverse)
        if not bool(valid):
            np.savez(args.output/'failed_validity.npz',value=y.detach().cpu().numpy(),
                     logdet=ld.detach().cpu().numpy(),valid=valid.detach().cpu().numpy())
            raise FloatingPointError('dense validity false: '+arm+' inverse='+str(inverse))
        return y,ld
    checks={}
    def close(label,a,b,atol=1e-4,rtol=0.):
        error=(a-b).abs();limit=atol+rtol*b.abs()
        finite=bool(torch.isfinite(a).all() & torch.isfinite(b).all())
        passed=finite and bool((error<=limit).all())
        checks[label]={'max_absolute_error':float(error.max()) if finite else None,
                       'atol':atol,'rtol':rtol,'passed':passed}
        write(args.output/'numerical_checks.json',checks)
        if not passed:
            np.savez(args.output/'failed_comparison.npz',actual=a.detach().cpu().numpy(),reference=b.detach().cpu().numpy())
            raise AssertionError('pre-timing gate failed: '+label)
    state['phase']='pre_timing_gates'
    for label,values in [('full32',full),('small32',small32),('small64',small64)]:
        # No performance samples are accepted until all these gates pass.
        tol=1e-10 if label=='small64' else 1e-4
        outputs={}
        with torch.no_grad():
            for inverse in (False,True):
                ref=None
                for arm in ARMS:
                    # First-use wall includes possible compile setup AND execution;
                    # not presented as a pure compiler or warmed kernel duration.
                    torch.cuda.synchronize();start=time.perf_counter()
                    actual=call(arm,values,inverse);torch.cuda.synchronize()
                    setup.append({'case':label,'arm':arm,'inverse':inverse,
                                  'first_use_wall_seconds':time.perf_counter()-start})
                    write(args.output/'first_use_setup.json',setup)
                    if arm=='reference':ref=actual
                    for term,a,b in zip(('value','logdet'),actual,ref):close(f'{label}_{arm}_{inverse}_{term}',a,b,tol)
                    if not inverse:outputs[arm]=actual
            for arm,(y,ld) in outputs.items():
                back,ild=call(arm,[y,*values[1:]],True)
                close(f'{label}_{arm}_roundtrip',back,values[0],tol)
                close(f'{label}_{arm}_logdet_cancel',ld+ild,torch.zeros_like(ld),tol)
    for inverse in (False,True):
        gradients={}
        for arm in ARMS:
            values=[v.detach().clone().requires_grad_(True) for v in small32]
            torch.cuda.synchronize();start=time.perf_counter()
            y,ld=call(arm,values,inverse)
            gradients[arm]=torch.autograd.grad((y+.1*ld).sum(),values)
            torch.cuda.synchronize()
            setup.append({'case':'small32_gradient','arm':arm,'inverse':inverse,
                          'first_use_wall_seconds':time.perf_counter()-start})
            write(args.output/'first_use_setup.json',setup)
        for arm in ARMS:
            for i,(a,b) in enumerate(zip(gradients[arm],gradients['reference'])):
                close(f'gradient_{inverse}_{arm}_{i}',a,b,1e-4,1e-3)
    del gradients,outputs,actual,ref,values,y,ld,back,ild
    torch.cuda.synchronize()
    write(args.output/'GATES_PASSED.json',{'all_numerical_and_gradient_checks_passed':True,
          'check_count':len(checks),'timing_has_started':False})
    state['phase']='warmup'
    variants=[(arm,inverse) for inverse in (False,True) for arm in ARMS]
    with torch.no_grad():
        for _ in range(10):
            for arm,inverse in variants:
                result=call(arm,full,inverse);del result
        torch.cuda.synchronize()
        order_rng=random.Random(ORDER_SEED);rows=[];orders=[]
        state['phase']='timing'
        for rep in range(30):
            order=variants.copy();order_rng.shuffle(order);orders.append(order)
            for arm,inverse in order:
                torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
                before=torch.cuda.memory_allocated()
                begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                start=time.perf_counter();begin.record()
                result=call(arm,full,inverse)
                end.record();torch.cuda.synchronize();wall=time.perf_counter()-start
                rows.append({'repetition':rep,'arm':arm,'inverse':inverse,
                    'wall_seconds':wall,'cuda_event_milliseconds':begin.elapsed_time(end),
                    'allocated_before_bytes':before,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                    'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
                    'peak_incremental_allocated_bytes':torch.cuda.max_memory_allocated()-before})
                del result
            write(args.output/'timings.json',rows)
        write(args.output/'order.json',orders)
    summary={}
    for arm,inverse in variants:
        selected=[r for r in rows if r['arm']==arm and r['inverse']==inverse]
        summary[f'{arm}_{"inverse" if inverse else "forward"}']={
            'repetitions':len(selected),
            'median_wall_seconds':float(np.median([r['wall_seconds'] for r in selected])),
            'median_cuda_event_milliseconds':float(np.median([r['cuda_event_milliseconds'] for r in selected])),
            'max_peak_allocated_bytes':max(r['peak_allocated_bytes'] for r in selected),
            'max_peak_incremental_allocated_bytes':max(r['peak_incremental_allocated_bytes'] for r in selected)}
    write(args.output/'summary.json',{'variants':summary,
          'compile_first_use_wall_seconds_sum':sum(r['first_use_wall_seconds'] for r in setup if r['arm']=='dense_compiled'),
          'interpretation':'checked-call synthetic microbenchmark; not full-model speed or training evidence',
          'break_even':'not promised; first-use includes compilation and execution, workload/call count and amortization unknown'})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--expected-commit',required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    state={'phase':'source_guard'};started=time.perf_counter()
    def stop(signum,frame):
        raise TimeoutError('benchmark termination/deadline signal '+str(signum))
    signal.signal(signal.SIGTERM,stop)
    signal.signal(signal.SIGALRM,stop)
    # Preserve before the scheduler twenty-minute cap; no new timing accepted
    # after this fixed nineteen-minute whole-process deadline.
    signal.alarm(19*60)
    try:
        run(args,state)
        signal.alarm(0)
        write(args.output/'COMPLETE.json',{'status':'completed_synthetic_kernel_benchmark',
             'elapsed_seconds':time.perf_counter()-started,
             'payload_sha256':{p.name:digest(p) for p in args.output.iterdir() if p.is_file()}})
    except Exception as e:
        signal.alarm(0)
        write(args.output/'FAILED.json',{'status':'failed','phase':state['phase'],'exception':repr(e),
             'traceback':traceback.format_exc(),'elapsed_seconds':time.perf_counter()-started,
             'payload_sha256':{p.name:digest(p) for p in args.output.iterdir() if p.is_file()}})
        raise


if __name__=='__main__':main()
