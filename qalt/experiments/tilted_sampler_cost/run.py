#!/usr/bin/env python3
"""Prospective CPU sampler benchmark; explicit numerical-only or timing mode.

No fitting, datasets, GPUs, or scheduler API. Root must freeze this runner and
protocol before deployment. Exact source/parameter bytes are pinned below.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import random
import resource
import signal
import subprocess
import sys
import time
import tracemalloc

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
REFERENCE='research/transport_iteration_20260908/pro8_artifacts/positive_class_reference.py'
PARAMETERS='research/transport_iteration_20260908/pro8_artifacts/positive_class_results.json'
SOURCE_FILES=(REFERENCE,PARAMETERS,
    'qalt/experiments/tilted_sampler_cost/run.py',
    'qalt/experiments/tilted_sampler_cost/candidate.py',
    'qalt/experiments/tilted_sampler_cost/PROTOCOL.md',
    'qalt/experiments/tilted_sampler_cost/run.slurm',
    'qalt/tests/test_tilted_sampler_cost.py',
    'research/transport_iteration_20260908/central_tilt_correctness.py',
    'research/transport_iteration_20260908/central_tilt_correctness.json',
    'research/transport_iteration_20260908/central_tilt_review.md')

PINS={'reference':'38cb7d9b79111b99e3af961aa28cdf2ca6f5a2b427245434e2cea407c7b6c73b',
      'parameters':'92a0203e85f838a06f8f265359a12d1fb21839e2fa55ae99942446232396cdd3',
      'candidate':'e660a4d814eb25ad16405ccd72ca7c47b7d2bd1de749ca394524a36a88e42bbe'}
SEEDS=(2026090901,2026090902,2026090903)
NFES=(4,8,16,32,64)
ARMS=('original_exact','central_exact')+tuple(f'{kind}_{n}' for kind in ('original_fm','cached_fm','endpoint_fm') for n in NFES)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write(path,value):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temporary.replace(path)


def load_source(name,path,pin):
    payload=path.read_bytes()
    if hashlib.sha256(payload).hexdigest()!=pin:raise ValueError(name+' source differs from pin')
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module
    exec(compile(payload,str(path),'exec'),module.__dict__)
    return module


def source_guard(commit,out):
    actual=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    if actual!=commit:raise ValueError('expected full commit must equal HEAD')
    payloads={}
    for relative in SOURCE_FILES:
        payload=(ROOT/relative).read_bytes()
        frozen=subprocess.check_output(['git','show',f'{commit}:{relative}'],cwd=ROOT)
        if payload!=frozen:raise ValueError('dirty source closure: '+relative)
        payloads[relative]=payload
    folder=out/'sources';folder.mkdir()
    for index,(name,payload) in enumerate(payloads.items()):
        (folder/f'{index:02d}_{Path(name).name}').write_bytes(payload)
    identity={'source_commit':commit,'sha256':{name:hashlib.sha256(payload).hexdigest() for name,payload in payloads.items()}}
    write(out/'source_identity.json',identity)
    return identity


def run(args,state):
    lifecycle_start=time.perf_counter() if args.mode=='timing' else None
    identity=source_guard(args.expected_commit,args.output)
    reference_path=ROOT/REFERENCE;candidate_path=HERE/'candidate.py';parameters_path=ROOT/PARAMETERS
    for variable in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        if os.environ.get(variable)!='1':raise ValueError('set '+variable+'=1 before invocation')
    import numpy as np
    import scipy
    from scipy.special import ndtr
    reference=load_source('pinned_tilt_reference',reference_path,PINS['reference'])
    candidate=load_source('pinned_tilt_candidate',candidate_path,PINS['candidate'])
    if sha(parameters_path)!=PINS['parameters']:raise ValueError('saved fitted parameter artifact mismatch')
    for module,path in ((reference,reference_path),(candidate,candidate_path)):
        if Path(module.__file__).resolve()!=path.resolve() or sys.modules[module.__name__] is not module:
            raise ValueError('imported source origin mismatch')
    if (reference.D,reference.COARSE,reference.DETAIL,reference.FEATURES,reference.NFES)!=(3072,192,2880,16,NFES):
        raise ValueError('frozen reference shape or solver grid changed')
    archived=json.loads(parameters_path.read_text())['training']
    gamma,index=archived['fitted_gamma'],archived['fitted_index']
    if gamma not in (-.5,.5) or not isinstance(index,int) or not 0<=index<16:
        raise ValueError('invalid archived fitted catalog parameters')
    setup_start=time.perf_counter() if args.mode=='timing' else None
    dictionary=reference.orthogonal_dictionary()
    if not np.allclose(dictionary@dictionary.T,np.eye(16),rtol=0,atol=1e-12):raise ValueError('public dictionary not orthonormal')
    np.save(args.output/'public_dictionary.npy',dictionary)
    inputs={}
    for seed in SEEDS:
        rng=np.random.default_rng(seed)
        for batch in (1,64):
            source=rng.normal(size=(batch,3072))
            inputs[(seed,batch)]=source
            np.save(args.output/f'source_{seed}_{batch}.npy',source)
    metadata={'scope':'fixed fitted mathematical-class CPU sampler implementation comparison; not native quality',
        'mode':args.mode,'seeds':SEEDS,'batches':[1,64],'primary_arms':ARMS,
        'warmups':5,'repetitions':30,'gamma':gamma,'summary_index':index,
        'archived_training_performed_here':False,'source_sha256':PINS,
        'source_identity':identity,'runner_sha256':sha(__file__),'protocol_sha256':sha(HERE/'PROTOCOL.md'),
        'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,
        'host':platform.node(),'slurm_job':os.environ.get('SLURM_JOB_ID'),
        'threads':{v:os.environ[v] for v in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')},
        'primary_cached_time_cost':'constructed inside every call',
        'endpoint_fm_call_accounting':{str(n):{'equivalent_mathematical_stages':n,'nontrivial_field_kernel_calls':n-2,'analytic_endpoint_stages':2,'final_predictor_allocated':False} for n in NFES},
        'cached_fm_root':'same central/tail exact root as central_exact; both root and time arithmetic optimized',
        'dictionary_and_source_setup_seconds':None if setup_start is None else time.perf_counter()-setup_start,
        'guard_import_parameters_setup_seconds':None if setup_start is None else setup_start-lifecycle_start}
    write(args.output/'metadata.json',metadata)
    def optimized(source,nfe=None,cache=None,endpoint=False):
        # Root, public dense summary, inverse Haar, and sigmoid are inside every call.
        coarse=candidate.quantile_transport(source[:,:192],gamma)
        tilt=.5+.1*(2*ndtr(source[:,:192]@dictionary[index])-1)
        if nfe is None:residual=candidate.quantile_transport(source[:,192:],tilt[:,None])
        elif endpoint:residual=candidate.endpoint_heun(source[:,192:],tilt[:,None],nfe)[0]
        elif cache is None:residual=candidate.heun(source[:,192:],tilt[:,None],nfe)[0]
        else:
            residual=source[:,192:].copy();h=2/nfe;calls=0
            for step in range(nfe//2):
                first=candidate.cached_field(residual,tilt[:,None],cache[step]);calls+=1
                second=candidate.cached_field(residual+h*first,tilt[:,None],cache[step+1]);calls+=1
                residual+=h/2*(first+second)
            if calls!=nfe:raise AssertionError('changed nominal NFE')
        return reference.outer_decode(coarse,residual)

    def sample(arm,source,cache=None):
        if arm=='original_exact':value=reference.generate(source,gamma,index,dictionary)
        elif arm=='central_exact':value=optimized(source)
        elif arm.startswith('original_fm_'):value=reference.generate(source,gamma,index,dictionary,'fm',int(arm.rsplit('_',1)[1]))
        else:value=optimized(source,int(arm.rsplit('_',1)[1]),cache,endpoint=arm.startswith('endpoint_fm_'))
        if value.shape!=(len(source),3,32,32) or not np.isfinite(value).all() or np.any((value<0)|(value>1)):
            raise FloatingPointError('invalid entire-sampler output')
        return value

    state['phase']='numerical_gates';checks={};authoritative={}
    # All six source banks and every arm pass before any warmup/timing is allowed.
    for (seed,batch),source in inputs.items():
        outputs={}
        for arm in ARMS:
            state.update(phase='numerical_gate_sampling',seed=seed,batch=batch,arm=arm)
            value=sample(arm,source)
            np.save(args.output/f'output_{seed}_{batch}_{arm}.npy',value)
            outputs[arm]=value
            authoritative[(seed,batch,arm)]=value
        comparisons=[('central_exact','original_exact')]+[(f'{kind}_{n}',f'original_fm_{n}') for kind in ('cached_fm','endpoint_fm') for n in NFES]
        for left,right in comparisons:
            error=float(np.max(np.abs(outputs[left]-outputs[right])))
            if error>1e-12:raise AssertionError('sampler output numerical gate '+left)
            checks[f'{seed}_{batch}_{left}']={'max_reference_output_error':error}
        for arm in ('original_exact','central_exact'):
            c,r=reference.outer_encode(outputs[arm]);u=candidate.inverse_transport(c,gamma)
            tilt=.5+.1*(2*ndtr(u@dictionary[index])-1)
            recovered=np.concatenate((u,candidate.inverse_transport(r,tilt[:,None])),axis=1)
            error=float(np.max(np.abs(recovered-source)))
            if error>2e-10:raise AssertionError('whole-pipeline exact roundtrip gate')
            checks[f'{seed}_{batch}_{arm}_roundtrip']={'max_source_error':error}
        # Amortized schedule variant is also checked before any primary timing.
        for n in NFES:
            cache=tuple(candidate.time_constants(i*2/n) for i in range(n//2+1))
            cached=sample(f'cached_fm_{n}',source,cache)
            if not np.array_equal(cached,outputs[f'cached_fm_{n}']):raise AssertionError('schedule cache changed output')
        write(args.output/'checks.json',checks)
    write(args.output/'GATES_PASSED.json',{'passed':True,'checks':len(checks),'timing_started':False})
    if args.mode=='numerical-only':return
    # Releasing gate outputs keeps only source banks and small dictionary resident.
    del authoritative,outputs,c,r,u,tilt,recovered,source,value,cached,cache
    primary=[];orders=[];amortized=[];memory=[];cache_setup=[]
    for (seed,batch),source in inputs.items():
        state.update(phase='primary_warmup',seed=seed,batch=batch)
        for arm in ARMS:
            for _ in range(5):sample(arm,source)
        order_rng=random.Random(seed+batch+401)
        for repetition in range(30):
            order=list(ARMS);order_rng.shuffle(order)
            orders.append({'phase':'primary','seed':seed,'batch':batch,'rep':repetition,'arms':order})
            state['phase']='primary_timing'
            for arm in order:
                start=time.perf_counter_ns();out=sample(arm,source);elapsed=time.perf_counter_ns()-start
                del out
                primary.append({'seed':seed,'batch':batch,'arm':arm,'rep':repetition,'nanoseconds':elapsed})
            write(args.output/'primary_timings.json',primary);write(args.output/'orders.json',orders)
        # Explicit secondary schedule-amortization diagnostic. Primary remains per-call.
        start=time.perf_counter_ns()
        caches={n:tuple(candidate.time_constants(i*2/n) for i in range(n//2+1)) for n in NFES}
        cache_setup.append({'seed':seed,'batch':batch,'all_schedules_setup_nanoseconds':time.perf_counter_ns()-start})
        for n in NFES:
            for _ in range(5):sample(f'cached_fm_{n}',source,caches[n])
        for repetition in range(30):
            order=list(NFES);order_rng.shuffle(order)
            for n in order:
                state['phase']='secondary_amortized_timing'
                start=time.perf_counter_ns();out=sample(f'cached_fm_{n}',source,caches[n]);elapsed=time.perf_counter_ns()-start
                del out
                amortized.append({'seed':seed,'batch':batch,'nfe':n,'rep':repetition,'nanoseconds':elapsed})
            write(args.output/'secondary_amortized_timings.json',amortized)
        # Instrument memory in separate untimed sampler calls; traced allocations
        # are not equivalent to allocator RSS and do not establish intrinsic space.
        for arm in ARMS:
            tracemalloc.start();baseline=tracemalloc.get_traced_memory()[0]
            out=sample(arm,source);current,peak=tracemalloc.get_traced_memory();tracemalloc.stop()
            memory.append({'seed':seed,'batch':batch,'arm':arm,'traced_peak_minus_baseline_bytes':peak-baseline,'output_bytes':out.nbytes})
            del out
        write(args.output/'memory_diagnostics.json',memory);write(args.output/'schedule_setup.json',cache_setup)
    summary={}
    for seed in SEEDS:
        for batch in (1,64):
            summary[f'{seed}_{batch}']={arm:float(np.median([r['nanoseconds'] for r in primary if r['seed']==seed and r['batch']==batch and r['arm']==arm]))/1e9 for arm in ARMS}
    write(args.output/'summary.json',{'median_seconds':summary,'process_peak_rss':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'rss_unit':'KiB on Linux; bytes on macOS','primary_rows':len(primary),'secondary_rows':len(amortized),
        'limitation':'single-process implementation timings; no model quality or intrinsic complexity conclusion'})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--expected-commit',required=True)
    p.add_argument('--mode',choices=('numerical-only','timing'),required=True)
    args=p.parse_args();external_start=time.perf_counter() if args.mode=='timing' else None
    args.output.mkdir(parents=True,exist_ok=False);state={'phase':'guard'}
    def stop(signum,frame):raise TimeoutError('fixed 19-minute limit or termination')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGALRM,stop);signal.alarm(19*60)
    try:
        run(args,state);signal.alarm(0)
        write(args.output/'COMPLETE.json',{'status':'completed_'+args.mode,'timing_performed':args.mode=='timing',
            'external_seconds_before_final_hashing':None if external_start is None else time.perf_counter()-external_start,
            'payload_sha256':{str(q.relative_to(args.output)):sha(q) for q in sorted(args.output.rglob('*')) if q.is_file()}})
    except Exception as error:
        signal.alarm(0)
        write(args.output/'FAILED.json',{'status':'failed','state':state,'error':repr(error),
            'payload_sha256':{str(q.relative_to(args.output)):sha(q) for q in sorted(args.output.rglob('*')) if q.is_file()}})
        raise


if __name__=='__main__':main()
