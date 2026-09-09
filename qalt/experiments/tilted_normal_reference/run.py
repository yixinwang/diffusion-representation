"""Frozen full-dimensional synthetic finite-family validation; no native data."""
from __future__ import annotations
import argparse
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
import types
import numpy as np
import scipy

ROOT=Path(__file__).resolve().parents[3]
REFERENCE='research/transport_iteration_20260908/pro8_artifacts/positive_class_reference.py'
REFERENCE_SHA256='38cb7d9b79111b99e3af961aa28cdf2ca6f5a2b427245434e2cea407c7b6c73b'
TIMING_SEEDS=(2026090901,2026090902,2026090903)
RECOVERY_CELLS=tuple((gamma,index,2026091001+offset*16+index)
    for offset,gamma in enumerate((-.5,.5)) for index in range(16))
SOURCE_FILES=(REFERENCE,'qalt/experiments/tilted_normal_reference/run.py',
    'qalt/experiments/tilted_normal_reference/PROTOCOL.md',
    'qalt/experiments/tilted_normal_reference/run.slurm',
    'qalt/tests/test_tilted_normal_reference_protocol.py')
FULL_RT_LIMIT=1e-9
TAIL_RT_LIMIT=1e-10
DICTIONARY_LIMIT=1e-12


def array_hash(value):
    value=np.ascontiguousarray(value)
    sha=hashlib.sha256(str((value.shape,value.dtype.str)).encode());sha.update(value.view(np.uint8))
    return sha.hexdigest()


def clean(value):
    if isinstance(value,dict):return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [clean(v) for v in value]
    if isinstance(value,np.ndarray):return clean(value.tolist())
    if isinstance(value,np.generic):return clean(value.item())
    if isinstance(value,float) and not math.isfinite(value):return {'nonfinite':repr(value)}
    return value


def write_json(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    with temporary.open('w') as handle:
        json.dump(clean(value),handle,indent=2,allow_nan=False);handle.write('\n');handle.flush();os.fsync(handle.fileno())
    temporary.replace(path)


def verified_reference(commit,out):
    actual=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    if actual!=commit:raise ValueError('full source commit must equal HEAD exactly')
    payloads={}
    for relative in SOURCE_FILES:
        content=(ROOT/relative).read_bytes()
        frozen=subprocess.check_output(['git','show',f'{commit}:{relative}'],cwd=ROOT)
        if content!=frozen:raise ValueError(f'dirty source closure: {relative}')
        payloads[relative]=content
    if hashlib.sha256(payloads[REFERENCE]).hexdigest()!=REFERENCE_SHA256:
        raise ValueError('delivered reference bytes differ from reviewed artifact')
    folder=out/'sources';folder.mkdir()
    for i,(relative,content) in enumerate(payloads.items()):(folder/f'{i:02d}_{Path(relative).name}').write_bytes(content)
    write_json(out/'source_identity.json',dict(source_commit=commit,
        source_sha256={p:hashlib.sha256(b).hexdigest() for p,b in payloads.items()},reference_sha256=REFERENCE_SHA256))
    # Execute precisely the verified bytes. No source-path override or patched
    # reference functions are used by the study entry point.
    reference=types.ModuleType('verified_pro8_reference');reference.__file__=str(ROOT/REFERENCE)
    exec(compile(payloads[REFERENCE],reference.__file__,'exec'),reference.__dict__)
    if (reference.D,reference.COARSE,reference.DETAIL,reference.FEATURES,reference.NFES)!=(3072,192,2880,16,(4,8,16,32,64)):
        raise ValueError('reference dimensions or NFE grid differ')
    return reference


def make_observations(reference,seed,gamma,index,dictionary):
    """Simulator-only teacher parameters; returned fit inputs are observations."""
    rng=np.random.default_rng(seed)
    root_source=rng.normal(size=(32,reference.D))
    head_source=rng.normal(size=(256,reference.D))
    root=reference.generate(root_source,gamma,index,dictionary)
    head=reference.generate(head_source,gamma,index,dictionary)
    if not np.isfinite(root).all() or not np.isfinite(head).all():raise FloatingPointError('nonfinite synthetic fitting observations')
    hashes=dict(root_source=array_hash(root_source),head_source=array_hash(head_source),
        root_observations=array_hash(root),head_observations=array_hash(head),dictionary=array_hash(dictionary))
    return root,head,rng,hashes


def validate_reference_report(report):
    checks=report['numerical']
    for name,limit in (('full_source_roundtrip_max',FULL_RT_LIMIT),('head_tail_roundtrip_max_on_abs50',TAIL_RT_LIMIT),('orthogonal_dictionary_error',DICTIONARY_LIMIT)):
        if not math.isfinite(checks[name]) or checks[name]>limit:raise FloatingPointError(f'reference numerical gate failed: {name}')
    if not checks['exact_copy_bitwise']:raise AssertionError('exact conditional copy differs')
    for batch in ('1','64'):
        timing=report['timing'][batch]
        if set(timing['raw_seconds'])!={'exact','fm_4','fm_8','fm_16','fm_32','fm_64'}:raise AssertionError('missing timing arm')
        for values in timing['raw_seconds'].values():
            if len(values)!=9 or not np.isfinite(values).all() or min(values)<=0:raise AssertionError('invalid interleaved timing repetitions')


def timing_case(reference,seed,out):
    path=out/f'timing_seed_{seed}';path.mkdir()
    dictionary=reference.orthogonal_dictionary()
    # Reconstruct provenance outside original timing clocks; no wrapper changes
    # any of the delivered benchmark's fitting or generation functions.
    begin=time.perf_counter()
    root,head,rng,hashes=make_observations(reference,seed,.5,7,dictionary)
    z=rng.normal(size=(32,reference.D));hashes['roundtrip_source']=array_hash(z)
    orders={}
    labels=['exact']+[f'fm_{n}' for n in reference.NFES]
    for batch in (1,64):
        hashes[f'timing_source_batch_{batch}']=array_hash(rng.normal(size=(batch,reference.D)))
        orders[str(batch)]=[rng.permutation(labels).tolist() for _ in range(9)]
    write_json(path/'provenance.json',dict(seed=seed,truth_gamma=.5,truth_index=7,
        root_arrays=32,head_arrays=256,hashes=hashes,reconstructed_interleaving=orders,
        provenance_reconstruction_seconds=time.perf_counter()-begin))
    del root,head
    begin=time.perf_counter()
    report=reference.run(seed=seed,repeats=9)
    report['harness_reference_call_seconds']=time.perf_counter()-begin
    write_json(path/'raw_reference_report.json',report)
    validate_reference_report(report)
    write_json(path/'status.json',{'status':'completed','correct_selection':report['training']['correct_selection']})
    return dict(seed=seed,report_path=str(path.relative_to(out)/'raw_reference_report.json'),
        correct_selection=report['training']['correct_selection'],numerical=report['numerical'],timing=report['timing'])


def recovery_case(reference,cell,out):
    gamma,index,seed=cell;label='minus' if gamma<0 else 'plus'
    path=out/f'recovery_{label}_index_{index:02d}_seed_{seed}';path.mkdir()
    dictionary=reference.orthogonal_dictionary();root=head=None
    begin=time.perf_counter()
    try:
        root,head,_,hashes=make_observations(reference,seed,gamma,index,dictionary)
        simulation_seconds=time.perf_counter()-begin
        write_json(path/'inputs.json',dict(seed=seed,truth_gamma=gamma,truth_index=index,hashes=hashes,root_arrays=32,head_arrays=256))
        begin=time.perf_counter()
        fitted_gamma,fitted_index,scores=reference.fit(root,head,dictionary)
        fit_seconds=time.perf_counter()-begin
        if not np.isfinite(scores).all():raise FloatingPointError('nonfinite exact dictionary likelihood scores')
        report=dict(seed=seed,truth_gamma=gamma,truth_index=index,fitted_gamma=fitted_gamma,fitted_index=fitted_index,
            correct_sign=fitted_gamma==gamma,correct_index=fitted_index==index,
            correct_selection=fitted_gamma==gamma and fitted_index==index,all_dictionary_scores=scores.tolist(),
            score_gap=float(np.sort(scores)[-1]-np.sort(scores)[-2]),hashes=hashes,
            simulation_seconds=simulation_seconds,fit_seconds=fit_seconds,
            interpretation='one declared training seed for this sign/index; not repeated trials of each condition')
        write_json(path/'report.json',report)
        return report
    except BaseException:
        if root is not None and head is not None:np.savez_compressed(path/'failure_observations.npz',root=root,head=head,dictionary=dictionary)
        raise


def capture_reference_failure(error,out):
    """Preserve available partial original-run outputs only after an exception."""
    tb=error.__traceback__
    while tb:
        frame=tb.tb_frame
        if frame.f_code.co_filename==str(ROOT/REFERENCE) and frame.f_code.co_name=='run':
            names=('gamma','index','scores','tail_error','timings','times','batch','labels','data_seconds','fit_seconds')
            write_json(out/'partial_reference_report.json',{k:frame.f_locals[k] for k in names if k in frame.f_locals})
            arrays={k:frame.f_locals[k] for k in ('root','head','z','source','x','copy','out') if isinstance(frame.f_locals.get(k),np.ndarray)}
            if arrays:np.savez_compressed(out/'partial_reference_arrays.npz',**arrays)
        tb=tb.tb_next


def stop(signum,frame):raise TimeoutError('scheduler warning; preserving partial validation')


def execute(reference,out,state):
    timing=[];recovery=[]
    for seed in TIMING_SEEDS:
        state.update(phase='original_reference_timing',seed=seed)
        timing.append(timing_case(reference,seed,out))
        write_json(out/'progress.json',dict(completed_timing_seeds=len(timing),completed_recovery_cells=len(recovery)))
    for cell in RECOVERY_CELLS:
        state.update(phase='observation_only_recovery',cell=cell)
        recovery.append(recovery_case(reference,cell,out))
        write_json(out/'progress.json',dict(completed_timing_seeds=len(timing),completed_recovery_cells=len(recovery)))
    summary=dict(status='completed_synthetic_validation',timing_runs=timing,recovery_cells=recovery,
        timing_correct_selections=sum(r['correct_selection'] for r in timing),recovery_correct_selections=sum(r['correct_selection'] for r in recovery),
        scope='finite public dictionary and exact locked canonical velocity/Heun; NOT native image/video or unrestricted FM evidence',
        interval_validation_performed=False,real_data_accessed=False)
    write_json(out/'summary.json',summary)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-commit',required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter();state={'phase':'source_guard'};status='failed_partial'
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGUSR1,stop)
    try:
        reference=verified_reference(args.source_commit,args.output)
        for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
            if os.environ.get(name)!='1':raise ValueError(f'{name}=1 required')
        write_json(args.output/'manifest.json',dict(source_commit=args.source_commit,reference_sha256=REFERENCE_SHA256,
            timing_seeds=TIMING_SEEDS,recovery_cells=RECOVERY_CELLS,repeats=9,nfes=reference.NFES,
            root_arrays=32,head_arrays=256,dimensions=[reference.D,reference.COARSE,reference.DETAIL],
            numerical_limits=dict(full_source=FULL_RT_LIMIT,tail=TAIL_RT_LIMIT,dictionary=DICTIONARY_LIMIT),
            python=sys.version,numpy=np.__version__,scipy=scipy.__version__,platform=platform.platform(),host=platform.node(),
            slurm_job=os.environ.get('SLURM_JOB_ID'),real_data_accessed=False,network_accessed=False,
            interval_audit_included=False,command=[sys.executable,*sys.argv]))
        dictionary=reference.orthogonal_dictionary();np.save(args.output/'public_dictionary.npy',dictionary)
        execute(reference,args.output,state);status='completed_synthetic_validation'
    except BaseException as error:
        try:capture_reference_failure(error,args.output)
        except BaseException as secondary:state['partial_preservation_error']=repr(secondary)
        write_json(args.output/'failure.json',dict(status=status,state=state,error=repr(error),traceback=traceback.format_exc()))
        raise
    finally:
        hashes={str(p.relative_to(args.output)):hashlib.sha256(p.read_bytes()).hexdigest() for p in args.output.rglob('*') if p.is_file() and p.name!='COMPLETE.json'}
        write_json(args.output/'COMPLETE.json',dict(status=status,payload_sha256=hashes,
            elapsed_seconds=time.perf_counter()-started,peak_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            peak_rss_units='KiB on Linux, bytes on macOS',final_state=state))


if __name__=='__main__':main()
