"""Frozen shared-analysis CUDA development study; reused repair is descriptive."""
from __future__ import annotations
import argparse
import hashlib
import json
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
import run_shared as utility
from qalt.learned_latent_flow_matching import LearnedLatentFlowMatching
from qalt.observed_flow_data import load_observed_flow_data
# Eager import makes the conditional decoder part of the pre-data source check.
import qalt.global_conditional_spline

ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = Path('/ocean/datasets/community/cifar/cifar-10/cifar-10-batches-py')
MODULES = ('__init__', 'core', 'spline', 'multiscale_flow', 'flow_matching',
           'learned_latent_flow_matching', 'global_conditional_spline',
           'data_integrity', 'observed_flow_data')
SOURCE_FILES = tuple(f'qalt/src/qalt/{name}.py' for name in MODULES) + (
    'qalt/tests/test_spline.py', 'qalt/tests/test_multiscale_flow.py',
    'qalt/tests/test_flow_matching.py', 'qalt/tests/test_learned_latent_flow_matching.py',
    'qalt/tests/test_global_conditional_spline.py', 'qalt/tests/test_observed_flow_data.py',
    'qalt/tests/test_observed_shared_pilot.py',
    'qalt/data/observed_manifest_v1.json',
    'qalt/experiments/observed_flow_pilot/PROTOCOL.md',
    'qalt/experiments/observed_flow_pilot/SHARED_PROTOCOL.md',
    'qalt/experiments/observed_flow_pilot/run_shared.py',
    'qalt/experiments/observed_flow_pilot/run_shared_study.py',
    'qalt/experiments/observed_flow_pilot/run_shared.slurm',
)
SOURCE_LIMIT = 1e-3
LOGDET_LIMIT = 1e-2


def source_guard(requested):
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    if requested != commit:
        raise ValueError('requested full source commit must equal HEAD exactly')
    hashes = {}
    for relative in SOURCE_FILES:
        actual = (ROOT / relative).read_bytes()
        committed = subprocess.check_output(['git', 'show', f'{commit}:{relative}'], cwd=ROOT)
        if actual != committed:
            raise ValueError(f'source differs from frozen commit: {relative}')
        hashes[relative] = hashlib.sha256(actual).hexdigest()
    for name in MODULES:
        imported = sys.modules.get('qalt' if name == '__init__' else f'qalt.{name}')
        if imported is None or Path(imported.__file__).resolve() != (ROOT/f'qalt/src/qalt/{name}.py').resolve():
            raise ValueError(f'imported project source differs from checked checkout: {name}')
    if Path(utility.__file__).resolve() != (ROOT/'qalt/experiments/observed_flow_pilot/run_shared.py').resolve():
        raise ValueError('imported utilities differ from checked checkout')
    return commit, hashes


def file_hashes(out):
    return {str(p.relative_to(out)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(out.rglob('*')) if p.is_file() and p.name != 'status.json'}


def failure(out, state, models, error):
    record = {'status': 'failed', 'exception': repr(error), 'traceback': traceback.format_exc(),
              'state': state, 'checkpoints': {}, 'checkpoint_errors': {}}
    for name, model in models.items():
        try:
            path = out/f'failure_{name}.pt'
            with path.open('xb') as handle:
                torch.save(model.state_dict(), handle)
            record['checkpoints'][name] = path.name
        except BaseException as secondary:
            record['checkpoint_errors'][name] = repr(secondary)
    with (out/'failure.json').open('x') as handle:
        json.dump(record, handle, indent=2, allow_nan=False)
        handle.flush(); os.fsync(handle.fileno())
    utility.atomic_json(out/'status.json', {**record, 'payload_sha256': file_hashes(out)})


def timeout(signum, frame):
    raise TimeoutError('scheduler termination warning received; preserving current study state')


@torch.no_grad()
def numerical_gate(analysis, candidate, out, device):
    generator = torch.Generator(device='cpu').manual_seed(utility.SEED+200)
    z = torch.randn(8, analysis.dimension, generator=generator).to(device)
    np.save(out/'numerical_gate_source.npy',z.cpu().numpy())
    # Untruncated Gaussian sources: the gate does not avoid difficult tails.
    logits, ld = analysis.decode_analysis(z)
    recovered, ild = analysis.encode_analysis(logits)
    coarse = analysis.coarse_prior.sample_from_gaussian(z[:, :analysis.latent_dimension], steps=16)
    noise = z[:, analysis.latent_dimension:].reshape(8,45,8,8)
    residual, rld = candidate.decode(noise, coarse)
    back, rild = candidate.encode(residual, coarse)
    joint = analysis._join_code(coarse, residual)
    candidate_logits, ald = analysis.decode_analysis(joint)
    joint_back, aild = analysis.encode_analysis(candidate_logits)
    arrays = dict(source=z, analysis_logits=logits, analysis_recovered=recovered,
        generated_coarse=coarse, residual=residual, residual_recovered=back,
        candidate_logits=candidate_logits, candidate_code=joint, candidate_code_recovered=joint_back)
    np.savez(out/'numerical_gate_arrays.npz', **{k:v.cpu().numpy() for k,v in arrays.items()})
    checks = dict(source_limit=SOURCE_LIMIT, logdet_limit=LOGDET_LIMIT,
        analysis_source_error=float((recovered-z).abs().max()),
        analysis_logdet_error=float((ld+ild).abs().max()),
        residual_source_error=float((back-noise).abs().max()),
        residual_logdet_error=float((rld+rild).abs().max()),
        candidate_code_error=float((joint_back-joint).abs().max()),
        candidate_analysis_logdet_error=float((ald+aild).abs().max()))
    finite = all(bool(torch.isfinite(v).all()) for v in (*arrays.values(), ld, ild, rld, rild, ald, aild))
    checks['finite'] = finite
    # JSON remains valid even for a nonfinite numerical failure.
    utility.atomic_json(out/'numerical_gate.json', {k:(v if not isinstance(v,float) or np.isfinite(v) else str(v)) for k,v in checks.items()})
    if not finite or max(checks[k] for k in ('analysis_source_error','residual_source_error','candidate_code_error')) > SOURCE_LIMIT or max(checks[k] for k in ('analysis_logdet_error','residual_logdet_error','candidate_analysis_logdet_error')) > LOGDET_LIMIT:
        raise FloatingPointError('frozen numerical inversion gate failed')
    # Exact-copy control shares every trained module; it receives no free weights.
    sampler = lambda source: utility.generate(analysis, candidate, source, kind='coupling')
    class ExactCopy:
        def __call__(self, source):
            return sampler(source)
    expected, copied = sampler(z), ExactCopy()(z)
    np.savez(out/'exact_copy_arrays.npz', source=z.cpu().numpy(),
             candidate=expected.cpu().numpy(), copied=copied.cpu().numpy())
    checks['exact_copy_equal'] = bool(torch.equal(expected, copied))
    utility.atomic_json(out/'numerical_gate.json', checks)
    if not checks['exact_copy_equal']:
        raise AssertionError('same-weight stochastic latent copy must tie exactly')
    return checks


def study(args, out, state, models):
    started = time.perf_counter()
    state['phase'] = 'source_guard'
    commit, hashes = source_guard(args.source_commit)
    source_dir = out/'sources'; source_dir.mkdir()
    for i, relative in enumerate(SOURCE_FILES):
        (source_dir/f'{i:02d}_{Path(relative).name}').write_bytes((ROOT/relative).read_bytes())
    utility.atomic_json(out/'source_identity.json', {'commit':commit, 'sha256':hashes})
    state['phase'] = 'guarded_test_preflight'
    with (out/'preflight_tests.txt').open('x') as handle:
        subprocess.run([sys.executable, '-m', 'pytest', '-q',
            *[str(ROOT/p) for p in SOURCE_FILES if p.startswith('qalt/tests/')]],
            cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, check=True)
    if not torch.cuda.is_available():
        raise RuntimeError('registered shared study requires one CUDA GPU; CPU fallback forbidden')
    device = torch.device('cuda:0')
    torch.set_num_threads(4)
    state['phase'] = 'cuda_probe'
    (torch.ones(2, device=device)*2).sum().item(); utility.synchronize(device)
    metadata = dict(commit=commit, seed=utility.SEED, settings=utility.DEFAULTS,
        device=str(device), gpu=torch.cuda.get_device_name(device), torch=torch.__version__,
        python=platform.python_version(), numpy=np.__version__, slurm_job=os.environ.get('SLURM_JOB_ID'),
        host=platform.node(), repair_reused=True, confirmatory_claim=False,
        full_model_likelihood_reported=False, source_dimension=3072,
        operational_composition='unused container residual_velocity removed; only analysis and coarse methods used')
    utility.atomic_json(out/'metadata.json', metadata)
    state['phase'] = 'canonical_data'; begin=time.perf_counter()
    data = load_observed_flow_data(DATA_ROOT)
    if not data.ledger['canonical_dataset_verified'] or data.ledger['allow_noncanonical_fixture'] or data.fit.shape != (4000,3,32,32) or data.repair.shape != (1000,3,32,32):
        raise ValueError('strict canonical 4000 fitting / 1000 repair selection required')
    utility.atomic_json(out/'data_ledger.json', data.ledger)
    np.savez(out/'record_ids.npz', fit=data.fit_ids, repair=data.repair_ids)
    timings={'canonical_load_seconds':time.perf_counter()-begin}
    state['phase']='logit_preprocessing'; begin=time.perf_counter()
    logits, jacobian=utility.logit_inputs(data.fit)
    np.save(out/'fit_outer_logit_jacobian.npy', jacobian)
    timings['logit_and_jacobian_save_seconds']=time.perf_counter()-begin
    state['phase']='model_construction'; begin=time.perf_counter()
    torch.manual_seed(utility.SEED)
    analysis=LearnedLatentFlowMatching(channels=3,size=32,levels=2,pre_layers=2,
        coarse_layers=6,detail_layers=4,width=32,bins=8,attention_heads=4,unit_interval=False)
    # The study supplies its own shape-matched residual decoders below.
    # Remove the unused FM field so it occupies neither resident GPU memory nor
    # this shared model's parameter count. Analysis methods do not reference it.
    del analysis.residual_velocity
    models['shared']=analysis
    candidate, reference, match=utility.make_decoders()
    models.update(coupling=candidate,residual_fm=reference)
    timings['model_construction_seconds']=time.perf_counter()-begin
    analysis_count=sum(p.numel() for p in analysis._analysis_parameters())
    coarse_count=utility.parameter_count(analysis.coarse_prior)
    metadata['parameter_counts']={'analysis':analysis_count,'coarse':coarse_count,**match,
        'candidate_resident':analysis_count+coarse_count+match['candidate_parameters'],
        'fm_resident':analysis_count+coarse_count+match['fm_parameters'],
        'analysis_only':analysis_count}
    utility.atomic_json(out/'metadata.json',metadata)
    state['phase']='training';begin=time.perf_counter()
    reports=utility.fit_shared_and_decoders(analysis,candidate,reference,logits,data.fit_ids,out,device=device)
    timings['all_fitting_external_seconds']=time.perf_counter()-begin
    common_setup=sum(timings[k] for k in ('canonical_load_seconds','logit_and_jacobian_save_seconds','model_construction_seconds'))
    reports['conservative_standalone_setup_and_fitting_seconds']={
        name:duration+common_setup for name,duration in reports['standalone_fitting_external_seconds'].items()}
    reports['standalone_cost_interpretation']='includes full common loading/logit/construction and shape-only capacity search for both; conservative measured study composition, not optimized standalone execution'
    utility.atomic_json(out/'training_report.json',reports)
    if any(reports[name]['updates'] < 1 for name in ('analysis','coarse','coupling','residual_fm')):
        raise RuntimeError('each registered stage must complete at least one update')
    for model in models.values():
        model.eval()
        for parameter in model.parameters():parameter.requires_grad_(False);parameter.grad=None
    del logits
    state['all_fits_frozen']=True
    utility.atomic_json(out/'status.json',{'status':'all_fits_frozen','state':state,'timings':timings})
    state['phase']='numerical_gate';begin=time.perf_counter()
    analysis.to(device);candidate.to(device);reference.cpu()
    with torch.no_grad():checks=numerical_gate(analysis,candidate,out,device)
    candidate.cpu();torch.cuda.empty_cache()
    timings['numerical_gate_seconds']=time.perf_counter()-begin
    evaluations={};latencies={};generation_memory={}
    arms=[('analysis_only','analysis_only',None,32),('coupling','coupling',candidate,32)]+[
        (f'residual_fm_nfe_{nfe}','fm',reference,nfe) for nfe in utility.DEFAULTS['residual_nfes']]
    for name,kind,decoder,nfe in arms:
        state.update(phase='evaluation',arm=name)
        begin=time.perf_counter()
        candidate.cpu();reference.cpu()
        analysis.coarse_prior.to('cpu' if kind=='analysis_only' else device)
        if decoder is not None:decoder.to(device)
        torch.cuda.empty_cache();utility.synchronize(device)
        timings[f'{name}_movement_seconds']=time.perf_counter()-begin
        torch.cuda.reset_peak_memory_stats(device)
        sampler=lambda source:utility.generate(analysis,decoder,source,kind=kind,residual_nfe=nfe,coarse_nfe=32)
        evaluations[name]=utility.evaluate_pairs(sampler,data.repair,dimension=3072,device=device,
            seed=utility.SEED+300,batch_size=64,pairs_per_image=1,record_ids=data.repair_ids,artifact_directory=out/name)
        utility.atomic_json(out/'evaluations.json',evaluations)
        state['phase']='latency'
        latencies[name]=utility.latency(sampler,dimension=3072,device=device,batch_sizes=(1,64),repeats=3)
        generation_memory[name]={'peak_allocated_bytes':torch.cuda.max_memory_allocated(device),
                                'peak_reserved_bytes':torch.cuda.max_memory_reserved(device)}
        utility.atomic_json(out/'latencies.json',latencies)
        utility.atomic_json(out/'generation_memory.json',generation_memory)
    paired={name:utility.paired_energy(evaluations['coupling'],value) for name,value in evaluations.items() if name!='coupling'}
    utility.atomic_json(out/'paired_energy.json',paired)
    timings['study_external_seconds_before_final_hashing']=time.perf_counter()-started
    timings['process_peak_rss_platform_units']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    timings['process_peak_rss_unit']='KiB on Linux; bytes on macOS'
    timings['cost_accounting']='external fitting includes cache, device transfers and checkpoints; stage clocks and full study wall separately reported'
    timings['unpartitioned_wall_including_preflight_evaluation_latency_and_io_seconds']=timings['study_external_seconds_before_final_hashing']-sum(v for k,v in timings.items() if k in ('canonical_load_seconds','logit_and_jacobian_save_seconds','model_construction_seconds','all_fitting_external_seconds','numerical_gate_seconds'))
    utility.atomic_json(out/'timings.json',timings)
    utility.atomic_json(out/'summary.json',dict(metadata=metadata,training=reports,numerical=checks,
        evaluations=evaluations,paired_energy=paired,timings=timings,
        exact_copy_cost='inherits candidate analysis, coarse and decoder fitting and sampling costs',
        analysis_only_cost={'fitting_seconds':reports['analysis']['elapsed_seconds'],
            'interpretation':'analysis stage only; not budget-matched full-flow comparator'},
        limitations='reused repair development only; no realistic dominance or state-of-the-art claim'))
    utility.atomic_json(out/'status.json',{'status':'completed_development_only','all_fits_frozen':True,
        'payload_sha256':file_hashes(out),'wall_seconds_before_status_write':time.perf_counter()-started})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-commit',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    state={'phase':'startup'};models={}
    signal.signal(signal.SIGTERM,timeout)
    signal.signal(signal.SIGUSR1,timeout)
    try:study(args,out,state,models)
    except BaseException as error:
        try:failure(out,state,models,error)
        except BaseException as secondary:print(f'failure preservation also failed: {secondary!r}',file=sys.stderr)
        raise


if __name__=='__main__':main()
