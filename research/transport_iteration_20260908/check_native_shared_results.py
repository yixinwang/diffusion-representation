#!/usr/bin/env python3
"""Independent saved-bank verifier; never trains, generates, or downloads data.

Hash-only mode intentionally cannot complete the canonical-data numerical audit.
A faithfully preserved failed study remains failed even when every hash matches.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np

ARMS = ['analysis_only','coupling']+[f'residual_fm_nfe_{n}' for n in (4,8,16,32,64)]


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def require(condition,message):
    if not condition:raise AssertionError(message)


def safe(root,relative):
    path=root/relative
    require(not path.is_symlink() and root.resolve() in path.resolve().parents,'unsafe payload path')
    return path


def compare(a,b,label):
    a,b=np.asarray(a),np.asarray(b)
    require(a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all(),label+' shape/finite')
    require(np.allclose(a,b,rtol=1e-11,atol=1e-12),label+' numerical mismatch')


def measurements(images):
    # Explicit channels/spatial slices, independent of production descriptors.
    luminance=(images[:,0]+images[:,1]+images[:,2])/3.
    left=np.mean(luminance[:,:16,:16],axis=(1,2))
    right=np.mean(luminance[:,16:,16:],axis=(1,2))
    h=np.mean((luminance[:,:,1:]-luminance[:,:,:-1])**2,axis=(1,2))
    v=np.mean((luminance[:,1:,:]-luminance[:,:-1,:])**2,axis=(1,2))
    return np.stack((left,right,left*right,h,v,h*v),axis=1)


def image_scores(real,generated):
    require(generated.shape==(2*len(real),3,32,32),'expected one pair per repair image')
    first=generated[::2].reshape(len(real),-1)
    second=generated[1::2].reshape(len(real),-1)
    observed=real.reshape(len(real),-1)
    d1=np.sqrt(np.sum((first-observed)**2,axis=1)/3072.)
    d2=np.sqrt(np.sum((second-observed)**2,axis=1)/3072.)
    d12=np.sqrt(np.sum((first-second)**2,axis=1)/3072.)
    return (d1+d2-d12)/2.


def verify(root,repo,*,data_root=None,hash_only=False):
    root,repo=Path(root).resolve(),Path(repo).resolve()
    status=read(root/'status.json')
    result={'study_status':status.get('status'),'full_numerical_audit':False,
            'confirmatory_claim':False,'errors':[],'checks':{}}
    require(status.get('status') in ('completed_development_only','failed'),
            'study not terminal; partial/active state cannot be treated as complete')
    expected=status.get('payload_sha256')
    require(isinstance(expected,dict),'terminal payload manifest missing')
    actual={str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() and p.name!='status.json'}
    require(actual==set(expected),'payload inventory differs from terminal manifest')
    for name,digest in expected.items():require(sha(safe(root,name))==digest,'payload hash '+name)
    result['checks']['payload_files']=len(expected)
    if (root/'source_identity.json').exists():
        identity=read(root/'source_identity.json');commit=identity['commit']
        require(len(commit)==40 and all(c in '0123456789abcdef' for c in commit),'invalid Git commit')
        source_hashes=identity['sha256']
        require(len(source_hashes)==22,'expected frozen 22-file source closure')
        snapshot=set()
        for relative,digest in source_hashes.items():
            blob=subprocess.check_output(['git','show',f'{commit}:{relative}'],cwd=repo)
            require(hashlib.sha256(blob).hexdigest()==digest,'Git blob mismatch '+relative)
            matches=[p for p in (root/'sources').iterdir()
                     if p.name.partition('_')[0].isdigit()
                     and p.name.partition('_')[2]==Path(relative).name]
            require(len(matches)==1,'source snapshot basename ambiguous/missing '+relative)
            filename=matches[0].name
            require(sha(safe(root,'sources/'+filename))==digest,'source snapshot mismatch '+relative)
            snapshot.add(filename)
        require({p.name for p in (root/'sources').iterdir()}==snapshot,'source snapshot inventory')
        result['checks']['source_commit']=commit
        # Canonical loader may run only from code matching the frozen blobs.
        if not hash_only:
            for name in ('data_integrity','observed_flow_data','__init__','core'):
                rel=f'qalt/src/qalt/{name}.py'
                require(sha(repo/rel)==source_hashes[rel],'local loader source differs from frozen run')
    else:
        require(status['status']=='failed','completed run lacks source identity')
        result['checks']['source_identity']='unavailable: failure preceded source freeze'
    if status['status']=='failed':
        failure=read(root/'failure.json')
        require(failure['status']=='failed','failure status mismatch')
        require(failure['state']==status['state'],'failure phase mismatch')
        result.update(audit_status='preserved_failed_run',failure_phase=failure['state'],
                      limitation='Hashes do not turn a failed or incomplete study into a successful experiment.')
        return result
    evaluations=read(root/'evaluations.json')
    require(set(evaluations)==set(ARMS),'completed run must have exactly seven evaluation arms')
    metadata=read(root/'metadata.json')
    require(metadata['source_dimension']==3072 and metadata['repair_reused'] is True and metadata['confirmatory_claim'] is False,'study scope metadata')
    summary=read(root/'summary.json')
    require(summary['metadata']==metadata and summary['evaluations']==evaluations,'summary metadata/evaluation mismatch')
    require(summary['training']==read(root/'training_report.json') and summary['paired_energy']==read(root/'paired_energy.json'),'summary training/paired mismatch')
    gate=read(root/'numerical_gate.json')
    require(gate['finite'] is True and gate['exact_copy_equal'] is True,'numerical gate failed')
    with np.load(root/'exact_copy_arrays.npz',allow_pickle=False) as copied:
        require(np.array_equal(copied['candidate'],copied['copied']),'actual exact copy mismatch')
    with np.load(root/'numerical_gate_arrays.npz',allow_pickle=False) as arrays:
        for key in arrays.files:require(np.isfinite(arrays[key]).all(),'nonfinite saved gate array')
        residual_source=arrays['source'][:,192:].reshape(8,45,8,8)
        for left,right,key in [(arrays['source'],arrays['analysis_recovered'],'analysis_source_error'),
                              (residual_source,arrays['residual_recovered'],'residual_source_error'),
                              (arrays['candidate_code'],arrays['candidate_code_recovered'],'candidate_code_error')]:
            error=float(np.max(np.abs(left-right)))
            compare(error,gate[key],'numerical gate '+key)
            require(error<=gate['source_limit'],'source inversion tolerance failed')
    # Determinant arrays were not saved: verify reported bound only, not a new Jacobian computation.
    for key in ('analysis_logdet_error','residual_logdet_error','candidate_analysis_logdet_error'):
        require(np.isfinite(gate[key]) and gate[key]<=gate['logdet_limit'],'reported determinant gate')
    result['checks']['jacobian_limitation']='reported determinant bounds checked; raw determinant arrays absent'
    ids=np.load(root/'record_ids.npz',allow_pickle=False)
    require(ids['fit'].shape==(4000,) and ids['repair'].shape==(1000,),'frozen ID shapes')
    repair=None
    if not hash_only:
        require(data_root is not None,'canonical --data-root required unless explicit --hash-only')
        sys.path.insert(0,str(repo/'qalt/src'))
        from qalt.observed_flow_data import load_observed_flow_data
        data=load_observed_flow_data(Path(data_root))
        require(data.ledger['canonical_dataset_verified'] is True and data.ledger['allow_noncanonical_fixture'] is False,'canonical-only audit')
        require(np.array_equal(ids['fit'],data.fit_ids) and np.array_equal(ids['repair'],data.repair_ids),'canonical record identities')
        require(data.repair.shape==(1000,3,32,32),'canonical repair shape')
        require(read(root/'data_ledger.json')==data.ledger,'canonical ledger mismatch')
        repair=data.repair
    all_scores={};reference_sources=None
    for arm in ARMS:
        folder=root/arm;record=evaluations[arm]
        require(record['generated_count']==2000 and record['pairs_per_image']==1 and record['dimension']==3072,'registered evaluation size '+arm)
        require(record['record_ids_supplied'] is True,'missing repair identity '+arm)
        local=read(folder/'hashes.json')
        require(set(local)=={p.name for p in folder.iterdir() if p.name!='hashes.json'},'bank inventory '+arm)
        for name,digest in local.items():require(sha(safe(folder,name))==digest,'bank hash '+arm+'/'+name)
        digest=hashlib.sha256();sources=[];features=[];scores=[];positions=[];boundaries=0
        for path in sorted(folder.glob('chunk_*.npz')):
            with np.load(path,allow_pickle=False) as bank:
                z,images,pos=bank['gaussian'],bank['generated'],bank['repair_positions']
                require(z.dtype==np.float32 and z.shape==(2*len(pos),3072),'source dtype/shape '+arm)
                require(images.dtype==np.float64 and images.shape==(2*len(pos),3,32,32),'generated dtype/shape '+arm)
                require(np.isfinite(z).all() and np.isfinite(images).all() and ((images>=0)&(images<=1)).all(),'bank finite/range '+arm)
                positions.extend(pos.tolist());sources.append(z.copy());digest.update(z.tobytes())
                features.append(measurements(images));boundaries+=int(np.count_nonzero((images==0)|(images==1)))
                if repair is not None:scores.extend(image_scores(repair[pos],images).tolist())
        require(positions==list(range(1000)),'repair positions missing/reordered '+arm)
        full_source=np.concatenate(sources)
        if reference_sources is None:reference_sources=full_source
        else:require(np.array_equal(full_source,reference_sources),'actual Gaussian banks differ '+arm)
        require(digest.hexdigest()==record['source_stream_sha256'],'source digest '+arm)
        require(hashlib.sha256(np.asarray(ids['repair'],dtype='<i8').tobytes()).hexdigest()==record['repair_ids_sha256'],'repair digest '+arm)
        compare(np.concatenate(features).mean(0),record['generated_descriptor_mean'],'generated descriptors '+arm)
        require(boundaries==record['rounded_boundary_values'],'boundary count '+arm)
        if repair is not None:
            require(hashlib.sha256(np.ascontiguousarray(repair,dtype=np.float64).tobytes()).hexdigest()==record['repair_values_sha256'],'repair values digest '+arm)
            scores=np.asarray(scores);all_scores[arm]=scores
            compare(scores,np.load(folder/'per_image_energy.npy',allow_pickle=False),'saved scores '+arm)
            compare(scores,record['per_image_energy'],'JSON scores '+arm)
            compare(scores.mean(),record['energy_mean'],'score mean '+arm)
            compare(scores.std(ddof=1)/np.sqrt(1000),record['descriptive_energy_se'],'score SE '+arm)
            compare(measurements(repair).mean(0),record['repair_descriptor_mean'],'repair descriptors '+arm)
    result['checks']['identical_source_banks']=7
    result['checks']['generated_metrics_and_boundaries']=7
    if repair is not None:
        paired=read(root/'paired_energy.json')
        require(set(paired)==set(ARMS)-{'coupling'},'paired arm inventory')
        for arm,record in paired.items():
            delta=all_scores['coupling']-all_scores[arm]
            compare(delta.mean(),record['first_minus_second'],'paired mean '+arm)
            compare(delta.std(ddof=1)/np.sqrt(1000),record['descriptive_paired_se'],'paired SE '+arm)
            require(record['images']==1000,'paired image count')
        result.update(audit_status='verified_completed_development_only',full_numerical_audit=True)
    else:
        result.update(audit_status='incomplete_hash_only',
            limitation='Canonical repair not opened: energy values, repair descriptors and paired differences not independently recomputed.')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root',required=True);parser.add_argument('--repo',required=True)
    parser.add_argument('--output-json',required=True)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--data-root');mode.add_argument('--hash-only',action='store_true')
    args=parser.parse_args();output=Path(args.output_json)
    require(not output.exists(),'audit output already exists')
    require(Path(args.input_root).resolve() not in output.resolve().parents,'audit output must be outside frozen payload')
    try:result=verify(args.input_root,args.repo,data_root=args.data_root,hash_only=args.hash_only)
    except Exception as error:
        result={'audit_status':'rejected','full_numerical_audit':False,'error':repr(error)}
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as handle:json.dump(result,handle,indent=2,allow_nan=False)
    print(json.dumps(result,indent=2))
    return int(result['audit_status']=='rejected')


if __name__=='__main__':raise SystemExit(main())
