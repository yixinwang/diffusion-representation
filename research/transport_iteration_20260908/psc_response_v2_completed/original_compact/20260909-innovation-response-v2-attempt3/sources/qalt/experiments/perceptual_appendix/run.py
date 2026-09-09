#!/usr/bin/env python3
"""Evaluation-only development appendix. No test-loading or model-fitting API."""
import argparse
import hashlib
import json
import platform
from pathlib import Path
import subprocess
import sys
import traceback
import numpy as np

HERE = Path(__file__).resolve().parent
QALT = HERE.parents[1]
sys.path.insert(0,str(QALT/'src'))
from evaluator import digest, validate_artifacts, make_extractor, extract, SOURCE_SHA256, WEIGHT_SHA256
from metrics import evaluate
from qalt.observed_flow_data import load_observed_flow_data

ARMS = ('analysis_only','coupling',*(f'residual_fm_nfe_{n}' for n in (4,8,16,32,64)))
PROTECTED_SOURCES = [HERE/'run.py',HERE/'metrics.py',HERE/'evaluator.py',HERE/'PROTOCOL.md',
                     QALT/'src/qalt/__init__.py',QALT/'src/qalt/core.py',QALT/'src/qalt/observed_flow_data.py',QALT/'src/qalt/data_integrity.py',
                     QALT/'data/observed_manifest_v1.json']


def write_json(path, obj):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
    temp.replace(path)


def source_guard(commit):
    root = QALT.parent
    actual = subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    if actual != commit:
        raise ValueError('source commit differs from frozen expected commit')
    result = {}
    for path in PROTECTED_SOURCES:
        rel = path.relative_to(root).as_posix()
        blob = subprocess.check_output(['git','show',f'{commit}:{rel}'],cwd=root)
        if blob != path.read_bytes():
            raise ValueError('source differs from frozen Git blob: '+rel)
        result[rel] = digest(path)
    for name in ('qalt','qalt.core','qalt.observed_flow_data','qalt.data_integrity'):
        expected = QALT/'src'/('qalt/__init__.py' if name == 'qalt' else name.replace('.','/')+'.py')
        if Path(sys.modules[name].__file__).resolve() != expected.resolve():
            raise ValueError('unexpected imported project module: '+name)
    for name in ('evaluator','metrics'):
        if Path(sys.modules[name].__file__).resolve() != (HERE/(name+'.py')).resolve():
            raise ValueError('unexpected imported appendix module: '+name)
    return result


def checked_payload(root, relative, hashes):
    p = Path(relative)
    if p.is_absolute() or '..' in p.parts or not p.parts or root.resolve() not in (root/p).resolve().parents:
        raise ValueError('unsafe payload path')
    path = root/p
    if relative not in hashes or digest(path) != hashes[relative]:
        raise ValueError('missing/mismatched input payload: '+relative)
    return path


def load_bank(root, arm, hashes):
    expected = [f'{arm}/chunk_{first:04d}.npz' for first in range(0,1000,32)]
    actual = sorted(k for k in hashes if k.startswith(arm+'/chunk_') and k.endswith('.npz'))
    if actual != expected:
        raise ValueError('unexpected arm chunk inventory')
    images, sources = [], []
    for first, rel in zip(range(0,1000,32), expected):
        n = min(32,1000-first)
        with np.load(checked_payload(root,rel,hashes),allow_pickle=False) as z:
            if set(z.files) != {'generated','gaussian','repair_positions'}:
                raise ValueError('unexpected bank fields')
            if not np.array_equal(z['repair_positions'],np.arange(first,first+n)):
                raise ValueError('repair-position mismatch')
            x, source = z['generated'],z['gaussian']
            if x.dtype != np.float64 or x.shape != (2*n,3,32,32):
                raise ValueError('bank layout/dtype differs from frozen schema')
            if source.shape != (2*n,3072) or source.dtype != np.float32 or not np.isfinite(source).all():
                raise ValueError('invalid common Gaussian bank')
            images.append(x); sources.append(source)
    return np.concatenate(images),np.concatenate(sources)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('input','data-root','inception-source','weights','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--expected-commit',required=True)
    parser.add_argument('--expected-input-status-sha256',required=True)
    parser.add_argument('--device',choices=('cpu','cuda'),required=True)
    args = parser.parse_args()
    # Never modify an existing destination, even when failure occurs.
    args.output.mkdir(parents=True,exist_ok=False)
    try:
        sources = source_guard(args.expected_commit)
        validate_artifacts(args.inception_source,args.weights)
        status_path = args.input/'status.json'
        if digest(status_path) != args.expected_input_status_sha256:
            raise ValueError('input status not the predeclared frozen artifact')
        status = json.loads(status_path.read_text())
        if status.get('status') != 'completed_development_only' or status.get('all_fits_frozen') is not True:
            raise ValueError('input study is incomplete')
        hashes = status['payload_sha256']
        actual={p.relative_to(args.input).as_posix() for p in args.input.rglob('*')
                if p.is_file() and p.name != 'status.json'}
        if actual != set(hashes):
            raise ValueError('input payload inventory differs from frozen manifest')
        # Verify complete published inventory before reading array contents.
        for relative in hashes:
            checked_payload(args.input,relative,hashes)
        evaluations=json.loads(checked_payload(args.input,'evaluations.json',hashes).read_text())
        if set(evaluations) != set(ARMS):
            raise ValueError('input must contain exactly seven frozen evaluation arms')
        ledger_path = checked_payload(args.input,'data_ledger.json',hashes)
        ids_path = checked_payload(args.input,'record_ids.npz',hashes)
        data = load_observed_flow_data(args.data_root)
        if not data.ledger['canonical_dataset_verified'] or data.ledger['allow_noncanonical_fixture']:
            raise ValueError('strict canonical loader required')
        if data.repair.shape != (1000,3,32,32) or data.repair.dtype != np.float64:
            raise ValueError('wrong repair selection')
        if data.ledger != json.loads(ledger_path.read_text()):
            raise ValueError('canonical ledger differs from frozen study')
        with np.load(ids_path,allow_pickle=False) as ids:
            if not np.array_equal(ids['repair'],data.repair_ids) or not np.array_equal(ids['fit'],data.fit_ids):
                raise ValueError('canonical selected IDs differ')
        # Strict loader necessarily verifies all five training files, but fitting
        # and excluded-discovery pixels never enter feature extraction/statistics.
        repair, repair_ids, ledger = data.repair,data.repair_ids,data.ledger
        del data
        import torch
        import torchvision
        torch.set_num_threads(1)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        model = make_extractor(args.inception_source,args.weights,args.device)
        real = extract(model,repair,args.device)
        np.save(args.output/'repair_features.npy',real)
        np.save(args.output/'repair_ids.npy',repair_ids)
        results = {'real_real_first500_last500_smaller_pair':evaluate(real[:500],real[500:])}
        source_bank = None
        for arm in ARMS:
            images, gaussian = load_bank(args.input,arm,hashes)
            if source_bank is None:
                source_bank = gaussian
            elif not np.array_equal(source_bank,gaussian):
                raise ValueError('arms do not share identical Gaussian draws')
            features = extract(model,images,args.device)
            np.save(args.output/(arm+'_features.npy'),features)
            results[arm] = evaluate(real,features)
            results[arm]['input_boundary_values'] = int(((images==0)|(images==1)).sum())
            write_json(args.output/'metrics.json',results)
        write_json(args.output/'provenance.json',{
            'source_commit':args.expected_commit,'source_sha256':sources,
            'input_status_sha256':args.expected_input_status_sha256,'input_payload_sha256':hashes,
            'inception_source_sha256':SOURCE_SHA256,'weights_sha256':WEIGHT_SHA256,
            'python':platform.python_version(),'numpy':np.__version__,'torch':torch.__version__,
            'torchvision':torchvision.__version__,'device':args.device,'batch_size':32,
            'input_chart':'float64 NCHW unit cube -> float32; no clipping or PNG quantization',
            'data_ledger':ledger,'training_performed':False,'repair_reused_development':True,
            'full_bank_counts':{'repair':1000,'each_generated_arm':2000},
            'common_gaussian_array_sha256':hashlib.sha256(source_bank.tobytes()).hexdigest(),
            'repair_input_boundary_values':int(((repair==0)|(repair==1)).sum()),
            'interpretation':'descriptive development appendix; not confirmation or superiority evidence'})
        write_json(args.output/'COMPLETE.json',{'status':'completed_evaluation_only_development',
            'payload_sha256':{p.name:digest(p) for p in sorted(args.output.iterdir()) if p.is_file()}})
    except Exception as exc:
        write_json(args.output/'FAILED.json',{'status':'failed','exception':repr(exc),'traceback':traceback.format_exc(),
            'payload_sha256':{p.name:digest(p) for p in sorted(args.output.iterdir()) if p.is_file()}})
        raise


if __name__ == '__main__':
    main()
