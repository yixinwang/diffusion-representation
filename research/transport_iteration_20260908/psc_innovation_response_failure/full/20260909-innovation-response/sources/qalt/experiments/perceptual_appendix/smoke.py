#!/usr/bin/env python3
"""One fixed fabricated-input Inception forward; no datasets or bank API."""
import argparse
import hashlib
import json
import platform
from pathlib import Path
import subprocess
import time
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def write(path, value):
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--expected-commit',required=True)
    p.add_argument('--inception-source',type=Path,required=True)
    p.add_argument('--weights',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    start=time.perf_counter()
    try:
        actual=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        if actual!=args.expected_commit:raise ValueError('source commit mismatch')
        sources={}
        for name in ('smoke.py','evaluator.py','run_smoke.slurm'):
            path=HERE/name;rel=path.relative_to(ROOT).as_posix()
            if path.read_bytes()!=subprocess.check_output(['git','show',f'{actual}:{rel}'],cwd=ROOT):
                raise ValueError('modified source '+rel)
            sources[rel]=sha(path)
        from evaluator import make_extractor, extract, SOURCE_SHA256, WEIGHT_SHA256
        import evaluator
        if Path(evaluator.__file__).resolve()!=HERE/'evaluator.py':
            raise ValueError('unexpected evaluator import')
        import numpy as np
        import torch
        import torchvision
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.manual_seed(20260909)
        model=make_extractor(args.inception_source,args.weights,'cpu')
        if model.training or any(parameter.requires_grad for parameter in model.parameters()):
            raise AssertionError('extractor not frozen/eval')
        images=np.linspace(0.0,1.0,2*3*32*32,dtype=np.float64).reshape(2,3,32,32)
        # Exactly one model forward: extract batches32 and there are two images.
        features=extract(model,images,'cpu')
        if features.shape!=(2,2048) or features.dtype!=np.float32 or not np.isfinite(features).all():
            raise AssertionError('invalid fabricated features')
        record={'status':'fabricated_extractor_smoke_passed','source_commit':actual,
            'source_sha256':sources,'inception_source_sha256':SOURCE_SHA256,
            'weights_sha256':WEIGHT_SHA256,'python':platform.python_version(),
            'numpy':np.__version__,'torch':torch.__version__,'torchvision':torchvision.__version__,
            'device':'cpu','input':'numpy.linspace(0,1,6144,float64).reshape(2,3,32,32)',
            'input_sha256':hashlib.sha256(images.tobytes()).hexdigest(),
            'feature_shape':list(features.shape),'feature_dtype':str(features.dtype),
            'feature_sha256':hashlib.sha256(features.tobytes()).hexdigest(),
            'feature_min':float(features.min()),'feature_max':float(features.max()),
            'feature_all_finite':True,'forward_count':1,'elapsed_seconds':time.perf_counter()-start,
            'real_data_read':False,'generated_banks_read':False,'training_performed':False,
            'interpretation':'dependency/numerical compatibility only; no generation quality evidence'}
        write(args.output/'smoke.json',record)
        write(args.output/'COMPLETE.json',{'status':record['status'],
            'payload_sha256':{'smoke.json':sha(args.output/'smoke.json')}})
    except Exception as exc:
        write(args.output/'FAILED.json',{'status':'failed','exception':repr(exc),
              'traceback':traceback.format_exc(),'elapsed_seconds':time.perf_counter()-start})
        raise


if __name__=='__main__':main()
