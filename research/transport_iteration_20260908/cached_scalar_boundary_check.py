"""Numerical boundary qualification only; no model/data/training or timings."""
import json
from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'qalt/src'))
from qalt.cached_global_innovation import integrated_linear


def check():
    report={}
    for dtype,tolerance in ((torch.float32,2e-5),(torch.float64,5e-14)):
        g=torch.Generator().manual_seed(992)
        raw=30*torch.randn(4096,7,dtype=dtype,generator=g)
        cases={}
        for sign in (-1,1):
            endpoint=torch.tensor(4.*sign,dtype=dtype)
            source=torch.nextafter(endpoint,torch.tensor(0.,dtype=dtype)).expand(4096)
            result=integrated_linear(source,raw)
            inverse=integrated_linear(result.value,raw,inverse=True)
            error=float((inverse.value-source).abs().max())
            assert bool(result.valid & inverse.valid) and error<=tolerance
            cases[str(sign)]={'valid':True,'max_roundtrip':error,
                'at_or_beyond_tail_count':int((sign*result.value>=4).sum()),
                'strictly_beyond_tail_count':int((sign*result.value>4).sum()),
                'maximum_endpoint_overshoot':float(torch.clamp(sign*result.value-4,min=0).max())}
        report[str(dtype)]={'roundtrip_tolerance':tolerance,'cases':cases}
    return {'scope':'Finite-precision boundary rounding qualification, not an algebraic normalization defect or exact floating C1 guarantee. No clipping.', 'seed':992,'logit_scale':30,'rows':4096,'torch':torch.__version__,'checks':report}


if __name__=='__main__':print(json.dumps(check(),indent=2))
