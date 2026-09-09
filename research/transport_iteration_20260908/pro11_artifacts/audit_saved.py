#!/usr/bin/env python3
"""Recheck saved synthetic fits without fitting or accessing a repository.

The population KL is recomputed by a 2-D quadrature implementation that does
not call model.entropy_F, entropy_F_prime, pair_kl, or Fixture.evaluate.
"""
from __future__ import annotations
import hashlib, json, math
from pathlib import Path
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import ndtri
from model import FittedModel, feature

HERE=Path(__file__).resolve().parent

def digest(a):
    return hashlib.sha256(memoryview(np.ascontiguousarray(a)).cast('B')).hexdigest()

def entropy_integrals(t, e, nodes=96):
    x,w=leggauss(nodes);x=(x+1)/2;w=w/2
    h=np.outer(feature(x),feature(x)).ravel()
    ww=np.outer(w,w).ravel()
    tt=np.asarray(t).ravel();ee=np.asarray(e).ravel()
    ans=np.zeros_like(tt);b=np.zeros_like(tt);c=np.zeros_like(tt)
    for start in range(0,len(tt),64):
        stop=min(start+64,len(tt))
        th=tt[start:stop,None]*h;eh=ee[start:stop,None]*h
        ans[start:stop]=((1+th)*np.log1p(th))@ww
        le=np.log1p(eh)
        b[start:stop]=le@ww
        c[start:stop]=le@(ww*h)
    return ans,b,c

rows=[]
for seed in (1109101,1109102,1109103):
    record=json.loads((HERE/f'learning_seed_{seed}.json').read_text())
    for name,expected in record['environment']['source_sha256'].items():
        assert hashlib.sha256((HERE/name).read_bytes()).hexdigest()==expected,(seed,name)
    for world,rec in record['worlds'].items():
        path=HERE/f'fit_{world}_{seed}.npz'
        with np.load(path,allow_pickle=False) as a:
            model=FittedModel(a['root_prob'],list(a['pairings']),a['theta'],
                              rec['structural_pass'],720,2000,.35,.45,[])
            z=a['source_check'];y=a['generated_check']
            assert digest(z)==rec['fresh_roundtrip_source']['sha256']
            assert digest(y)==rec['generated']['sha256']
            assert np.array_equal(y,model.sample(z,32))
            uz=model.encode_uniform(y,32)
            zerror=float(np.max(np.abs(ndtri(uz)-z)));assert zerror<1e-9
            root=float(np.sum(a['evaluator_root_prob']*np.log(a['evaluator_root_prob']/a['root_prob'])))
            q,w=leggauss(20);K=model.bins
            cx=((np.arange(K)[:,None]+(q[None,:]+1)/2)/K).ravel()
            wt=np.tile(w/(2*K),K)
            true=a['evaluator_sign'][None,:]*(rec['offset']+rec['amplitude']*
                          np.sin(2*math.pi*cx[:,None]+a['evaluator_phase'][None,:]))
            est=model.context(cx);cond=0.;product=0.
            for g in range(4):
                edges=set(map(tuple,np.sort(a['evaluator_pairings'][g],axis=1)))
                hit=sum(tuple(v) in edges for v in np.sort(a['pairings'][g],axis=1))
                f,b,c=entropy_integrals(true[:,g],est[:,g])
                cond+=float(wt@(360*f-360*b-hit*true[:,g]*c))
                product+=float(wt@(360*f))
            value=root+cond
            errors={'joint_KL':abs(value-rec['evaluation']['joint_KL']),
                    'root_KL':abs(root-rec['evaluation']['root_KL']),
                    'conditional_KL':abs(cond-rec['evaluation']['conditional_KL']),
                    'product_floor':abs(product-rec['evaluation']['conditional_product_floor'])}
            assert max(errors.values())<2e-10,errors
            rows.append({'seed':seed,'world':world,'recomputed_joint_KL':value,
                         'discrepancies':errors,'maximum_Gaussian_source_roundtrip_error':zerror,
                         'generated_bank_regeneration_exact':True,
                         'checkpoint_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
out={'status':'passed','models':6,'method':'independent 2D copula quadrature plus 20-node-per-cell context quadrature; no refitting',
     'maximum_KL_discrepancy':max(max(x['discrepancies'].values()) for x in rows),
     'maximum_Gaussian_source_roundtrip_error':max(x['maximum_Gaussian_source_roundtrip_error'] for x in rows),
     'records':rows,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(HERE/'saved_audit.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
