#!/usr/bin/env python3
"""Postprocess certified receipts and archived PSC medians; no integration or fits.

Decimal interval endpoints are consumed as printed. Upper endpoints admit a
quality target; lower endpoints only exclude it. Ratios are descriptive costs,
not mathematical latency bounds. The secondary all-call schedule is not used.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from decimal import Decimal, localcontext
from pathlib import Path

NS = (4, 8, 16, 32, 64)
PSC_SHA256 = 'e783b9d666f84a1cba1ed5e615410c8718649c2f8f286aaad115a5e18d9306ea'

def read(path: Path):
    return json.loads(path.read_text(), parse_float=Decimal)

def status(interval, target):
    lower, upper = map(Decimal, interval)
    if lower > upper:
        raise ValueError('reversed interval')
    return 'eligible' if upper <= target else ('excluded' if lower > target else 'unresolved')

def s(value):
    return str(value)

def summarize(results: Path, summary: Path) -> dict:
    if hashlib.sha256(summary.read_bytes()).hexdigest() != PSC_SHA256:
        raise ValueError('archived PSC summary hash mismatch')
    med = read(summary)['median_seconds']
    if set(med) != {f'{seed}_{batch}' for seed in (2026090901,2026090902,2026090903) for batch in (1,64)}:
        raise ValueError('wrong source/batch set')
    certified = {}
    for n in NS:
        raw = read(results / f'n{n}_first_attempt.json')
        risk = read(results / f'n{n}_risk.json')
        if raw['N'] != n or risk['N'] != n:
            raise ValueError('receipt grid mismatch')
        a,b = raw['conditional_KL']
        l,u = risk['unconditional_KL']
        if not (0 <= a <= b and 0 <= l <= u):
            raise ValueError('invalid KL enclosure')
        if max(b-a,u-l) > Decimal('2e-8'):
            raise ValueError('requested width not attained')
        certified[str(n)] = {
            'conditional_KL': [s(a),s(b)],
            'conditional_width': s(b-a),
            'expected_training_KL': [s(l),s(u)],
            'expected_training_width': s(u-l),
            'learning_correction_upper': s(risk['learning_correction'][1]),
            'wrong_head_bound_upper': s(risk['root_correct_wrong_head_bound'][1]),
            'certificate_source_sha256': hashlib.sha256((results/f'n{n}_first_attempt.json').read_bytes()).hexdigest(),
            'risk_source_sha256': hashlib.sha256((results/f'n{n}_risk.json').read_bytes()).hexdigest(),
        }
    exact_bound = read(results / 'n64_risk.json')['exact_expected_upper'][1]
    output = {'units':'forward KL nats per complete D3072 array',
              'dimensions':{'full':3072,'root':192,'residual':2880},
              'grid':list(NS),'interval_endpoints':'outward decimal strings',
              'exact_expected_risk_upper':s(exact_bound), 'certified':certified,
              'primary_psc_summary_sha256':PSC_SHA256,
              'cost_scope':'ratios of paired archived primary medians, not latency guarantees or independent fits',
              'targets':[], 'paired_ratios':{},
              'no_eligible_grid_comparator_is_not_infinite_speedup':True}
    with localcontext() as ctx:
        ctx.prec=40
        for n in NS:
            output['paired_ratios'][str(n)]={key:s(v[f'endpoint_fm_{n}']/v['central_exact']) for key,v in med.items()}
        for exponent in range(1,7):
            target=Decimal(10)**(-exponent)
            entry={'target':s(target), 'exact_conditional_eligible':True,
                   'exact_expected_eligible':bool(exact_bound<=target),
                   'conditional_status':{}, 'expected_status':{},
                   'fastest_primary_by_source_batch':{}, 'paired_ratio_ranges':{}}
            for n in NS:
                c=certified[str(n)]
                entry['conditional_status'][str(n)]=status(c['conditional_KL'],target)
                entry['expected_status'][str(n)]=status(c['expected_training_KL'],target)
            for key,values in med.items():
                eligible=[n for n in NS if entry['expected_status'][str(n)]=='eligible']
                selected=min(eligible,key=lambda n:values[f'endpoint_fm_{n}']) if eligible else None
                entry['fastest_primary_by_source_batch'][key]=selected
            chosen=set(entry['fastest_primary_by_source_batch'].values())
            entry['common_fastest_eligible_N']=next(iter(chosen)) if len(chosen)==1 else None
            if None not in chosen:
                for batch in (1,64):
                    ratios=[]
                    for key,values in med.items():
                        if key.endswith(f'_{batch}'):
                            n=entry['fastest_primary_by_source_batch'][key]
                            ratios.append(values[f'endpoint_fm_{n}']/values['central_exact'])
                    entry['paired_ratio_ranges'][str(batch)]=[s(min(ratios)),s(max(ratios))]
            output['targets'].append(entry)
    return output

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,default=Path(__file__).resolve().parent/'results')
    p.add_argument('--summary',type=Path,default=Path(__file__).resolve().parent/'sources/psc_summary.json')
    p.add_argument('--output',type=Path)
    a=p.parse_args(); report=summarize(a.results,a.summary)
    text=json.dumps(report,indent=2)+'\n'
    if a.output:
        with a.output.open('x') as f:f.write(text)
    else: print(text,end='')
if __name__=='__main__':main()
