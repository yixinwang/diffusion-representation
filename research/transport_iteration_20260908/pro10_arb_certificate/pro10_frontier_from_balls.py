"""Inspect conservative intervals from preserved Arb reports; no integration or timing."""
from pathlib import Path
import sys,json,hashlib
sys.path.insert(0,str(Path(__file__).resolve().parent/'pro10-arb-deps'))
from flint import arb,ctx
ctx.prec=128
root=Path(__file__).resolve().parent
pg=(-arb(32)*192/72).exp()
rho=1-arb(9)/16*(1-(-arb(2880)*(arb(1)/10)**2*(arb(1)/2)**2/(24*(arb(8)/5))).exp())
pj=15*rho**256
failure=17280*pj+17408*pg
exact=96*pj+224*pg
rows={}
for n in (4,8,16,32,64):
    p=root/f'pro10_arb_average{n}.json';r=json.loads(p.read_text())
    assert r['status']=='finite_enclosure' and r['scope']=='uniform e average'
    assert r['calls']==n and r['cutoff']==10 and r['version']=='0.8.0'
    assert r['source_sha256']==hashlib.sha256((root/'pro10_arb_compact.py').read_bytes()).hexdigest()
    imaginary=arb(r['imaginary_ball'])
    assert imaginary.is_finite() and imaginary.lower()<=0 and imaginary.upper()>=0
    compact=arb(r['compact_joint_ball']);upper=arb(r['full_joint_upper_ball']).upper()
    lower=compact.lower()
    assert lower>0 and upper>lower and upper-lower<arb(1)/50000000
    ulo=(lower*(1-pg-pj)).lower();uup=(upper+failure).upper()
    def classify(lo,hi,target):
        return 'eligible' if hi<=target else 'excluded' if lo>target else 'unresolved'
    rows[n]={'input_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'conditional_lower':str(lower),'conditional_upper':str(upper),
        'expected_lower':str(ulo),'expected_upper':str(uup),'full_interval_width_less_than_2e_minus8':True,
        'targets':{f'1e-{k}':{'conditional':classify(lower,upper,arb(10)**(-k)),
            'expected':classify(ulo,uup,arb(10)**(-k))} for k in range(1,7)}}
report={'purpose':'classification of independently reviewed trusted-computation enclosures for the restricted ideal-law example',
    'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'failure_allowance_ball':str(failure),'exact_expected_upper_ball':str(exact.upper()),'heun':rows,
    'exact_eligible_all_targets':bool(exact<arb(1)/1000000)}
print(json.dumps(report,indent=2))
