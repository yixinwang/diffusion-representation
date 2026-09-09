"""Arb arithmetic from reviewed Pro10 receipts; no integration, fitting or timing."""
from pathlib import Path
import hashlib,json,sys
# Local isolated dependency fallback; normal python-flint installations also work.
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[2]/'pro10-arb-deps'))
from flint import arb,ctx
import flint
ctx.prec=128
PACKAGE=HERE/'pro10_arb_certificate'
manifest=json.loads((PACKAGE/'PAYLOAD_SHA256.json').read_text())
verified={}
for name,entry in manifest.items():
 p=PACKAGE/name;raw=p.read_bytes();sha=hashlib.sha256(raw).hexdigest()
 assert len(raw)==entry['bytes'] and sha==entry['sha256'],name
 verified[name]=sha
pg=(-arb(32)*192/72).exp()
rho=1-arb(9)/16*(1-(-arb(2880)*(arb(1)/10)**2*(arb(1)/2)**2/(24*(arb(8)/5))).exp())
pj=15*rho**256
rows={}
for n in (4,8,16,32,64):
 name=f'pro10_arb_average{n}.json';r=json.loads((PACKAGE/name).read_text())
 assert r['status']=='finite_enclosure' and r['scope']=='uniform e average' and r['calls']==n
 assert r['source_sha256']==verified['pro10_arb_compact.py']
 assert arb(r['imaginary_ball']).contains(0)
 a=arb(r['compact_joint_ball']).lower();b=arb(r['full_joint_upper_ball']).upper()
 W=96+arb(3)/2*b+arb(3)/2*(1440*b).sqrt()
 correction=pj*W+17408*pg
 lo=(a*(1-pg-pj)).lower();hi=(b+correction).upper();width=(hi-lo).upper()
 assert lo>0 and width<arb(1)/50000000
 rows[n]={'input_sha256':verified[name],'oracle_lower_ball':str(a),'oracle_upper_ball':str(b),
  'wrong_head_risk_bound_ball':str(W),'sharpened_failure_correction_ball':str(correction),
  'expected_lower_ball':str(lo),'expected_upper_ball':str(hi),'expected_interval_width_upper_ball':str(width),
  'expected_width_less_than_2e_minus8':bool(width<arb(1)/50000000),
  'expected_targets':{f'1e-{k}':('eligible' if hi<=arb(10)**(-k) else 'excluded' if lo>arb(10)**(-k) else 'unresolved') for k in range(1,7)}}
assert rows[64]['expected_targets']['1e-5']=='eligible'
report={'scope':'new analytical wrong-head correction applied only to reviewed preserved Arb enclosures; no Pro12 MPFR receipt verified',
 'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'python_flint_version':flint.__version__,'precision_bits':ctx.prec,
 'payload_manifest_sha256':hashlib.sha256((PACKAGE/'PAYLOAD_SHA256.json').read_bytes()).hexdigest(),'verified_payload_entries':len(verified),
 'root_failure_probability_ball':str(pg),'head_failure_probability_ball':str(pj),
 'scalar_density_ratio_bound':'3/2','scalar_true_vs_wrong_tilt_KL_bound':'1/30',
 'old_conservative_failure_allowance_preserved_ball':str(17280*pj+17408*pg),
 'exact_sampler_old_expected_upper_preserved_ball':str((96*pj+224*pg).upper()),'heun':rows}
output=HERE/'pro12_sharp_wrong_head_check.json'
if output.exists():raise FileExistsError(output)
output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
