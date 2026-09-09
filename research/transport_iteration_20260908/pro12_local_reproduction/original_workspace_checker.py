"""Audit completed local certificate receipts; no integration, model fits or jobs."""
from pathlib import Path
import json,hashlib,subprocess,platform
from decimal import Decimal
w=Path(__file__).resolve().parent
out=w/'pro12-local-reproduction'
original=w/'diffusion-representation/research/transport_iteration_20260908/pro12_artifacts'
def read(p):return json.loads(p.read_text(),parse_float=Decimal)
complete=json.loads((out/'COMPLETE.json').read_text())
assert complete['status']=='completed_all_requested_checks' and complete['full_fixed_grid']
for rel,h in complete['payload_sha256'].items():assert hashlib.sha256((out/rel).read_bytes()).hexdigest()==h,rel
assert not (out/'FAILED.json').exists()
auth=json.loads((w/'pro12-local-authentication.json').read_text())
for rel,h in json.loads((out/'SOURCE_HASHES.json').read_text()).items():
    assert hashlib.sha256((original/rel).read_bytes()).hexdigest()==h
results=[]
for n in (4,8,16,32,64):
    raw=read(out/f'n{n}_first_attempt.json'); hist=read(original/f'results/n{n}_first_attempt.json')
    risk=read(out/f'n{n}_risk.json'); historicalrisk=read(original/f'results/n{n}_risk.json')
    a,b=raw['conditional_KL'];lo,hi=risk['unconditional_KL']
    assert 0<=a<=b and 0<=lo<=hi and hi-lo<Decimal('2e-8')
    assert max(a,hist['conditional_KL'][0])<=min(b,hist['conditional_KL'][1])
    exact=raw['conditional_KL']==hist['conditional_KL']
    results.append({'N':n,'conditional_KL':[str(a),str(b)],'expected_KL':[str(lo),str(hi)],'conditional_receipt_equal':exact,'expected_receipt_equal':risk['unconditional_KL']==historicalrisk['unconditional_KL'],'seconds':str(raw['seconds']),'cells':raw['cells'],'mpfr':raw['mpfr']})
alt=read(out/'n4_order8_precision192.json');main=read(out/'n4_first_attempt.json')
assert max(alt['conditional_KL'][0],main['conditional_KL'][0])<=min(alt['conditional_KL'][1],main['conditional_KL'][1])
assert Decimal(results[-1]['expected_KL'][1])<Decimal('1e-5')
processes=[]
for f in sorted(out.glob('*.process.json')):
    p=json.loads(f.read_text());assert p['returncode']==0 and not p['timeout'];processes.append({'file':f.name,**p})
unit=json.loads((out/'unit_tests.json').read_text());assert unit['status']=='pass' and unit['checks']==256
report={'status':'independently_reproduced_all_five_plus_changed_order','scope':'ideal-real scalar maps and exact-real catalog-learning expected risk; not floating generator KL or model fitting','authenticated_original_files':len(auth['authenticated_files']),'authenticated_output_payloads':len(complete['payload_sha256']),'checks':unit,'platform':platform.platform(),'machine':platform.machine(),'certificate_binary_linkage':subprocess.check_output(['/usr/bin/otool','-L',str(out/'certify')]).decode(),'results':results,'changed_order_N4_interval':list(map(str,alt['conditional_KL'])),'changed_order_seconds':str(alt['seconds']),'all_primary_integration_seconds':str(sum(read(out/f'n{n}_first_attempt.json')['seconds'] for n in (4,8,16,32,64))),'processes':processes,'historical_cost_replay':False,'new_fit_or_sampler_benchmark':False}
with (w/'pro12-local-independent-check.json').open('x') as f:json.dump(report,f,indent=2)
print(json.dumps({k:report[k] for k in ('status','authenticated_output_payloads','results','all_primary_integration_seconds')},indent=2))
