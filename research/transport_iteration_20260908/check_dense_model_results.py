import argparse,json,hashlib,subprocess,random
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);p.add_argument('--repo',type=Path,required=True);p.add_argument('--native',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();root=a.results
read=lambda name:json.loads((root/name).read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
c=read('COMPLETE.json');s=read('source_identity.json');eq=read('equivalence.json');summary=read('summary.json');native=read('native_identity.json')
assert c['status']=='completed_checkpoint_backend_benchmark' and s['commit']=='3d801dcc1508cc3e203df5ef97c3834f2462c50e'
assert set(c['payload_sha256'])=={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() and p.name!='COMPLETE.json'}
for name,h in c['payload_sha256'].items():assert sha(root/name)==h
for i,(name,h) in enumerate(s['sha256'].items()):
 b=subprocess.check_output(['git','show',s['commit']+':'+name],cwd=a.repo);assert hashlib.sha256(b).hexdigest()==h
 assert (root/'sources'/f'{i:02d}_{Path(name).name}').read_bytes()==b
assert sha(a.native/'status.json')==native['terminal_status_sha256']
for name,h in native['accessed_payload_sha256'].items():assert sha(a.native/name)==h
saved_source=np.load(root/'common_gaussian_source.npy',allow_pickle=False)
with np.load(a.native/'coupling/chunk_0000.npz',allow_pickle=False) as z:assert np.array_equal(saved_source,z['gaussian'])
computed={}
def check(name,x,y,report):
 assert x.shape==y.shape and np.isfinite(x).all() and np.isfinite(y).all()
 err=np.abs(x-y);passed=bool(np.all(err<=report['atol']+report['rtol']*np.abs(y)))
 maximum=float(err.max(initial=0));assert maximum==report['maximum_absolute_error'] and passed==report['passed']==True
 computed[name]=maximum
for batch in (1,64):
 ref=np.load(root/f'pipeline_{batch}_reference.npy',allow_pickle=False)
 for arm in ('reference','dense_eager','dense_compiled'):
  x=np.load(root/f'pipeline_{batch}_{arm}.npy',allow_pickle=False);check(f'pipeline_{batch}_{arm}',x,ref,eq[f'pipeline_{batch}_{arm}'])
with np.load(root/'backward_reference.npz',allow_pickle=False) as z:ref={k:z[k] for k in z.files}
for arm in ('reference','dense_eager','dense_compiled'):
 with np.load(root/f'backward_{arm}.npz',allow_pickle=False) as z:
  assert set(z.files)==set(ref)==set(eq['backward_'+arm])
  for name in z.files:check('backward_'+arm+'_'+name,z[name],ref[name],eq['backward_'+arm][name])
rt=read('roundtrip_checks.json')
with np.load(root/'generated_backward_inputs.npz',allow_pickle=False) as z:
 assert float(np.max(np.abs(ref['encoded']-z['noise'])))==rt['conditional_noise_roundtrip']
 assert float(np.max(np.abs(z['code']-z['recovered_code'])))==rt['analysis_code_roundtrip']
assert rt['conditional_noise_roundtrip']<=1e-3 and rt['analysis_code_roundtrip']<=1e-3
assert rt['conditional_logdet_cancellation']<=1e-2 and rt['analysis_logdet_cancellation']<=1e-2
assert not rt['coarse_Heun_inverse_verified'] and not rt['full_source_inverse_claim']
rows=read('timings.json');orders=read('order.json');assert len(rows)==270 and len(orders)==90
rng=random.Random(2026091101);expected=[]
for case in ('pipeline_1','pipeline_64','encode_nll_backward_32'):
 for repeat in range(30):
  order=['reference','dense_eager','dense_compiled'];rng.shuffle(order)
  expected.append({'case':case,'repeat':repeat,'order':order})
assert expected==orders
for row,(case,repeat,arm) in zip(rows,[(o['case'],o['repeat'],arm) for o in orders for arm in o['order']]):
 assert (row['case'],row['repeat'],row['arm'])==(case,repeat,arm)
 assert all(np.isfinite(row[k]) and row[k]>=0 for k in ('wall_seconds','cuda_event_milliseconds','movement_seconds','post_call_finite_check_seconds'))
 assert row['incremental_peak_allocated_bytes']==row['peak_allocated_bytes']-row['baseline_allocated_bytes']>=0
extra={}
for case in ('pipeline_1','pipeline_64','encode_nll_backward_32'):
 for arm in ('reference','dense_eager','dense_compiled'):
  v=[r for r in rows if r['case']==case and r['arm']==arm];assert len(v)==30
  r={'repetitions':30,'median_wall_seconds':float(np.median([x['wall_seconds'] for x in v])),
     'median_cuda_event_ms':float(np.median([x['cuda_event_milliseconds'] for x in v])),
     'maximum_incremental_peak_bytes':max(x['incremental_peak_allocated_bytes'] for x in v)}
  assert r==summary['cases'][case+'_'+arm]
  extra[case+'_'+arm]={'median_wall_plus_postcheck_seconds':float(np.median([x['wall_seconds']+x['post_call_finite_check_seconds'] for x in v])),
                     'median_movement_seconds':float(np.median([x['movement_seconds'] for x in v]))}
assert summary['setup']==read('setup.json')
assert summary['compiled_first_use_seconds']==sum(x.get('first_use_seconds',0) for x in summary['setup'] if x.get('arm')=='dense_compiled')
result={'status':'audit_pass','payload_hashes':len(c['payload_sha256']),'source_hashes_and_snapshots':len(s['sha256']),
        'native_payload_hashes_verified':len(native['accessed_payload_sha256']),'common_gaussian_exact':True,
        'saved_array_equivalence_checks':len(computed),'max_pipeline_error':max(v for k,v in computed.items() if k.startswith('pipeline_')),
        'max_gradient_error':max(v for k,v in computed.items() if 'gradient::' in k),
        'numerical_checks':rt,'timing_samples':270,'seeded_order_exact':True,'summary_recomputed':True,
        'latency_with_postchecks':extra,'model_rerun':False,'source_commit':s['commit']}
a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
