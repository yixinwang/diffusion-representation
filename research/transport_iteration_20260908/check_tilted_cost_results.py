"""Independent saved-output and timing audit. No fit or benchmark timing rerun."""
import argparse,hashlib,json,subprocess,random,types,sys
from pathlib import Path
import numpy as np
from scipy.special import ndtr
p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);p.add_argument('--repo',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();root=a.results
read=lambda name:json.loads((root/name).read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
c=read('COMPLETE.json');identity=read('source_identity.json');meta=read('metadata.json');checks=read('checks.json');summary=read('summary.json');gates=read('GATES_PASSED.json')
assert c['status']=='completed_timing' and c['timing_performed']
assert identity['source_commit']=='9391d3ab89ad0667036d39a85f2e59409da34dc6'
# Published schema uses payload_sha256; never infer missing payload completion.
assert set(c['payload_sha256'])=={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() and p.name!='COMPLETE.json'}
for name,h in c['payload_sha256'].items():assert sha(root/name)==h
for i,(name,h) in enumerate(identity['sha256'].items()):
 snap=root/'sources'/f'{i:02d}_{Path(name).name}';frozen=subprocess.check_output(['git','show',identity['source_commit']+':'+name],cwd=a.repo)
 assert snap.read_bytes()==frozen and hashlib.sha256(frozen).hexdigest()==h
assert sha(root/'sources/00_positive_class_reference.py')=='38cb7d9b79111b99e3af961aa28cdf2ca6f5a2b427245434e2cea407c7b6c73b'
assert sha(root/'sources/01_positive_class_results.json')=='92a0203e85f838a06f8f265359a12d1fb21839e2fa55ae99942446232396cdd3'
assert sha(root/'sources/03_candidate.py')=='e660a4d814eb25ad16405ccd72ca7c47b7d2bd1de749ca394524a36a88e42bbe'
def load(name,path):
 m=types.ModuleType(name);sys.modules[name]=m;exec(compile(path.read_bytes(),str(path),'exec'),m.__dict__);return m
ref=load('frozen_cost_reference',root/'sources/00_positive_class_reference.py');candidate=load('frozen_cost_candidate',root/'sources/03_candidate.py')
def forbidden(*args,**kwargs):raise RuntimeError('fitting/timing runner prohibited')
ref.fit=ref.run=forbidden
params=read('sources/01_positive_class_results.json')['training'];gamma,index=params['fitted_gamma'],params['fitted_index']
assert meta['gamma']==gamma and meta['summary_index']==index and not meta['archived_training_performed_here']
seeds=[2026090901,2026090902,2026090903];nfes=[4,8,16,32,64]
arms=['original_exact','central_exact']+[f'{kind}_{n}' for kind in ('original_fm','cached_fm','endpoint_fm') for n in nfes]
assert meta['seeds']==seeds and meta['batches']==[1,64] and meta['primary_arms']==arms
assert len(checks)==78 and gates=={'passed':True,'checks':78,'timing_started':False}
dictionary=np.load(root/'public_dictionary.npy',allow_pickle=False);assert dictionary.shape==(16,192)
assert np.max(np.abs(dictionary@dictionary.T-np.eye(16)))<1e-12
comparison_errors={};roundtrip_errors={};replay_errors={};source_hash_matches={};output_count=0;secondary_replay_errors={}
for seed in seeds:
 rng=np.random.default_rng(seed)
 for batch in (1,64):
  source=np.load(root/f'source_{seed}_{batch}.npy',allow_pickle=False);regenerated=rng.normal(size=(batch,3072))
  assert source.shape==(batch,3072) and source.dtype==np.float64 and np.isfinite(source).all()
  source_hash_matches[f'{seed}_{batch}']=np.array_equal(source,regenerated)
  outputs={}
  for arm in arms:
   x=np.load(root/f'output_{seed}_{batch}_{arm}.npy',allow_pickle=False);assert x.shape==(batch,3,32,32) and x.dtype==np.float64 and np.isfinite(x).all() and x.min()>=0 and x.max()<=1
   outputs[arm]=x;output_count+=1
  pairs=[('central_exact','original_exact')]+[(f'{kind}_{n}',f'original_fm_{n}') for kind in ('cached_fm','endpoint_fm') for n in nfes]
  for left,right in pairs:
   key=f'{seed}_{batch}_{left}';err=float(np.max(np.abs(outputs[left]-outputs[right])))
   assert err==checks[key]['max_reference_output_error'] and err<=1e-12;comparison_errors[key]=err
  for arm in ('original_exact','central_exact'):
   coarse,residual=ref.outer_encode(outputs[arm]);u=ref.inverse_transport(coarse,gamma)
   tilt=.5+.1*(2*ndtr(u@dictionary[index])-1)
   back=np.concatenate((u,ref.inverse_transport(residual,tilt[:,None])),axis=1)
   err=float(np.max(np.abs(back-source)));key=f'{seed}_{batch}_{arm}_roundtrip'
   assert err<=2e-10 and abs(err-checks[key]['max_source_error'])<=1e-12;roundtrip_errors[key]=err
  # Recompute frozen reference outputs only; no new quality evaluation or timings.
  for arm in ['original_exact']+[f'original_fm_{n}' for n in nfes]:
   x=ref.generate(source,gamma,index,dictionary) if arm=='original_exact' else ref.generate(source,gamma,index,dictionary,'fm',int(arm.rsplit('_',1)[1]))
   err=float(np.max(np.abs(x-outputs[arm])));assert err<=1e-12;replay_errors[f'{seed}_{batch}_{arm}']=err
  # Independently reconstruct the secondary all-call cached schedule path.
  # It is NOT endpoint-specialized: first/final kernels are both evaluated.
  coarse=candidate.quantile_transport(source[:,:192],gamma)
  tilt=.5+.1*(2*ndtr(source[:,:192]@dictionary[index])-1)
  for n in nfes:
   h=2/n;cache=tuple(candidate.time_constants(i*h) for i in range(n//2+1))
   residual=source[:,192:].copy();calls=0
   for step in range(n//2):
    first=candidate.cached_field(residual,tilt[:,None],cache[step]);calls+=1
    second=candidate.cached_field(residual+h*first,tilt[:,None],cache[step+1]);calls+=1
    residual+=h/2*(first+second)
   assert calls==n
   rebuilt=ref.outer_decode(coarse,residual)
   error=float(np.max(np.abs(rebuilt-outputs[f'cached_fm_{n}'])))
   assert error<=1e-12;secondary_replay_errors[f'{seed}_{batch}_{n}']=error
assert output_count==102
primary=read('primary_timings.json');secondary=read('secondary_amortized_timings.json');orders=read('orders.json');memory=read('memory_diagnostics.json');schedule=read('schedule_setup.json')
assert len(primary)==3060 and len(secondary)==900 and len(orders)==180 and len(memory)==102 and len(schedule)==6
expected_primary=[];expected_secondary=[];expected_orders=[]
for seed in seeds:
 for batch in (1,64):
  rng=random.Random(seed+batch+401)
  for rep in range(30):
   order=arms.copy();rng.shuffle(order);expected_orders.append({'phase':'primary','seed':seed,'batch':batch,'rep':rep,'arms':order})
   expected_primary.extend((seed,batch,arm,rep) for arm in order)
  for rep in range(30):
   order=nfes.copy();rng.shuffle(order);expected_secondary.extend((seed,batch,n,rep) for n in order)
assert orders==expected_orders
for row,key in zip(primary,expected_primary):assert (row['seed'],row['batch'],row['arm'],row['rep'])==key and isinstance(row['nanoseconds'],int) and row['nanoseconds']>0
for row,key in zip(secondary,expected_secondary):assert (row['seed'],row['batch'],row['nfe'],row['rep'])==key and isinstance(row['nanoseconds'],int) and row['nanoseconds']>0
secondary_medians={}
for seed in seeds:
 for batch in (1,64):
  for arm in arms:
   values=[r['nanoseconds'] for r in primary if (r['seed'],r['batch'],r['arm'])==(seed,batch,arm)];assert len(values)==30
   assert float(np.median(values))/1e9==summary['median_seconds'][f'{seed}_{batch}'][arm]
  secondary_medians[f'{seed}_{batch}']={str(n):float(np.median([r['nanoseconds'] for r in secondary if (r['seed'],r['batch'],r['nfe'])==(seed,batch,n)]))/1e9 for n in nfes}
for row in memory:assert row['traced_peak_minus_baseline_bytes']>=0 and row['output_bytes']==row['batch']*3072*8
for row in schedule:assert row['all_schedules_setup_nanoseconds']>0
# Count actual nontrivial field dispatches on a tiny slice of an existing source.
field=candidate.cached_field;counts={}
for n in nfes:
 counter=[0]
 def counted(*args):counter[0]+=1;return field(*args)
 candidate.cached_field=counted
 y,account=candidate.endpoint_heun(source[:1,:4],.5,n)
 assert counter[0]==n-2 and account['equivalent_mathematical_stages']==n and account['nontrivial_field_kernel_calls']==n-2 and account['analytic_endpoint_stages']==2 and not account['final_predictor_allocated']
 assert {k:account[k] for k in meta['endpoint_fm_call_accounting'][str(n)]}==meta['endpoint_fm_call_accounting'][str(n)]
 counter[0]=0;yy,regular=candidate.heun(source[:1,:4],.5,n);assert counter[0]==n and regular['field_calls']==n
 assert np.max(np.abs(y-yy))<=1e-12;counts[str(n)]={'mathematical_stages':n,'endpoint_field_calls':n-2,'regular_field_calls':n}
candidate.cached_field=field
result={'status':'audit_pass','payload_hashes':len(c['payload_sha256']),'source_hashes_and_snapshots':len(identity['sha256']),
 'outputs_verified':output_count,'gates_independently_checked':78,'max_reference_parity_error':max(comparison_errors.values()),
 'max_exact_roundtrip_error':max(roundtrip_errors.values()),'max_frozen_reference_replay_error':max(replay_errors.values()),
 'saved_gaussian_regeneration_exact':source_hash_matches,'primary_timings':3060,'secondary_timings':900,
 'all_orders_and_primary_medians_exact':True,'secondary_medians_seconds':secondary_medians,'secondary_arm_family':'cached_fm_N all-call Heun, schedule amortized; N actual field calls','secondary_reconstructed_outputs':len(secondary_replay_errors),'secondary_max_output_error':max(secondary_replay_errors.values()),'endpoint_call_accounting':counts,
 'one_archived_fitted_model':{'gamma':gamma,'index':index},'no_fit_or_timing_rerun':True,
 'memory_scope':'separate tracemalloc diagnostics, not intrinsic complexity/allocator RSS or production memory',
 'source_commit':identity['source_commit']}
a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ('secondary_medians_seconds',)},indent=2))
