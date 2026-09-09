"""Read-only frozen synthetic benchmark record audit; no timing or model fit."""
import argparse,json,hashlib,subprocess,random
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);p.add_argument('--repo',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
read=lambda name:json.loads((a.results/name).read_text())
sha=lambda b:hashlib.sha256(b).hexdigest()
c,m,g=read('COMPLETE.json'),read('metadata.json'),read('GATES_PASSED.json')
assert c['status']=='completed_synthetic_kernel_benchmark'
assert m['source_commit']=='bb024437a761c096375a042239f77262724dca14'
assert set(c['payload_sha256'])=={p.name for p in a.results.iterdir() if p.is_file()}-{'COMPLETE.json'}
for name,h in c['payload_sha256'].items():assert sha((a.results/name).read_bytes())==h
for name,h in m['source_sha256'].items():assert sha(subprocess.check_output(['git','show',m['source_commit']+':'+name],cwd=a.repo))==h
assert g=={'all_numerical_and_gradient_checks_passed':True,'check_count':78,'timing_has_started':False}
checks=read('numerical_checks.json');assert len(checks)==78 and all(x['passed'] for x in checks.values())
for key,value in checks.items():
 assert np.isfinite(value['max_absolute_error'])
 if not key.startswith('gradient_'):
  assert value['rtol']==0 and value['max_absolute_error']<=value['atol']
  assert value['atol']==(1e-10 if key.startswith('small64_') else 1e-4)
 else:assert value['atol']==1e-4 and value['rtol']==1e-3
arms=('reference','dense_eager','dense_compiled');variants=[(arm,inverse) for inverse in (False,True) for arm in arms]
rng=random.Random(20260910);order=[]
for _ in range(30):
 v=variants.copy();rng.shuffle(v);order.append(v)
assert read('order.json')==[[list(v) for v in row] for row in order]
rows=read('timings.json');assert len(rows)==180
for row,(rep,variant) in zip(rows,[(r,v) for r,vs in enumerate(order) for v in vs]):
 assert row['repetition']==rep and (row['arm'],row['inverse'])==variant
 assert row['wall_seconds']>0 and row['cuda_event_milliseconds']>0
 assert row['peak_allocated_bytes']-row['allocated_before_bytes']==row['peak_incremental_allocated_bytes']>=0
 assert row['peak_reserved_bytes']>=row['peak_allocated_bytes']
summary=read('summary.json');computed={}
for arm,inverse in variants:
 selected=[r for r in rows if r['arm']==arm and r['inverse']==inverse];assert len(selected)==30
 key=arm+('_inverse' if inverse else '_forward')
 r={'repetitions':30,'median_wall_seconds':float(np.median([x['wall_seconds'] for x in selected])),
    'median_cuda_event_milliseconds':float(np.median([x['cuda_event_milliseconds'] for x in selected])),
    'max_peak_allocated_bytes':max(x['peak_allocated_bytes'] for x in selected),
    'max_peak_incremental_allocated_bytes':max(x['peak_incremental_allocated_bytes'] for x in selected)}
 assert summary['variants'][key]==r;computed[key]=r
setup=read('first_use_setup.json');assert len(setup)==24
assert summary['compile_first_use_wall_seconds_sum']==sum(x['first_use_wall_seconds'] for x in setup if x['arm']=='dense_compiled')
inputs={}
with np.load(a.results/'fabricated_inputs.npz',allow_pickle=False) as archive:
 assert len(archive.files)==12
 for label,shape,dtype in [('full32',(64,2880),np.float32),('small32',(4,17),np.float32),('small64',(4,17),np.float64)]:
  for i,k in enumerate((None,8,8,7)):
   x=archive[f'{label}_{i}'];assert x.shape==(shape if k is None else (*shape,k)) and x.dtype==dtype and np.isfinite(x).all()
   inputs[f'{label}_{i}']=sha(x.tobytes())
maxima={name:max(v['max_absolute_error'] for k,v in checks.items() if k.startswith(name)) for name in ('full32_','small32_','small64_','gradient_')}
ratios={}
for direction in ('forward','inverse'):
 ref=computed['reference_'+direction];eager=computed['dense_eager_'+direction];compiled=computed['dense_compiled_'+direction]
 ratios[direction]={'compiled_wall_ratio_reference_over_compiled':ref['median_wall_seconds']/compiled['median_wall_seconds'],
  'eager_wall_ratio_reference_over_eager':ref['median_wall_seconds']/eager['median_wall_seconds'],
  'compiled_incremental_peak_ratio':compiled['max_peak_incremental_allocated_bytes']/ref['max_peak_incremental_allocated_bytes']}
result={'status':'audit_pass','payload_hashes_verified':len(c['payload_sha256']),'source_git_blob_hashes_verified':len(m['source_sha256']),
 'numerical_check_count':78,'timing_count':180,'order_reproduced':True,'summary_exactly_recomputed':True,
 'gpu_outputs_gradients_independently_recomputed':False,'numerical_maxima':maxima,'input_raw_array_sha256':inputs,
 'computed_summary':computed,'ratios':ratios,'first_use_compiled_sum_seconds':summary['compile_first_use_wall_seconds_sum'],
 'setup_rows':setup,'closure_sha256':sha((a.results/'COMPLETE.json').read_bytes())}
a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ('input_raw_array_sha256','setup_rows')},indent=2))
