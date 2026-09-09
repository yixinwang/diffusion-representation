"""Audit saved synthetic reference reports; no fitting or timing reruns."""
import argparse,hashlib,json,subprocess,types
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);p.add_argument('--repo',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
read=lambda p:json.loads(p.read_text());root=a.results;closure=read(root/'COMPLETE.json');identity=read(root/'source_identity.json');manifest=read(root/'manifest.json');summary=read(root/'summary.json')
sha=lambda b:hashlib.sha256(b).hexdigest()
def ah(x):
 x=np.ascontiguousarray(x);h=hashlib.sha256(str((x.shape,x.dtype.str)).encode());h.update(x.view(np.uint8));return h.hexdigest()
assert closure['status']=='completed_synthetic_validation'
assert set(closure['payload_sha256'])=={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() and p.name!='COMPLETE.json'}
for name,h in closure['payload_sha256'].items():assert sha((root/name).read_bytes())==h
commit='82bb8bf6dac93e9ad7ea63c70ec7ca4ac77bccfa';refsha='38cb7d9b79111b99e3af961aa28cdf2ca6f5a2b427245434e2cea407c7b6c73b'
assert identity['source_commit']==manifest['source_commit']==commit
assert identity['reference_sha256']==manifest['reference_sha256']==refsha
for i,(name,h) in enumerate(identity['source_sha256'].items()):
 frozen=subprocess.check_output(['git','show',commit+':'+name],cwd=a.repo)
 assert sha(frozen)==h and (root/'sources'/f'{i:02d}_{Path(name).name}').read_bytes()==frozen
source=(root/'sources/00_positive_class_reference.py').read_bytes();assert sha(source)==refsha
m=types.ModuleType('audit_original_reference');exec(compile(source,'verified_snapshot','exec'),m.__dict__)
def forbid(*args,**kwargs):raise RuntimeError('no fitting/timing reruns permitted')
m.fit=forbid;m.run=forbid
saved_dictionary=np.load(root/'public_dictionary.npy',allow_pickle=False)
assert saved_dictionary.shape==(16,192) and np.isfinite(saved_dictionary).all()
new_dictionary=m.orthogonal_dictionary()
dictionary_error=float(np.max(np.abs(new_dictionary-saved_dictionary)))
# Preserved dictionary is authoritative across QR backend rounding differences.
provenance=[];timings=[];numerical=[];selections=[]
def streams(seed,gamma,index,expected,with_timing=False):
 rng=np.random.default_rng(seed)
 zs=[rng.normal(size=(n,3072)) for n in (32,256)]
 actual={'root_source':ah(zs[0]),'head_source':ah(zs[1]),'dictionary':ah(saved_dictionary)}
 # Recreate only already frozen synthetic observations, never fit or evaluate quality.
 actual['root_observations']=ah(m.generate(zs[0],gamma,index,saved_dictionary))
 actual['head_observations']=ah(m.generate(zs[1],gamma,index,saved_dictionary))
 orders={}
 if with_timing:
  actual['roundtrip_source']=ah(rng.normal(size=(32,3072)))
  labels=['exact','fm_4','fm_8','fm_16','fm_32','fm_64']
  for batch in (1,64):
   actual[f'timing_source_batch_{batch}']=ah(rng.normal(size=(batch,3072)))
   orders[str(batch)]=[rng.permutation(labels).tolist() for _ in range(9)]
 matches={k:actual[k]==h for k,h in expected.items()}
 assert matches['dictionary']
 provenance.append({'seed':seed,'hash_matches':matches})
 return orders
for seed in (2026090901,2026090902,2026090903):
 case=root/f'timing_seed_{seed}';r=read(case/'raw_reference_report.json');prov=read(case/'provenance.json');status=read(case/'status.json')
 assert status['status']=='completed' and r['training']['fitted_gamma']==.5 and r['training']['fitted_index']==7 and r['training']['correct_selection']
 selections.append(True);num=r['numerical'];numerical.append(num)
 for key,tol in [('full_source_roundtrip_max',1e-9),('head_tail_roundtrip_max_on_abs50',1e-10),('orthogonal_dictionary_error',1e-12)]:assert np.isfinite(num[key]) and num[key]<=tol
 assert num['exact_copy_bitwise']
 orders=streams(seed,.5,7,prov['hashes'],True);assert orders==prov['reconstructed_interleaving']
 for batch,table in r['timing'].items():
  assert set(table['raw_seconds'])=={'exact','fm_4','fm_8','fm_16','fm_32','fm_64'}
  for arm,values in table['raw_seconds'].items():
   assert len(values)==9 and np.isfinite(values).all() and min(values)>0
   med=float(np.median(values));assert med==table['median_seconds'][arm]
   if arm!='exact':assert med/table['median_seconds']['exact']==table['fm_over_exact'][arm]
   timings.append({'seed':seed,'batch':int(batch),'arm':arm,'median_seconds':med})
for offset,gamma in enumerate((-.5,.5)):
 for index in range(16):
  seed=2026091001+offset*16+index;label='minus' if gamma<0 else 'plus';case=root/f'recovery_{label}_index_{index:02d}_seed_{seed}'
  r=read(case/'report.json');inp=read(case/'inputs.json');scores=np.array(r['all_dictionary_scores'])
  assert r['seed']==seed and r['truth_gamma']==gamma and r['truth_index']==index
  assert len(scores)==16 and np.isfinite(scores).all() and int(scores.argmax())==r['fitted_index']==index
  assert r['fitted_gamma']==gamma and r['correct_selection'] and r['correct_sign'] and r['correct_index']
  assert r['score_gap']==float(np.sort(scores)[-1]-np.sort(scores)[-2])
  assert inp['hashes']==r['hashes'];streams(seed,gamma,index,inp['hashes']);selections.append(True)
assert len(selections)==35 and summary['timing_correct_selections']==3 and summary['recovery_correct_selections']==32
assert len(summary['timing_runs'])==3 and len(summary['recovery_cells'])==32
result={'status':'audit_pass','payload_hashes_verified':len(closure['payload_sha256']),'source_hashes_and_snapshots_verified':len(identity['source_sha256']),
 'correct_fit_selections':sum(selections),'raw_timing_samples':len(timings)*9,'timing_medians_independently_recomputed':timings,
 'numerical_reports':numerical,'source_dictionary_regeneration_max_abs':dictionary_error,
 'frozen_stream_reconstruction':provenance,'reference_sha256':refsha,'no_fitting_or_timing_rerun':True}
a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ('frozen_stream_reconstruction','timing_medians_independently_recomputed')},indent=2))
