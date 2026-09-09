"""Independent saved-state ideal-density quadrature and numerical audit; no fits."""
import json,hashlib,subprocess,importlib.util,argparse
from pathlib import Path
import numpy as np
parser=argparse.ArgumentParser();parser.add_argument('--results',type=Path,required=True);parser.add_argument('--repo',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();root=args.results;read=lambda n:json.loads((root/n).read_text());inventory=read('ARTIFACTS.json');status=read('status.json');summary=read('summary.json');env=read('environment.json');freeze=read('ALL_FITS_FROZEN.json');rev='effc2c565b050193561978c9bce686b3a3f996eb'
assert status['state']=='complete' and len(status['cells'])==6 and freeze['before_any_population_evaluation'] and freeze['cells']==status['cells'] and env['source_commit']==rev
verified=[]
for name,item in inventory.items():
 f=root/name
 if f.exists():assert f.stat().st_size==item['bytes'] and hashlib.sha256(f.read_bytes()).hexdigest()==item['sha256'];verified.append(name)
for name,h in env['source_sha256'].items():
 blob=subprocess.check_output(['git','show',rev+':'+name],cwd=args.repo);assert hashlib.sha256(blob).hexdigest()==h and (root/'sources'/Path(name).parent.name/Path(name).name).read_bytes()==blob
spec=importlib.util.spec_from_file_location('independent_population',Path(__file__).with_name('trapezoid_independent_population.py'));module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
result={};max_pop=max_refine=max_rt=max_ld=max_density=0.;sources={}
for cell in status['cells']:
 truth=read(cell+'/truth_evaluator_only.json');cfg=read(cell+'/continuous.json')['config'];seed=truth['fixture_seed'];rows={};root_probs=np.asarray(truth['root_probabilities']);assert np.all(root_probs*cfg['root_bins']>=.5) and np.array_equal(root_probs[0],np.full(cfg['root_bins'],1/cfg['root_bins']))
 z=np.load(root/cell/'common_gaussian.npy',allow_pickle=False);assert z.shape==(64,3072) and z.dtype==np.float64 and np.isfinite(z).all();replay=np.random.default_rng(seed+30000000).standard_normal((64,3072));sources[cell]={'byte_exact_local_replay':bool(np.array_equal(z,replay)),'max_local_replay_error':float(np.max(abs(z-replay)))}
 for arm in ['continuous','binned','constant','product']:
  state=read(cell+'/'+arm+'.json');reported=summary[cell][arm];assert reported==read(cell+'/'+arm+'_evaluation.json')
  for block,edges in enumerate(state['pairs']):
   endpoints=[v for edge in edges for v in edge];assert len(endpoints)==len(set(endpoints)) and (len(edges)==0 or sorted(endpoints)==list(range(cfg['block_size'])))
  high=module.direct_population(truth,state,80,48);low=module.direct_population(truth,state,48,32)
  discrepancy=max(abs(high['joint_kl']-reported['population']['joint_kl']),abs(high['root_kl']-reported['population']['root_kl']),max(abs(x-y) for x,y in zip(high['residual_block_kl'],reported['population']['residual_block_kl'])))
  refinement=abs(high['joint_kl']-low['joint_kl']);assert discrepancy<=1e-9 and refinement<=1e-9 and high['joint_kl']>=0
  assert high['correct_pairs_per_block']==reported['population']['correct_pairs_per_block'];max_pop=max(max_pop,discrepancy);max_refine=max(max_refine,refinement)
  with np.load(root/cell/(arm+'_numerical.npz'),allow_pickle=False) as f:source=f['source'];output=f['output'];ld=f['forward_logdet']
  with np.load(root/cell/(arm+'_inverse.npz'),allow_pickle=False) as f:recovered=f['recovered'];ild=f['inverse_logdet'];logp=f['log_prob']
  assert np.array_equal(source,z) and all(np.isfinite(x).all() for x in [output,ld,recovered,ild,logp]);assert output.min()>0 and output.max()<1
  rt=float(np.max(abs(source-recovered)));cancel=float(np.max(abs(ld+ild)));assert rt==reported['max_source_roundtrip'] and cancel==reported['max_logdet_cancellation'] and rt<=1e-9 and cancel<=1e-8
  density=float(np.max(abs(logp-(-.5*(recovered**2+np.log(2*np.pi)).sum(1)+ild))));assert density<=1e-8;max_rt=max(max_rt,rt);max_ld=max(max_ld,cancel);max_density=max(max_density,density)
  if arm=='continuous':
   with np.load(root/cell/'exact_copy.npz',allow_pickle=False) as f:assert np.array_equal(f['output'],output) and np.array_equal(f['forward_logdet'],ld) and np.array_equal(f['log_prob'],logp)
   assert reported['exact_copy_equal']
  rows[arm]={'independent_population':high,'vs_series_max_abs_discrepancy':discrepancy,'direct_quadrature_refinement_difference':refinement,'pair_counts':[len(e) for e in state['pairs']],'graph_failures':state['graph_failures'],'max_source_roundtrip':rt,'max_ld_cancellation':cancel,'gaussian_change_of_variables_logp_error':density}
  print(cell+'/'+arm+' verified',flush=True)
 result[cell]=rows
for seed in [13109101,13109102,13109103]:assert np.array_equal(np.load(root/f'{seed}_positive/common_gaussian.npy'),np.load(root/f'{seed}_zero_mean/common_gaussian.npy'))
record={'status':'independent_saved_state_and_population_audit_pass','source_files':len(env['source_sha256']),'locally_verified_inventory_payloads':len(verified),'inventory_payloads':len(inventory),'omitted_local_payloads':{n:v for n,v in inventory.items() if n not in verified},'population_models':24,'exact_copy_controls':6,'max_population_vs_series_discrepancy':max_pop,'max_direct_quadrature_refinement_difference':max_refine,'max_source_roundtrip':max_rt,'max_logdet_cancellation':max_ld,'max_gaussian_change_of_variables_logp_error':max_density,'registered_numerical_sources':sources,'results':result,'limitations':['Ordinary numerical quadrature, not certified intervals.','No refitting or independent reproduction of fitted coefficients; full fit arrays retained and hash-checked remotely.','No neural, native-data or unrestricted latent superiority claim.']}
with args.output.open('x') as f:json.dump(record,f,indent=2)
print(json.dumps({k:v for k,v in record.items() if k not in ['results','omitted_local_payloads','registered_numerical_sources']}),flush=True)
