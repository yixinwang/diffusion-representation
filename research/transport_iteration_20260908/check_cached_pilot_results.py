"""Frozen saved-bank audit; no fitting, generation or feature extraction."""
import argparse,hashlib,json,subprocess,sys
from pathlib import Path
import numpy as np
from scipy.special import expit
import torch
p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);p.add_argument('--repair',type=Path,required=True);p.add_argument('--repo',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();root=a.results
read=lambda n:json.loads((root/n).read_text())
def sha(f):
 h=hashlib.sha256()
 with f.open('rb') as stream:
  for block in iter(lambda:stream.read(2**20),b''):h.update(block)
 return h.hexdigest()
status=read('status.json');source=read('source_identity.json');assert status['status']=='completed_development_only' and status['all_fits_frozen']
assert set(status['payload_sha256'])=={f.relative_to(root).as_posix() for f in root.rglob('*') if f.is_file() and f.relative_to(root).as_posix()!='status.json'}
for n,h in status['payload_sha256'].items():assert sha(root/n)==h
for n,h in source['sha256'].items():
 blob=subprocess.check_output(['git','show',source['commit']+':'+n],cwd=a.repo);assert hashlib.sha256(blob).hexdigest()==h and (root/'sources'/n).read_bytes()==blob
repair=np.load(a.repair/'repair.npy',allow_pickle=False);ids=np.load(a.repair/'repair_ids.npy',allow_pickle=False);rstatus=json.loads((a.repair/'status.json').read_text())
assert rstatus['status']=='strict_canonical_repair_identity_pass' and repair.shape==(1000,3,32,32) and repair.dtype==np.float64
for n,h in rstatus['payload_sha256'].items():assert sha(a.repair/n)==h
assert np.isfinite(repair).all() and repair.min()>0 and repair.max()<1
with np.load(root/'record_ids.npz',allow_pickle=False) as native_ids:
 assert np.array_equal(ids,native_ids['repair']);fit_ids=native_ids['fit'].copy()
value_hash=hashlib.sha256(repair.tobytes()).hexdigest();id_hash=hashlib.sha256(np.asarray(ids,dtype='<i8').tobytes()).hexdigest()
def descriptors(x):
 gray=np.asarray(x,dtype=np.float64).mean(axis=1);tl=gray[:,:16,:16].mean((1,2));br=gray[:,16:,16:].mean((1,2));h=(np.diff(gray,axis=2)**2).mean((1,2));v=(np.diff(gray,axis=1)**2).mean((1,2));return np.column_stack([tl,br,tl*br,h,v,h*v])
real_desc=descriptors(repair);cov=lambda d:float(d[:,2].mean()-d[:,0].mean()*d[:,1].mean());outer=(-np.log(repair)-np.log1p(-repair)).reshape(1000,-1).sum(1)
feature_scope={};exec((Path(__file__).with_name('cached_pilot_feature_metrics.py')).read_text(),{'np':np},feature_scope);measure=feature_scope['measure']
results={};discrepancies={};noise_exact={};numerical_count=0;checkpoint_checks={};descriptor_max_error=0.;training_order_checks=0
for seed in [77201,77202,77203]:
 prefix=root/f'seed_{seed}';pair=read(f'seed_{seed}/pair_identity.json');assert pair['repair_values_sha256']==value_hash and pair['repair_ids_sha256']==id_hash and pair['dimension']==3072 and pair['pairs_per_image']==1
 noise=np.load(prefix/'common_gaussian.npy',allow_pickle=False);assert noise.shape==(2000,3072) and noise.dtype==np.float32 and np.isfinite(noise).all() and sha(prefix/'common_gaussian.npy')==pair['source_sha256']
 recreated=torch.randn(2000,3072,generator=torch.Generator().manual_seed(seed+300)).numpy();noise_exact[str(seed)]={'byte_exact':bool(np.array_equal(noise,recreated)),'max_abs_difference':float(np.max(abs(noise-recreated)))}
 fitting=json.loads((prefix/'fitting.json').read_text());stages=fitting['stages'];assert fitting['all_fits_frozen']
 fork_checkpoint=torch.load(prefix/'MJ_fork.pt',map_location='cpu',weights_only=True)
 for stage in ['analysis','root','MJ_prefix','M','J','S','RQS']:
  report=stages[stage];assert report['status']=='completed' and report['updates']>0 and len(report['losses'])==report['updates'] and np.isfinite(report['losses']).all()
  assert report['deadline_overrun_seconds']<=report['max_step_seconds']+.01
  rng=np.random.default_rng(seed+({'analysis':10,'root':20}.get(stage,30)))
  if stage in ['M','J']:rng.bit_generator.state=fork_checkpoint['rng']
  digest=hashlib.sha256()
  for _ in range(report['updates']):digest.update(np.asarray(fit_ids[rng.integers(0,len(fit_ids),size=32)],dtype='<i8').tobytes())
  assert digest.hexdigest()==report['record_order_sha256'];training_order_checks+=1
  checkpoint_name={'root':'shared','MJ_prefix':'MJ_fork'}.get(stage,stage)
  assert rng.bit_generator.state==torch.load(prefix/(checkpoint_name+'.pt'),map_location='cpu',weights_only=True)['rng']
 shared=torch.load(prefix/'shared.pt',map_location='cpu',weights_only=True)['model']
 frozen_prefixes=('pre_analysis.','analysis.','coarse_decoder.')
 checkpoint_checks[str(seed)]={}
 for name in ['M','S','RQS','J']:
  model=torch.load(prefix/(name+'.pt'),map_location='cpu',weights_only=True)['model']
  keys=[k for k in shared if k.startswith(('coarse_decoder.',) if name=='J' else frozen_prefixes)]
  assert keys and all(torch.equal(shared[k],model[k]) for k in keys)
  checkpoint_checks[str(seed)][name]={'frozen_shared_tensors_equal':len(keys),'joint_analysis_change_expected':name=='J'}
 real_features=np.load(prefix/'repair_features.npy',allow_pickle=False);assert real_features.shape==(1000,2048) and np.isfinite(real_features).all()
 for arm in ['S','M','J','RQS','A_only','root_only']:
  folder=prefix/arm;key=f'{seed}/{arm}';saved=json.loads((folder/'metrics.json').read_text());generated=np.load(folder/'generated.npy',mmap_mode='r',allow_pickle=False);logits=np.load(folder/'logits.npy',mmap_mode='r',allow_pickle=False)
  assert generated.shape==logits.shape==(2000,3,32,32) and generated.dtype==np.float64 and logits.dtype==np.float32
  assert np.isfinite(generated).all() and np.isfinite(logits).all() and generated.min()>=0 and generated.max()<=1 and np.array_equal(generated,generated.astype(np.float32).astype(np.float64))
  endpoint={'zero':int((generated==0).sum()),'one':int((generated==1).sum())};assert endpoint==saved['sigmoid_endpoint_counts']
  sigmoid_error=0.
  for first in range(0,2000,64):sigmoid_error=max(sigmoid_error,float(np.max(np.abs(generated[first:first+64]-expit(np.asarray(logits[first:first+64],dtype=np.float64))))))
  assert sigmoid_error<=1.2e-7
  desc=descriptors(generated);stored=np.load(folder/'descriptors.npy',allow_pickle=False);desc_error=float(np.max(np.abs(desc-stored)));assert desc_error<=1e-13;descriptor_max_error=max(descriptor_max_error,desc_error)
  draws=generated.reshape(1000,2,-1);target=repair.reshape(1000,-1);norm=lambda x:np.linalg.norm(x,axis=1)/np.sqrt(3072)
  energy=.5*(norm(draws[:,0]-target)+norm(draws[:,1]-target)-norm(draws[:,0]-draws[:,1]));stored=np.load(folder/'energy.npy',allow_pickle=False);assert np.max(np.abs(energy-stored))<=1e-14
  parts=np.load(folder/'repair_nll_components.npy',allow_pickle=False);assert parts.shape==(1000,5) and np.isfinite(parts).all();assert np.max(np.abs(parts[:,4]-parts[:,:4].sum(1)))<=1e-10;assert np.max(np.abs(parts[:,3]+outer))<=1e-10
  features=np.load(folder/'features.npy',allow_pickle=False);assert features.shape==(2000,2048) and features.dtype==np.float32 and np.isfinite(features).all();fm=measure(real_features,features)
  computed={'kid':fm['kid_unbiased_full_bank'],'energy_mean':float(energy.mean()),'covariance_error':abs(cov(desc)-cov(real_desc)),'complete_nll':float(parts[:,-1].mean()/3072),'residual_nll':float(parts[:,1].mean()/2880)}
  for k,v in computed.items():discrepancies[key+'/'+k]=abs(v-saved[k]);assert discrepancies[key+'/'+k]<=1e-10
  assert np.max(np.abs(desc[:,3:5].mean(0)-saved['gradient_means']))<=1e-14 and np.max(np.abs(real_desc[:,3:5].mean(0)-saved['repair_gradient_means']))<=1e-14
  for k,v in saved['prdc'].items():assert abs(v-fm[k])<=1e-10
  if arm in ['S','M','J','RQS']:
   z=np.load(folder/'numerical.npz',allow_pickle=False);assert z['source'].shape==(8,3072) and np.isfinite(z['source']).all();assert np.array_equal(z['logits'],z['exact_copy']) and np.max(abs(z['recovered']-z['source']))<=1e-3 and np.max(abs(z['ld']+z['inverse_ld']))<=1e-2;numerical_count+=1
  results[key]={'computed':computed,'prdc':{k:fm[k] for k in saved['prdc']},'endpoint_counts':endpoint,'float64_sigmoid_vs_saved_float32_max_error':sigmoid_error};print(key+' verified',flush=True)
record={'status':'full_saved_bank_metric_audit_pass','payload_hashes':len(status['payload_sha256']),'source_files':len(source['sha256']),'arms_audited':len(results),'numerical_banks':numerical_count,'local_different_environment_gaussian_replay':noise_exact,'max_metric_discrepancy':max(discrepancies.values()),'repair_value_sha256':value_hash,'repair_id_sha256':id_hash,'results':results,'checkpoint_frozen_blocks':checkpoint_checks,'training_order_hashes_and_final_rng_verified':training_order_checks,'descriptor_max_abs_discrepancy':descriptor_max_error,'limitations':['Local Gaussian regeneration differs from original platform; separate native-noise-check.json records same-environment replay.','No model likelihood or feature extraction rerun; NLL arithmetic and outer chart verified from saved components.','Float32 CUDA sigmoid compared to float64 CPU logistic with explicit tolerance, not byte-equality.','Checkpoint frozen analysis/root blocks verified exactly; optimizer trajectory not replayed.','Reused-repair development study; no confirmatory significance.']}
with a.output.open('x') as f:json.dump(record,f,indent=2)
print(json.dumps({k:v for k,v in record.items() if k!='results'}),flush=True)
