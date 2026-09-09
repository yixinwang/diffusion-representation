import hashlib,json,subprocess,tarfile
from pathlib import Path
import numpy as np
import torch
import argparse
ap=argparse.ArgumentParser();ap.add_argument('--workspace',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args();root=args.workspace.resolve()
if not (root/'cpu-grad-full').exists():
 with tarfile.open(root/'cpu-grad-full.tar') as tar:tar.extractall(root/'cpu-grad-full',filter='data')
p=root/'cpu-grad-full/20260909-response-cpu-grad';status=json.loads((p/'status.json').read_text());source=json.loads((p/'source_identity.json').read_text());repo=root.parent/'diffusion-representation'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for rel,h in status['payload_sha256'].items():assert sha(p/rel)==h,rel
for rel,h in source['diagnostic_sha256'].items():
 f=p/'diagnostic_sources'/rel;assert sha(f)==h;assert f.read_bytes()==subprocess.check_output(['git','show',f"{source['diagnostic_commit']}:{rel}"],cwd=repo)
original=json.loads((root/'machinecheck.json').read_text());assert source['reference_commit']==original['commit']
for rel,h in source['reference_sha256'].items():assert sha(root/'full/20260909-innovation-response/sources'/rel)==h
ids=np.load(p/'batch_identity.npz');assert ids['fit_subset_indices'].tolist()==original['reconstructed_draw']['fit_subset_indices'];assert ids['canonical_record_ids'].tolist()==original['reconstructed_draw']['canonical_record_ids']
report={'payload_count':len(status['payload_sha256']),'diagnostic_source_count':len(source['diagnostic_sha256']),'reference_source_count':len(source['reference_sha256']),'decoder_outcome':status['decoder_outcome'],'source':source['diagnostic_commit'],'layers':[]}
for i in range(4):
 d=torch.load(p/f'layer_{i}_kernel_locals.pt',map_location='cpu',weights_only=True);c=torch.load(p/f'layer_{i}_conditioner.pt',map_location='cpu',weights_only=True);inside=d['inside']
 independent={'good_bins':(torch.isfinite(d['widths'])&(d['widths']>0)&torch.isfinite(d['heights'])&(d['heights']>0)).all(-1),'good_disc':torch.isfinite(d['disc'])&(d['disc']>=0),'good_theta':torch.isfinite(d['theta'])&(d['theta']>=0)&(d['theta']<=1),'good_terms':torch.isfinite(d['delta'])&(d['delta']>0)&torch.isfinite(d['denominator'])&(d['denominator']>0)&torch.isfinite(d['numerator'])&(d['numerator']>0)}
 counts={}
 for name,v in independent.items():
  assert torch.equal(v,d[name]),(i,name);counts[name]=int((inside&~v).sum())
 # Original uncontained denominator can be independently rebuilt from b/root/a.
 den=torch.where(d['b']<0,2*d['a'],d['b']+d['root']);v=torch.isfinite(den)&(den!=0);assert torch.equal(v,d['good_den']);counts['good_den']=int((inside&~v).sum())
 assert all(n==0 for n in counts.values())
 mask=c['mask'].expand_as(inside);theta=d['theta'];positions=(inside&(theta<1)).nonzero();gap=(1-theta)[inside&(theta<1)];j=int(gap.argmin());coord=positions[j].tolist()
 report['layers'].append({'index':i,'invalid_counts':counts,'inside_count':int(inside.sum()),'exact_theta_one_count':int((inside&(theta==1)).sum()),'theta_min':float(theta[inside].min()),'theta_max':float(theta[inside].max()),'closest_below_one':{'coord':coord,'gap':float(gap[j]),'coupling_inactive':bool(mask[tuple(coord)])}})
batch=torch.load(p/'actual_batch.pt',map_location='cpu',weights_only=True)
rt=torch.load(p/'candidate_roundtrip.pt',map_location='cpu',weights_only=True)
encoded=torch.load(p/'candidate_encoded.pt',map_location='cpu',weights_only=True)
gradients=torch.load(p/'candidate_gradients.pt',map_location='cpu',weights_only=True)
assert torch.equal(rt['z'],encoded['z']) and torch.equal(rt['ld'],encoded['ld'])
ref=json.loads((p/'candidate_comparison.json').read_text())
rterr=float((rt['recovered_residual']-batch['residual']).abs().max());lderr=float((rt['ld']+rt['inverse_ld']).abs().max())
assert rterr==ref['roundtrip_max_abs'] and lderr==ref['logdet_cancellation_max_abs']
assert rterr<=1e-3 and lderr<=1e-2
assert len(gradients)==48 and all(g is not None and bool(torch.isfinite(g).all()) for g in gradients.values())
ck=torch.load(root/'full/20260909-innovation-response/seed_78201/RQS_prefix_failed.pt',map_location='cpu',weights_only=True)
for name,g in gradients.items():assert g.shape==ck['model']['residual_decoder.'+name].shape
old=root/'cpu-full/20260909-response-diagnostic-cpu';oldbatch=torch.load(old/'actual_batch.pt',map_location='cpu',weights_only=True)
mode_compare={k:{'exact':torch.equal(v,oldbatch[k]),'maxabs':float((v-oldbatch[k]).abs().max())} for k,v in batch.items()}
oldencoded=torch.load(old/'encoded.pt',map_location='cpu',weights_only=True);newencoded=torch.load(p/'encoded.pt',map_location='cpu',weights_only=True)
mode_compare.update({'encoded_'+k:{'exact':torch.equal(v,oldencoded[k]),'maxabs':float((v-oldencoded[k]).abs().max())} for k,v in newencoded.items()})
report['candidate']={'roundtrip_max_abs':rterr,'logdet_cancellation_max_abs':lderr,'finite48_saved_gradients':True,'gradient_shapes_match_failed_checkpoint':True,'parameter_gradient_total_elements':sum(v.numel() for v in gradients.values()),'nonzero_gradient_tensors':sum(bool((v!=0).any()) for v in gradients.values()),'live_state_unchanged_guard_reported':ref['state_unchanged'],'gradient_recomputation_performed':False}
report['cpu_no_grad_comparison']=mode_compare
report['limits']='CPU gradient-enabled capture is valid but does not reproduce GPU training-forward exception. All saved candidate gradients finite and shapes verified; gradients are not independently differentiated anew. State unchanged is the authenticated runner live fingerprint guard, not a separately saved before/after model pair. No root cause or repair quality claim.' 
args.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
