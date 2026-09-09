import hashlib,json,subprocess
from pathlib import Path
import numpy as np
import torch
root=Path(__file__).resolve().parent
p=root/'full/20260909-innovation-response'
repo=root.parent/'diffusion-representation'
sha=lambda b:hashlib.sha256(b).hexdigest()
status=json.loads((p/'status.json').read_text());identity=json.loads((p/'source_identity.json').read_text())
checks={}
for rel,h in status['payload_sha256'].items():
 assert sha((p/rel).read_bytes())==h,rel
checks['payload_count']=len(status['payload_sha256'])
for rel,h in identity['sha256'].items():
 b=(p/'sources'/rel).read_bytes();assert sha(b)==h,rel
 assert b==subprocess.check_output(['git','show',f"{identity['commit']}:{rel}"],cwd=repo),rel
checks['source_count']=len(identity['sha256']);checks['commit']=identity['commit']
checks['failure']=json.loads((p/'failure.json').read_text())
checks['stage_reports']={f.stem:{k:v for k,v in json.loads(f.read_text()).items() if k not in ['losses']} for f in sorted(p.glob('seed_*/*_progress.json'))}
q=p/'seed_78201';report=json.loads((q/'RQS_prefix_progress.json').read_text())
ck=torch.load(q/'RQS_prefix_failed.pt',map_location='cpu',weights_only=True)
ids=np.load(p/'record_ids.npz')['fit'];rng=np.random.default_rng(78201+30);h=hashlib.sha256()
for draw in range(report['updates']+1):
 index=rng.integers(0,len(ids),size=32);h.update(np.asarray(ids[index],dtype='<i8').tobytes())
assert rng.bit_generator.state==ck['rng'],'saved RNG mismatch'
assert h.hexdigest()==report['record_order_sha256'],'record draw order mismatch'
np.savez(root/'failed-minibatch-identities.npz',fit_subset_indices=index,canonical_record_ids=ids[index])
checks['reconstructed_draw']={'successful_updates':report['updates'],'failed_draw_one_based':report['updates']+1,'batch_size':32,'fit_subset_indices':index.tolist(),'canonical_record_ids':ids[index].tolist(),'rng_exact':True,'record_order_hash_exact':True,'post_draw_rng':rng.bit_generator.state,'checkpoint_architecture':ck['architecture'],'optimizer_steps':sorted(set(int(v['step'].item()) for v in ck['optimizer']['state'].values()))}
checks['all_fits_frozen']=(p/'ALL_FITS_FROZEN.json').exists()
checks['evaluation_payloads']=[str(f.relative_to(p)) for f in p.rglob('*') if f.is_file() and any(k in f.name for k in ['generated','features','quality','generation_progress','numerical'])]
checks['note']='Strict loader loaded selected repair inputs during canonical setup; no repair evaluation path reached. Failed draw is reconstructed after all successful updates; no optimizer step on failure inside forward. No model or dataset rerun.'
checks['file_inventory']={str(f.relative_to(root/'full')):{'sha256':sha(f.read_bytes()),'bytes':f.stat().st_size} for f in sorted((root/'full').rglob('*')) if f.is_file()}
(root/'machinecheck.json').write_text(json.dumps(checks,indent=2)+'\n')
print(json.dumps({k:checks[k] for k in ['payload_count','source_count','commit','reconstructed_draw','all_fits_frozen','evaluation_payloads']},indent=2))
