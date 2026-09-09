"""Independent completed-fit ledger/state audit; prepare before outputs are read."""
import argparse,hashlib,json,math
from pathlib import Path
import numpy as np
import torch

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def lr(t):
 return 2e-5+1.8e-4*t/30 if t<=30 else 2e-5+9e-5*(1+math.cos(math.pi*(min(t,600)-30)/570))
def run(root):
 status=json.loads((root/'status.json').read_text());assert status['status']=='completed_FIT_only_qualification'
 inventory=status['payload_sha256'];assert set(inventory)=={str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() and p.name!='status.json'}
 for name,value in inventory.items():assert sha(root/name)==value,name
 fit=status['fit'];assert fit['status']=='frozen' and fit['updates']>0
 assert fit['elapsed_seconds']>=600 and abs(fit['overrun_seconds']-(fit['elapsed_seconds']-600))<1e-8
 assert 0<=fit['optimizer_constructor_seconds']<status['full_standalone_wall_seconds']
 assert status['full_standalone_wall_seconds']>=fit['elapsed_seconds']+fit['optimizer_constructor_seconds']
 assert status['seed']==79201 and status['field_untrained'] is True
 assert status['parameter_counts']=={'full_model':6484387,'codec':3472915,'field':3011472}
 ids=np.load(root/'fit_ids.npy',allow_pickle=False);rng=np.random.default_rng(79202);previous=0.;completed=0;pending=None;max_lr_error=0.
 for line in (root/'fit_steps.jsonl').read_text().splitlines():
  row=json.loads(line)
  if row['event']=='draw':
   assert pending is None and row['step']==completed+1
   assert 0<=row['draw_stage_seconds']<600 and row['draw_stage_seconds']>=previous
   expected=rng.integers(0,4000,size=32);assert np.array_equal(expected,row['indices']) and np.array_equal(ids[expected],row['fit_ids'])
   error=abs(lr(row['draw_stage_seconds'])-row['lr']);assert error<1e-15;max_lr_error=max(max_lr_error,error);pending=row
  else:
   assert row['event']=='completed' and pending is not None and row['step']==pending['step']
   assert math.isfinite(row['loss']) and row['stage_seconds']>=pending['draw_stage_seconds']
   previous=row['stage_seconds'];completed+=1;pending=None
 assert pending is None and completed==fit['updates'] and previous<=fit['elapsed_seconds']
 assert [r['nominal_seconds'] for r in fit['checkpoints']]==[60.,150.,300.,450.,600.]
 lastupdates=0
 for record in fit['checkpoints']:
  assert record['actual_seconds']>=record['nominal_seconds'] and lastupdates<=record['updates']<=completed
  assert sha(root/record['path'])==record['sha256'];lastupdates=record['updates']
  ck=torch.load(root/record['path'],map_location='cpu',weights_only=False)
  assert ck['updates']==record['updates'] and ck['nominal_stage_seconds']==record['nominal_seconds'] and ck['stage']==0
  assert ck['stage_elapsed_seconds']==record['actual_seconds'];del ck
 initial=torch.load(root/'initial.pt',map_location='cpu',weights_only=False)
 final=torch.load(root/'fit_frozen.pt',map_location='cpu',weights_only=False)
 assert initial['updates']==0 and initial['stage']==0 and final['updates']==completed and final['stage']==1
 assert final['numpy_rng']==rng.bit_generator.state
 frozen=json.loads((root/'FIT_FROZEN.json').read_text());assert frozen['checkpoint_sha256']==sha(root/'fit_frozen.pt') and frozen['codec_frozen'] is True
 assert final['model'].keys()==initial['model'].keys()
 fieldkeys=[k for k in initial['model'] if k.startswith('field.')];assert fieldkeys
 assert all(torch.equal(initial['model'][k],final['model'][k]) for k in fieldkeys)
 changed=sum(not torch.equal(initial['model'][k],final['model'][k]) for k in initial['model'] if k.startswith('codec.'));assert changed>0
 normalized=torch.load(root/'normalized_state.pt',map_location='cpu',weights_only=True)
 assert all(torch.equal(final['model'][k],normalized[k]) for k in final['model'] if k.startswith(('codec.','field.')))
 cache=np.load(root/'fit_normalized_cache.npy',mmap_mode='r',allow_pickle=False);assert cache.shape==(4000,16,8,8) and cache.dtype==np.float32
 sums=np.zeros(16);squares=np.zeros(16)
 for first in range(0,4000,64):
  z=np.asarray(cache[first:first+64],dtype=np.float64);assert np.isfinite(z).all();sums+=z.sum(axis=(0,2,3));squares+=(z*z).sum(axis=(0,2,3))
 mean=sums/256000;variance=squares/256000-mean**2
 assert np.max(np.abs(mean))<3e-6 and np.max(np.abs(variance-1))<3e-6
 return {'status':'PASSED_COMPLETE_FIT_LEDGER_AND_STATE_AUDIT','completed_updates':completed,'fit_elapsed_seconds':fit['elapsed_seconds'],'overrun_seconds':fit['overrun_seconds'],'full_standalone_wall_seconds':status['full_standalone_wall_seconds'],'max_LR_error':max_lr_error,'unchanged_field_state_tensors':len(fieldkeys),'changed_codec_state_tensors':changed,'frozen_normalized_codec_field_exact':True,'cache_max_channel_mean_error':float(np.max(np.abs(mean))),'cache_max_channel_variance_error':float(np.max(np.abs(variance-1))),'scope':'Saved ledger replay with independent PCG64 draws and wall-LR calculation; complete checkpoint/source payload hashes; field unchanged and frozen codec exact through normalization. No model rerun or canonical image reload, no generalization or generation claim.'}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('output',type=Path);a=p.parse_args();result=run(a.root)
 with a.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
