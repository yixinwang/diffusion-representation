"""Exact first16 saved generated rows of all21 arms, no selection or clipping."""
from pathlib import Path
import numpy as np,json,hashlib,platform
root=Path('/ocean/projects/mth260022p/ywang26/diffusion-results/20260909-innovation-response-v2-attempt3')
out=Path('/ocean/projects/mth260022p/ywang26/diffusion-results/20260909-response-v2-first16');out.mkdir(exist_ok=False)
status_raw=(root/'status.json').read_bytes();assert hashlib.sha256(status_raw).hexdigest()=='d25b0dcee8475f61c4bf7a9ac412783ed5c9e5d30821e73e20d2653592c03722'
status=json.loads(status_raw);assert status['status']=='completed_development_only'
arrays={};origins={}
for seed in (78201,78202,78203):
 for arm in ('P_frozen','P_joint','I_frozen','I_joint','RQS_frozen','RQS_joint','S42'):
  rel=f'seed_{seed}/{arm}/generated.npy';x=np.load(root/rel,mmap_mode='r',allow_pickle=False)
  assert x.shape==(2000,3,32,32) and x.dtype==np.float64
  selected=np.asarray(x[:16],dtype=np.float32)
  assert np.isfinite(selected).all() and np.array_equal(selected.astype(np.float64),x[:16])
  arrays[f'{seed}_{arm}']=selected
  origins[f'{seed}_{arm}']={'native_file':rel,'native_file_sha256':status['payload_sha256'][rel],'rows':[0,16],'selected_float32_bytes_sha256':hashlib.sha256(selected.tobytes()).hexdigest()}
 rel=f'seed_{seed}/common_gaussian.npy';x=np.load(root/rel,mmap_mode='r',allow_pickle=False);assert x.shape==(2000,3072)
 arrays[f'{seed}_source']=np.array(x[:16],copy=True)
 origins[f'{seed}_source']={'native_file':rel,'native_file_sha256':status['payload_sha256'][rel],'rows':[0,16]}
np.savez(out/'first16.npz',**arrays)
raw=(out/'first16.npz').read_bytes();record={'source_status_sha256':hashlib.sha256(status_raw).hexdigest(),'numpy':np.__version__,'python':platform.python_version(),'selection':'First16 rows in stored order, every21arm; no clipping or filtering. float64saved pixels cast tofloat32 and exact backcast asserted. Not a quality selected gallery.','origins':origins,'output':{'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}}
(out/'provenance.json').write_text(json.dumps(record,indent=2));print(json.dumps({'output':str(out),'sha256':record['output']['sha256'],'bytes':len(raw)}))
