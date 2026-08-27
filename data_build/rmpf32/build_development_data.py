from __future__ import annotations
import hashlib, json, re, tarfile
from pathlib import Path
import cv2, numpy as np
from datasets import load_dataset
from huggingface_hub import HfApi, hf_hub_download

OUT=Path('build/rmpf32_out'); RAW=Path('build/rmpf32_raw'); OUT.mkdir(parents=True,exist_ok=True); RAW.mkdir(parents=True,exist_ok=True)
SEED=20260828

def sha(p:Path)->str:
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def ih(a): return hashlib.sha256(np.asarray(a,dtype='<i8').tobytes()).hexdigest()

api=HfApi(); cifar_rev=api.dataset_info('uoft-cs/cifar10').sha; ds=load_dataset('uoft-cs/cifar10',revision=cifar_rev)
def carr(split):
 part=ds[split]; return np.stack([np.asarray(im.convert('RGB'),np.uint8) for im in part['img']]),np.asarray(part['label'],np.int64)
x,y=carr('train'); xt,yt=carr('test')
# Regenerate MCQF-v1 exclusions exactly, then the fresh v2 development roles exactly.
old_rng=np.random.default_rng(20260827); used=set()
for c in range(10):
 idx=old_rng.permutation(np.flatnonzero(y==c)); used.update(idx[:1800].tolist())
old_test=set()
for c in range(10): old_test.update(np.random.default_rng(20260827+991+c).permutation(np.flatnonzero(yt==c))[:200].tolist())
counts={'train':1600,'validation':300,'fiber_fit':300,'development':300}; roles={k:[] for k in counts}
for c in range(10):
 avail=np.asarray(sorted(set(np.flatnonzero(y==c).tolist())-used),np.int64); perm=np.random.default_rng(SEED+17*c).permutation(avail); cur=0
 for k,n in counts.items(): roles[k]+=perm[cur:cur+n].tolist(); cur+=n
conf=[]
for c in range(10):
 avail=np.asarray(sorted(set(np.flatnonzero(yt==c).tolist())-old_test),np.int64); conf+=np.random.default_rng(SEED+991+c).permutation(avail)[:200].tolist()
conf=np.asarray(sorted(conf),np.int64)
payload={}
for k,v in roles.items():
 a=np.asarray(sorted(v),np.int64); payload[k+'_x']=x[a]; payload[k+'_y']=y[a]; payload[k+'_source_id']=a
np.savez_compressed(OUT/'cifar10_32_rmpf_development.npz',**payload)

repo='sayakpaul/ucf101-subset'; rev=api.dataset_info(repo).sha
archive=Path(hf_hub_download(repo,'UCF101_subset.tar.gz',repo_type='dataset',revision=rev,local_dir=RAW))
expected='e9fcc76af48d320be88c5265f2e0576ecd615956976f6ce4742fdf2b042b71eb'
if sha(archive)!=expected: raise RuntimeError(('ucf hash',sha(archive)))
ex=RAW/'ucf'; ex.mkdir(exist_ok=True)
with tarfile.open(archive,'r:*') as h:
 for m in h.getmembers():
  if not m.isfile() or not m.name.lower().endswith('.avi'): continue
  t=(ex/m.name).resolve()
  if ex.resolve() not in t.parents: raise RuntimeError(m.name)
  t.parent.mkdir(parents=True,exist_ok=True); src=h.extractfile(m)
  if src is not None: t.write_bytes(src.read())
pat=re.compile(r'^v_(?P<a>.+)_g(?P<g>\d+)_c(?P<c>\d+)\.avi$',re.I)
meta=[]
for p in sorted(ex.rglob('*.avi')):
 m=pat.match(p.name)
 if m: meta.append((p,m.group('a'),int(m.group('g')),int(m.group('c'))))
actions=sorted({a for _,a,_,_ in meta}); aid={a:i for i,a in enumerate(actions)}; groups=sorted({g for *_,g,_ in meta})
def role(g):
 z=groups.index(g)%10
 return 'confirmation' if z in (8,9) else 'validation' if z==7 else 'development' if z==6 else 'fiber_fit' if z==5 else 'train'
records={k:[] for k in ('train','validation','fiber_fit','development')}; confirm_sources=[]; source_videos=[]
for vid,(p,a,g,cid) in enumerate(meta):
 rr=role(g); source_videos.append({'video_id':vid,'file':p.name,'action':a,'group':g,'clip_id':cid,'role':rr})
 if rr=='confirmation': confirm_sources.append(vid); continue
 cap=cv2.VideoCapture(str(p)); frames=[]
 while True:
  ok,f=cap.read()
  if not ok: break
  frames.append(cv2.cvtColor(f,cv2.COLOR_BGR2RGB))
 cap.release()
 if len(frames)<12: continue
 arr=np.stack(frames); h,w=arr.shape[1:3]; s=min(h,w); t=(h-s)//2; l=(w-s)//2; arr=arr[:,t:t+s,l:l+s]
 arr=np.stack([cv2.resize(f,(32,32),interpolation=cv2.INTER_AREA) for f in arr]).astype(np.uint8)
 starts=[0,max(0,len(arr)-12)] if len(arr)>=24 else [0]
 for view,st in enumerate(sorted(set(starts))):
  ids=np.linspace(st,min(len(arr)-1,st+11),6).round().astype(int); records[rr].append((arr[ids],aid[a],vid,g,cid,view))
def pack(rows):
 return {'x':np.stack([q[0] for q in rows]),'y':np.asarray([q[1] for q in rows],np.int64),'video_id':np.asarray([q[2] for q in rows],np.int64),'group_id':np.asarray([q[3] for q in rows],np.int64),'clip_id':np.asarray([q[4] for q in rows],np.int64),'view_id':np.asarray([q[5] for q in rows],np.int64)}
vp={}
for k,v in records.items():
 for n,a in pack(v).items(): vp[k+'_'+n]=a
np.savez_compressed(OUT/'ucf101_6x32_rmpf_development.npz',**vp)
arts={p.name:{'sha256':sha(p),'bytes':p.stat().st_size} for p in OUT.glob('*.npz')}
manifest={'dataset_version':'rmpf-cifar32-ucf32-development-v1','builder_seed':SEED,'confirmation_pixels_written':False,
 'cifar10':{'source_repo':'uoft-cs/cifar10','source_revision':cifar_rev,'split_sizes':{k:len(v) for k,v in roles.items()},'split_source_id_hashes':{k:ih(np.asarray(sorted(v))) for k,v in roles.items()},'sealed_confirmation_count':len(conf),'sealed_confirmation_source_id_hash':ih(50000+conf),'v1_overlap':0},
 'ucf101':{'source_repo':repo,'source_revision':rev,'archive_sha256':expected,'shape':[6,32,32,3],'split_sizes':{k:len(v) for k,v in records.items()},'sealed_confirmation_source_videos':confirm_sources,'sealed_confirmation_source_video_hash':ih(confirm_sources),'source_videos':source_videos},'artifacts':arts}
(OUT/'dataset_manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
(OUT/'SHA256SUMS').write_text(''.join(f'{sha(p)}  {p.name}\n' for p in sorted(OUT.glob('*')) if p.name!='SHA256SUMS'))
print(json.dumps({k:v for k,v in manifest.items() if k!='ucf101'},indent=2))
print(json.dumps({'ucf101':{k:v for k,v in manifest['ucf101'].items() if k!='source_videos'}},indent=2))
