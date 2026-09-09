"""Independent artifact/metadata/numerical audit; no decode or production imports."""
from pathlib import Path
from fractions import Fraction
import json,hashlib,subprocess
import numpy as np
import PIL
from PIL import Image
base=Path(__file__).resolve().parent;r=base/'retrieved/20260909-video-index';d=r/'decode'
errors=[]
def check(name,okay):
    if not okay:errors.append(name)
def sha(raw):return hashlib.sha256(raw).hexdigest()
status=json.loads((d/'status.json').read_text());launch=json.loads((r/'launcher_status.json').read_text())
check('completed',status['status']=='complete' and launch['status']=='complete')
manifest=json.loads((d/'ARTIFACTS.json').read_text())
for name,record in manifest.items():
    raw=(d/name).read_bytes();check('hash:'+name,sha(raw)==record['sha256'] and len(raw)==record['bytes'])
check('file_enumeration',set(manifest)=={p.name for p in d.iterdir() if p.is_file()}-{'ARTIFACTS.json','status.json'})
ledger_raw=(d/'authenticated_frames.jsonl').read_bytes();check('ledger_sha',sha(ledger_raw)=='9344c98746b1b76c45357731936e0a2c111e575a6f8da874d0789228fef9a5d0')
ledger=[json.loads(x) for x in ledger_raw.decode().splitlines()];matched=[json.loads(x) for x in (d/'matched_frames.jsonl').read_text().splitlines()]
check('all216frames',len(ledger)==len(matched)==216)
for i,(old,new) in enumerate(zip(ledger,matched)):check('metadata:'+str(i),all(old[k]==v for k,v in new.items()) and new['index']==i)
check('member_hash',status['member_sha256']=='699175c50544283f3b8537387403ff5b82958e4b18e590611129013131d1601a')
check('dependency_files',launch['dependency_files_verified']==264)
repo=base.parent/'diffusion-representation';prefix='research/transport_iteration_20260908/';rev='01a3067a245e686363c8f16724a825f2dfa86577'
for name,path in {'run_index_decode.py':'video_decode_prerequisite/run_index_decode.py','frozen_helpers.py':'video_decode_prerequisite/helpers.py','frozen_reader.py':'video_archive_prerequisite/archive_reader.py'}.items():
    check('source:'+name,(d/name).read_bytes()==subprocess.check_output(['git','-C',str(repo),'show',rev+':'+prefix+path]))
selection=json.loads((d/'selection.json').read_text());expected=[]
for k in range(8):
    t=Fraction(k*215,7);expected.append(min(range(216),key=lambda i:(abs(i-t),i)))
check('indices',[x['index'] for x in selection]==expected)
check('saved_indices',status['saved_frames']==expected)
checks=[]
for i in expected:
    stem=f'frame_{i:06d}_';load=lambda name:np.load(d/(stem+name+'.npy'),allow_pickle=False)
    original=load('original_rgb');pixels=load('processed_rgb');bits=load('uint32_noise');cube=load('unit_float64');logits=load('logit_float64');logits32=load('logit_float32')
    meta=json.loads((d/f'frame_{i:06d}.json').read_text())
    check('original_shape:'+str(i),original.dtype==np.uint8 and original.shape==(240,320,3))
    check('processed_shape:'+str(i),pixels.dtype==np.uint8 and pixels.shape==(64,64,3))
    h,w=original.shape[:2];short=min(h,w);nh=(2*h*64+short)//(2*short);nw=(2*w*64+short)//(2*short);top=(nh-64)//2;left=(nw-64)//2
    resize=np.asarray(Image.fromarray(original).resize((nw,nh),Image.Resampling.BILINEAR,reducing_gap=None).crop((left,top,left+64,top+64)))
    resize_equal=np.array_equal(resize,pixels);check('independent_resize:'+str(i),resize_equal)
    key={'schema':1,'record':'UCF101_subset/train/ApplyEyeMakeup/v_ApplyEyeMakeup_g02_c03.avi','frame_index':i,'role':'first-train-clip-feasibility-v1','algorithm':'SHA256-big-endian-seed-PCG64-uint32-midpoint-v1'}
    seed=hashlib.sha256(json.dumps(key,sort_keys=True,separators=(',',':')).encode()).digest()
    expected_bits=np.random.Generator(np.random.PCG64(int.from_bytes(seed,'big'))).integers(0,2**32,size=pixels.shape,dtype=np.uint32)
    check('noise:'+str(i),bits.dtype==np.uint32 and np.array_equal(bits,expected_bits))
    p=(pixels.astype(np.float64)+(bits.astype(np.float64)+.5)/2**32)/256
    check('cube:'+str(i),cube.dtype==np.float64 and np.array_equal(cube,p) and np.all((p>0)&(p<1)))
    logit=np.log(p/(1-p));err=float(np.max(abs(logit-logits)))
    check('logits64:'+str(i),logits.dtype==np.float64 and err<1e-13)
    check('logits32:'+str(i),logits32.dtype==np.float32 and np.array_equal(logits32,logits.astype(np.float32)) and np.isfinite(logits32).all())
    ld=float(np.log(1/(p*(1-p))).sum());outer=ld-p.size*np.log(256.)
    ld_err=abs(ld-meta['dequantization']['cube_to_logit_logdet_float64']);outer_err=abs(outer-meta['dequantization']['outer_logdet_float64'])
    check('logdet:'+str(i),max(ld_err,outer_err)<1e-9)
    check('frame_metadata:'+str(i),meta['exposed_metadata']==matched[i])
    check('seed_provenance:'+str(i),meta['dequantization']['key']==key and meta['dequantization']['seed_sha256']==seed.hex())
    checks.append({'index':i,'resize_exact_equal':resize_equal,'max_logit64_error':err,'cube_ld_error':ld_err,'outer_ld_error':outer_err,'processed_min':int(pixels.min()),'processed_max':int(pixels.max())})
result={'status':'artifact_and_numerical_checks_pass' if not errors else 'failed','errors':errors,'job':'45580328','matched_frames':len(matched),'indices':expected,'per_frame':checks,'audit_numpy':np.__version__,'audit_pillow':PIL.__version__,'qualification':'Original RGB was not independently decoded; hash/source/member/metadata provenance checked. Resize reproduced using local Pillow version; exact observed equality is not general version equivalence. No physical timing or model quality claim.','payload_sha256':{str(p.relative_to(r)):sha(p.read_bytes()) for p in r.rglob('*') if p.is_file()}}
(base/'audit.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='payload_sha256'},indent=2))
