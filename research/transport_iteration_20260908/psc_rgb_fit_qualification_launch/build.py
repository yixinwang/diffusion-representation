from pathlib import Path
import ast,subprocess,json,hashlib,shlex
out=Path(__file__).resolve().parent;repo=out.parents[0]/'diffusion-representation';rev=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
tree=ast.parse((repo/'qalt/experiments/rgb_codec_fit_qualification/run.py').read_text());files=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='FILES' for t in n.targets));pins={}
for rel in files:
 raw=(repo/rel).read_bytes();assert raw==subprocess.check_output(['git','show',rev+':'+rel],cwd=repo);pins[rel]=hashlib.sha256(raw).hexdigest()
base='/ocean/projects/mth260022p/ywang26';dest=base+'/diffusion-rgb-fit-qualification-20260909-'+rev[:8];staging=base+'/rgb-fit-qualification-launch-20260909-'+rev[:8];result=base+'/diffusion-results/20260909-rgb-fit-qualification-'+rev[:8]
config={'source_commit':rev,'source_sha256':pins,'checkout':dest,'staging':staging,'result_root':result,'new_project':base,'original_read_only_checkout':'/ocean/projects/mth250006p/ywang26/diffusion-representation','remote':'https://github.com/yixinwang/diffusion-representation.git'}
(out/'config.json').write_text(json.dumps(config,indent=2)+'\n')
script='''import os,json,subprocess,hashlib,traceback,time
from pathlib import Path
CONFIG=CONFIG_VALUE
stage=Path(CONFIG['staging']);stage.mkdir(parents=True,exist_ok=False)
started=time.time();env={**os.environ,'GIT_TERMINAL_PROMPT':'0','GIT_OPTIONAL_LOCKS':'0','TMPDIR':str(stage/'tmp'),'XDG_CACHE_HOME':str(stage/'cache')}
Path(env['TMPDIR']).mkdir();Path(env['XDG_CACHE_HOME']).mkdir()
def call(args,**kw):return subprocess.check_output(args,env=env,timeout=300,**kw)
original=CONFIG['original_read_only_checkout'];before=None;head=None
try:
 head=call(['git','-C',original,'rev-parse','HEAD']);before=call(['git','--no-optional-locks','-C',original,'status','--porcelain=v1','-z'])
 repo=Path(CONFIG['checkout']);assert not repo.exists()
 with (stage/'clone.log').open('wb') as log:
  subprocess.run(['git','clone','--depth=1','--filter=blob:none','--no-checkout',CONFIG['remote'],str(repo)],env=env,check=True,timeout=300,stdout=log,stderr=subprocess.STDOUT)
  subprocess.run(['git','-C',str(repo),'fetch','--depth=1','origin',CONFIG['source_commit']],env=env,check=True,timeout=300,stdout=log,stderr=subprocess.STDOUT)
  subprocess.run(['git','-C',str(repo),'sparse-checkout','set','qalt'],env=env,check=True,timeout=300,stdout=log,stderr=subprocess.STDOUT)
  subprocess.run(['git','-C',str(repo),'checkout','--detach',CONFIG['source_commit']],env=env,check=True,timeout=300,stdout=log,stderr=subprocess.STDOUT)
 assert call(['git','-C',str(repo),'rev-parse','HEAD']).decode().strip()==CONFIG['source_commit']
 for rel,sha in CONFIG['source_sha256'].items():
  raw=(repo/rel).read_bytes();assert hashlib.sha256(raw).hexdigest()==sha
  assert raw==call(['git','-C',str(repo),'show',CONFIG['source_commit']+':'+rel])
 assert call(['git','-C',original,'rev-parse','HEAD'])==head and call(['git','--no-optional-locks','-C',original,'status','--porcelain=v1','-z'])==before
 result={'status':'prepared_no_allocation','config':CONFIG,'original_head':head.decode().strip(),'original_status_sha256':hashlib.sha256(before).hexdigest(),'original_unchanged':True,'elapsed_seconds':time.time()-started}
except BaseException as exc:
 result={'status':'preparation_failed_no_allocation','config':CONFIG,'error':repr(exc),'traceback':traceback.format_exc(),'elapsed_seconds':time.time()-started}
 if before is not None:
  result['original_head']=head.decode().strip();result['original_status_sha256']=hashlib.sha256(before).hexdigest()
  try:result['original_unchanged']=call(['git','-C',original,'rev-parse','HEAD'])==head and call(['git','--no-optional-locks','-C',original,'status','--porcelain=v1','-z'])==before
  except BaseException as check:result['original_verification_error']=repr(check)
(stage/'preparation.json').write_text(json.dumps(result,indent=2)+'\\n');print(json.dumps(result),flush=True)
if result['status']!='prepared_no_allocation':raise SystemExit(1)
'''.replace('CONFIG_VALUE',repr(config))
(out/'prepare.remote.py').write_text(script)
# Prepared separately; execute once after source and storage verification.
launch='''#!/usr/bin/env bash
set -euo pipefail
STAGE=STAGE_VALUE
REPO=REPO_VALUE
RESULT=RESULT_VALUE
REV=REV_VALUE
export SOURCE_COMMIT="$REV" RESULT_ROOT="$RESULT"
export PYTHONPYCACHEPREFIX="$STAGE/pycache" XDG_CACHE_HOME="$STAGE/cache" TMPDIR="$STAGE/tmp"
export TORCH_HOME="$STAGE/torch-cache" CUDA_CACHE_PATH="$STAGE/cuda-cache"
export HF_HOME="$STAGE/hf-cache" TRITON_CACHE_DIR="$STAGE/triton-cache"
test ! -e "$RESULT"
cd "$REPO"
# One allocation, one worker, immediate release when the worker exits. No retries.
exec salloc --account=cis260243p --partition=GPU-shared --qos=gpuinteract --gres=gpu:v100-32:1 --exclude=v005 -N1 -n1 -c4 --mem=16000M -t01:00:00 --job-name=rgb-codec-fit-qualification \\
 srun --export=ALL --output="$STAGE/worker-%j.out" --error="$STAGE/worker-%j.err" bash qalt/experiments/rgb_codec_fit_qualification/run.sh
'''
for key,value in [('STAGE_VALUE',staging),('REPO_VALUE',dest),('RESULT_VALUE',result),('REV_VALUE',rev)]:launch=launch.replace(key,shlex.quote(value))
(out/'submit-once.sh').write_text(launch)
print(json.dumps(config,indent=2))
