import hashlib,json,shlex,subprocess,time
from pathlib import Path
repo=Path('work/diffusion-representation');out=Path('work/psc-response-failure');rev='08cbe585d570a9405be833a27c00d04eaecba327'
names=['qalt/experiments/response_failure_diagnostic/'+n for n in ['run.py','run.slurm','run_gpu.slurm','PROTOCOL.md']]+['qalt/tests/test_response_failure_diagnostic.py','qalt/src/qalt/reflected_dense_spline.py']
expected={n:hashlib.sha256(subprocess.check_output(['git','show',rev+':'+n],cwd=repo)).hexdigest() for n in names}
remote=r'''import subprocess,json,hashlib,os
from pathlib import Path
base=Path('/ocean/projects/mth250006p/ywang26');original=base/'diffusion-representation';tree=base/'diffusion-response-diagnostic-gpu-20260909';output=base/'diffusion-results/20260909-response-diagnostic-gpu';rev=REV;expected=EXPECTED
assert not tree.exists() and not output.exists()
before=subprocess.check_output(['git','-C',str(original),'status','--porcelain=v1','-z']);head=subprocess.check_output(['git','-C',str(original),'rev-parse','HEAD'])
subprocess.run(['git','-C',str(original),'fetch','origin','agent/observation-transport-audit-20260908'],check=True)
subprocess.run(['git','-C',str(original),'worktree','add','--quiet','--detach',str(tree),rev],check=True)
assert subprocess.check_output(['git','-C',str(original),'status','--porcelain=v1','-z'])==before
assert subprocess.check_output(['git','-C',str(original),'rev-parse','HEAD'])==head
assert subprocess.check_output(['git','-C',str(tree),'rev-parse','HEAD']).decode().strip()==rev
for name,h in expected.items():
 raw=(tree/name).read_bytes();assert hashlib.sha256(raw).hexdigest()==h and raw==subprocess.check_output(['git','-C',str(tree),'show',rev+':'+name])
command=['sbatch','--parsable','--job-name=response-diagnostic-gpu','--chdir',str(tree),'--export=ALL,SOURCE_COMMIT='+rev+',OUTPUT_DIR='+str(output),'--output',str(base/'diffusion-results/response-diagnostic-gpu-%j.out'),'qalt/experiments/response_failure_diagnostic/run_gpu.slurm']
print('External frozen closure verified and original dirty status/HEAD unchanged',flush=True)
r=subprocess.run(command,cwd=tree,capture_output=True,text=True)
record={'source':rev,'source_sha256':expected,'command':command,'submission_attempts':1,'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr,'original_head':head.decode().strip(),'original_status_sha256':hashlib.sha256(before).hexdigest(),'original_dirty_unchanged':True,'output':str(output),'checkout':str(tree)}
if r.returncode==0:record['job_id']=r.stdout.strip().split(';')[0]
print(json.dumps(record),flush=True)
if r.returncode:raise SystemExit(r.returncode)
print(subprocess.check_output(['squeue','-j',record['job_id'],'-h','-o','%i|%T|%R|%M']).decode(),flush=True)
'''.replace('REV',repr(rev)).replace('EXPECTED',repr(expected))
(out/'launch-gpu-diagnostic.remote.py').open('x').write(remote)
t=time.monotonic()
try:
 r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=20','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=3','bridges2-codex','/ocean/projects/mth250006p/ywang26/pytorch/bin/python -u -c '+shlex.quote(remote)],capture_output=True,text=True,timeout=600);d={'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
except subprocess.TimeoutExpired as e:d={'timeout_uncertain_no_retry':True,'stdout':str(e.stdout),'stderr':str(e.stderr)}
d['elapsed']=time.monotonic()-t;(out/'launch-gpu-diagnostic.json').open('x').write(json.dumps(d,indent=2));print(json.dumps(d,indent=2))
