import subprocess,json,hashlib,shlex,time
from pathlib import Path
rev='249ea6d6d166cb11b185ad03c0f62a564c86c335';local=Path('work/diffusion-representation');out=Path('work/psc-response-failure')
names=subprocess.check_output(['git','ls-tree','-r','--name-only',rev,'qalt/experiments/response_failure_diagnostic'],cwd=local,text=True).splitlines()+['qalt/tests/test_response_failure_diagnostic.py','qalt/src/qalt/reflected_dense_spline.py']
expected={n:hashlib.sha256(subprocess.check_output(['git','show',rev+':'+n],cwd=local)).hexdigest() for n in names}
remote=r'''import subprocess,json,hashlib
from pathlib import Path
base=Path('/ocean/projects/mth250006p/ywang26');original=base/'diffusion-representation';repo=base/'diffusion-response-full-20260909';rev=REV;expected=EXPECTED
assert not repo.exists()
for kind in ['cpu','gpu']:assert not (base/('diffusion-results/20260909-response-full-'+kind)).exists()
before=subprocess.check_output(['git','-C',str(original),'status','--porcelain=v1','-z']);head=subprocess.check_output(['git','-C',str(original),'rev-parse','HEAD'])
assert head.decode().strip()=='4553e04f71c545e401409d18d82731589974760b'
subprocess.run(['git','-C',str(original),'fetch','origin','agent/observation-transport-audit-20260908'],check=True)
subprocess.run(['git','-C',str(original),'worktree','add','--quiet','--detach',str(repo),rev],check=True)
assert subprocess.check_output(['git','-C',str(original),'status','--porcelain=v1','-z'])==before
assert subprocess.check_output(['git','-C',str(original),'rev-parse','HEAD'])==head
assert subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD']).decode().strip()==rev
for name,h in expected.items():
 raw=(repo/name).read_bytes();assert hashlib.sha256(raw).hexdigest()==h and raw==subprocess.check_output(['git','-C',str(repo),'show',rev+':'+name])
print(json.dumps({'kind':'source_verification','source':rev,'source_sha256':expected,'original_dirty_unchanged':True,'original_head':head.decode().strip(),'original_status_sha256':hashlib.sha256(before).hexdigest(),'checkout':str(repo)}),flush=True)
for kind in ['cpu','gpu']:
 output=base/('diffusion-results/20260909-response-full-'+kind)
 command=['sbatch','--parsable','--job-name=response-full-'+kind,'--chdir',str(repo),'--export=ALL,SOURCE_COMMIT='+rev+',OUTPUT_DIR='+str(output),'--output',str(base/('diffusion-results/response-full-'+kind+'-%j.out'))]
 if kind=='gpu':command+=['--dependency=afterany:45612641']
 command+=['qalt/experiments/response_failure_diagnostic/run_full_'+kind+'.slurm']
 r=subprocess.run(command,cwd=repo,capture_output=True,text=True)
 d={'kind':kind,'source':rev,'command':command,'submission_attempts':1,'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr,'output':str(output)}
 if r.returncode==0:d['job_id']=r.stdout.strip().split(';')[0]
 print(json.dumps(d),flush=True)
 if r.returncode:raise SystemExit(r.returncode)
 print(subprocess.check_output(['squeue','-j',d['job_id'],'-h','-o','%i|%T|%R|%M']).decode(),flush=True)
'''.replace('REV',repr(rev)).replace('EXPECTED',repr(expected))
(out/'launch-full-diagnostics.remote.py').open('x').write(remote);t=time.monotonic()
try:
 r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=20','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=3','bridges2-codex','/ocean/projects/mth250006p/ywang26/pytorch/bin/python -u -c '+shlex.quote(remote)],capture_output=True,text=True,timeout=600);d={'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
except subprocess.TimeoutExpired as e:d={'timeout_uncertain_no_retry':True,'stdout':str(e.stdout),'stderr':str(e.stderr)}
d['elapsed']=time.monotonic()-t;(out/'launch-full-diagnostics.json').open('x').write(json.dumps(d,indent=2));print(json.dumps(d,indent=2))
