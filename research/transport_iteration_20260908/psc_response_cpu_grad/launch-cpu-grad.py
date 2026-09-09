import subprocess,json,hashlib,shlex,time
from pathlib import Path
rev='08cbe585d570a9405be833a27c00d04eaecba327';local=Path('work/diffusion-representation');out=Path('work/psc-response-failure')
names=['qalt/experiments/response_failure_diagnostic/'+n for n in ['run.py','run.slurm','run_gpu.slurm','PROTOCOL.md']]+['qalt/tests/test_response_failure_diagnostic.py','qalt/src/qalt/reflected_dense_spline.py']
expected={n:hashlib.sha256(subprocess.check_output(['git','show',rev+':'+n],cwd=local)).hexdigest() for n in names}
remote=r'''import subprocess,json,hashlib
from pathlib import Path
base=Path('/ocean/projects/mth250006p/ywang26');repo=base/'diffusion-response-diagnostic-gpu-20260909';out=base/'diffusion-results/20260909-response-cpu-grad';rev=REV;expected=EXPECTED
assert not out.exists()
assert subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD']).decode().strip()==rev
for name,h in expected.items():
 raw=(repo/name).read_bytes();assert hashlib.sha256(raw).hexdigest()==h and raw==subprocess.check_output(['git','-C',str(repo),'show',rev+':'+name])
command=['sbatch','--parsable','--job-name=response-cpu-grad','--chdir',str(repo),'--export=ALL,SOURCE_COMMIT='+rev+',OUTPUT_DIR='+str(out),'--output',str(base/'diffusion-results/response-cpu-grad-%j.out'),'qalt/experiments/response_failure_diagnostic/run.slurm']
r=subprocess.run(command,cwd=repo,capture_output=True,text=True)
d={'source':rev,'source_sha256':expected,'command':command,'submission_attempts':1,'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr,'output':str(out),'checkout':str(repo),'kind':'CPU gradient-enabled diagnostic control; no GPU attribution or quality evaluation'}
if r.returncode==0:d['job_id']=r.stdout.strip().split(';')[0]
print(json.dumps(d),flush=True)
if r.returncode:raise SystemExit(r.returncode)
print(subprocess.check_output(['squeue','-j',d['job_id'],'-h','-o','%i|%T|%R|%M']).decode(),flush=True)
'''.replace('REV',repr(rev)).replace('EXPECTED',repr(expected))
(out/'launch-cpu-grad.remote.py').open('x').write(remote);t=time.monotonic()
try:
 r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=20','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=3','bridges2-codex','/ocean/projects/mth250006p/ywang26/pytorch/bin/python -u -c '+shlex.quote(remote)],capture_output=True,text=True,timeout=600);d={'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
except subprocess.TimeoutExpired as e:d={'timeout_uncertain_no_retry':True,'stdout':str(e.stdout),'stderr':str(e.stderr)}
d['elapsed']=time.monotonic()-t;(out/'launch-cpu-grad.json').open('x').write(json.dumps(d,indent=2));print(json.dumps(d,indent=2))
