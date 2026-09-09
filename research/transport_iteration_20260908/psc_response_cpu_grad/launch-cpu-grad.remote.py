import subprocess,json,hashlib
from pathlib import Path
base=Path('/ocean/projects/mth250006p/ywang26');repo=base/'diffusion-response-diagnostic-gpu-20260909';out=base/'diffusion-results/20260909-response-cpu-grad';rev='08cbe585d570a9405be833a27c00d04eaecba327';expected={'qalt/experiments/response_failure_diagnostic/run.py': '750d8bbed33a3660e8bd00887ddca9a52c9705d7413587d44274f25d48125643', 'qalt/experiments/response_failure_diagnostic/run.slurm': '8d30054fa1a416d7a37d179087e5dcc2aabed1f5abd59f2fa559ecd9f2750d53', 'qalt/experiments/response_failure_diagnostic/run_gpu.slurm': '0e3dfbea08c9df272816d9f5098b4af711af96b446b7415356371785320144a7', 'qalt/experiments/response_failure_diagnostic/PROTOCOL.md': '8774c8a6033d422737a31fcb19e88c5234e6a336928d7fd5fc8465ce138866ce', 'qalt/tests/test_response_failure_diagnostic.py': '94ef7baa74fe5606bcfee3431cecbfd841ccde62523404d583a89caf0b379cd2', 'qalt/src/qalt/reflected_dense_spline.py': 'd6d387ae1651d57d1fab352639389407995ca10eeed328fcfc69dd486abb40d7'}
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
