import subprocess,json,hashlib,os
from pathlib import Path
base=Path('/ocean/projects/mth250006p/ywang26');original=base/'diffusion-representation';tree=base/'diffusion-response-diagnostic-gpu-20260909';output=base/'diffusion-results/20260909-response-diagnostic-gpu';rev='08cbe585d570a9405be833a27c00d04eaecba327';expected={'qalt/experiments/response_failure_diagnostic/run.py': '750d8bbed33a3660e8bd00887ddca9a52c9705d7413587d44274f25d48125643', 'qalt/experiments/response_failure_diagnostic/run.slurm': '8d30054fa1a416d7a37d179087e5dcc2aabed1f5abd59f2fa559ecd9f2750d53', 'qalt/experiments/response_failure_diagnostic/run_gpu.slurm': '0e3dfbea08c9df272816d9f5098b4af711af96b446b7415356371785320144a7', 'qalt/experiments/response_failure_diagnostic/PROTOCOL.md': '8774c8a6033d422737a31fcb19e88c5234e6a336928d7fd5fc8465ce138866ce', 'qalt/tests/test_response_failure_diagnostic.py': '94ef7baa74fe5606bcfee3431cecbfd841ccde62523404d583a89caf0b379cd2', 'qalt/src/qalt/reflected_dense_spline.py': 'd6d387ae1651d57d1fab352639389407995ca10eeed328fcfc69dd486abb40d7'}
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
