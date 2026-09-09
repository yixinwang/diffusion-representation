import subprocess,json,hashlib,os
from pathlib import Path
base=Path('/ocean/projects/mth250006p/ywang26');original=base/'diffusion-representation';tree=base/'diffusion-response-diagnostic-20260909';output=base/'diffusion-results/20260909-response-diagnostic-cpu';rev='2b9ccd26f0d36634e0b88f4af02954fb250fb6a5';expected={'qalt/experiments/response_failure_diagnostic/run.py': '3d1e01b5493e370928c2cb83b106882a4883416a1eb4e66793512bc19f7f99de', 'qalt/experiments/response_failure_diagnostic/run.slurm': '8d30054fa1a416d7a37d179087e5dcc2aabed1f5abd59f2fa559ecd9f2750d53', 'qalt/experiments/response_failure_diagnostic/PROTOCOL.md': '8078e9c3b718154bb8097d1be77036c95577a5b92a0a890f6947d15698ea6685', 'qalt/tests/test_response_failure_diagnostic.py': '9206a5f7b1f7ad2b8b9f30f89b84e487046eaef8751d3231d3c6c2512b207ddd'}
assert not tree.exists() and not output.exists()
before=subprocess.check_output(['git','-C',str(original),'status','--porcelain=v1','-z']);head=subprocess.check_output(['git','-C',str(original),'rev-parse','HEAD'])
subprocess.run(['git','-C',str(original),'fetch','origin','agent/observation-transport-audit-20260908'],check=True)
subprocess.run(['git','-C',str(original),'worktree','add','--quiet','--detach',str(tree),rev],check=True)
assert subprocess.check_output(['git','-C',str(original),'status','--porcelain=v1','-z'])==before
assert subprocess.check_output(['git','-C',str(original),'rev-parse','HEAD'])==head
assert subprocess.check_output(['git','-C',str(tree),'rev-parse','HEAD']).decode().strip()==rev
for name,h in expected.items():
 raw=(tree/name).read_bytes();assert hashlib.sha256(raw).hexdigest()==h and raw==subprocess.check_output(['git','-C',str(tree),'show',rev+':'+name])
command=['sbatch','--parsable','--job-name=response-diagnostic-cpu','--chdir',str(tree),'--export=ALL,SOURCE_COMMIT='+rev+',OUTPUT_DIR='+str(output),'--output',str(base/'diffusion-results/response-diagnostic-cpu-%j.out'),'qalt/experiments/response_failure_diagnostic/run.slurm']
print('External frozen closure verified and original dirty status/HEAD unchanged',flush=True)
r=subprocess.run(command,cwd=tree,capture_output=True,text=True)
record={'source':rev,'source_sha256':expected,'command':command,'submission_attempts':1,'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr,'original_head':head.decode().strip(),'original_status_sha256':hashlib.sha256(before).hexdigest(),'original_dirty_unchanged':True,'output':str(output),'checkout':str(tree)}
if r.returncode==0:record['job_id']=r.stdout.strip().split(';')[0]
print(json.dumps(record),flush=True)
if r.returncode:raise SystemExit(r.returncode)
print(subprocess.check_output(['squeue','-j',record['job_id'],'-h','-o','%i|%T|%R|%M']).decode(),flush=True)
