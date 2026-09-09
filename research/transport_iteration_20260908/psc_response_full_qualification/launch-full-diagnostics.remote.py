import subprocess,json,hashlib
from pathlib import Path
base=Path('/ocean/projects/mth250006p/ywang26');original=base/'diffusion-representation';repo=base/'diffusion-response-full-20260909';rev='249ea6d6d166cb11b185ad03c0f62a564c86c335';expected={'qalt/experiments/response_failure_diagnostic/INDEPENDENT_REVIEW.md': '5d6386497ad1dbae552db4e07ab355c3f998dfad7fd5d059782005c7a0112b9c', 'qalt/experiments/response_failure_diagnostic/PROTOCOL.md': '5bcb6e66a9fa178a95f43af7e013c5d9c12f8f01d7522ab46a54ddc64f5fc77a', 'qalt/experiments/response_failure_diagnostic/full_model.py': '2fd5b4855e9677ada2c23dc317c5c97b3e5aa6ab2cbf3542a91912152e4e954a', 'qalt/experiments/response_failure_diagnostic/run.py': 'a023ae054c444fd4900845e4ce3851d9f87b4364b0720d40b747fd0d6c5e86a8', 'qalt/experiments/response_failure_diagnostic/run.slurm': '8d30054fa1a416d7a37d179087e5dcc2aabed1f5abd59f2fa559ecd9f2750d53', 'qalt/experiments/response_failure_diagnostic/run_full_cpu.slurm': 'd073ebe0d49355610b1a61c164f0a20f814f76eb45961a41b7b86a569113e8fc', 'qalt/experiments/response_failure_diagnostic/run_full_gpu.slurm': 'be2bb6385a6d2f24b3f8331de71c7a0b5c6414b7b19218b92d068223a0795315', 'qalt/experiments/response_failure_diagnostic/run_gpu.slurm': '0e3dfbea08c9df272816d9f5098b4af711af96b446b7415356371785320144a7', 'qalt/tests/test_response_failure_diagnostic.py': '37ebb3e78c95c2ea485ead033d6a52cea8ffb7dff06d3223f7e944cba3119eca', 'qalt/src/qalt/reflected_dense_spline.py': 'd6d387ae1651d57d1fab352639389407995ca10eeed328fcfc69dd486abb40d7'}
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
