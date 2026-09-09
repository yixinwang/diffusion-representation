import os,json,subprocess,hashlib,time,sys
from pathlib import Path
repo=Path('/ocean/projects/mth250006p/ywang26/diffusion-v2-interactive-preflight-20260909');out=Path('/ocean/projects/mth250006p/ywang26/diffusion-results/20260909-response-v2-interactive-preflight');identity=json.loads((out/'source_identity.json').read_text())
job=os.environ['SLURM_JOB_ID'];start=time.monotonic()
(out/'allocation_job.json').write_text(json.dumps({'job_id':job,'host':os.uname().nodename,'started_epoch':time.time(),'source_commit':identity['commit']},indent=2));print('ACTUAL_ALLOCATION_JOB '+job,flush=True)
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo).decode().strip()==identity['commit']
for name,h in identity['sha256'].items():assert hashlib.sha256((repo/name).read_bytes()).hexdigest()==h
import torch
assert torch.__version__=='2.10.0+cu128',torch.__version__
tests=['test_innovation_response_pilot_v2.py','test_innovation_response.py','test_dense_global_conditional_spline.py','test_reflected_dense_spline.py']
runner=(repo/'qalt/experiments/innovation_response_pilot_v2/run.py').read_text()
assert all("str(ROOT/'qalt/tests/"+name+"')" in runner for name in tests)
command=[sys.executable,'-m','pytest','-q']+[str(repo/'qalt/tests'/name) for name in tests]
with (out/'pytest.stdout').open('x') as stdout,(out/'pytest.stderr').open('x') as stderr:r=subprocess.run(command,cwd=repo,stdout=stdout,stderr=stderr)
record={'status':'tests_completed','returncode':r.returncode,'job_id':job,'torch':torch.__version__,'command':command,'runtime_seconds':time.monotonic()-start,'canonical_data_loaded':False,'training_performed':False,'source_commit':identity['commit']};(out/'test_status.json').write_text(json.dumps(record,indent=2));print((out/'pytest.stdout').read_text(),flush=True);print(json.dumps(record),flush=True);raise SystemExit(r.returncode)
