import subprocess,json,hashlib,shlex,time
from pathlib import Path
rev='3b8c0f7bef604a90ff329fd9608639527b67a972';local=Path('work/diffusion-representation');out=Path('work/psc-v2-interactive-preflight')
allnames=subprocess.check_output(['git','ls-tree','-r','--name-only',rev],cwd=local,text=True).splitlines()
def selected(n):
 p=Path(n)
 return n.startswith('qalt/src/qalt/') and p.suffix=='.py' or n.startswith('qalt/tests/') and len(p.parts)==3 and p.suffix=='.py' or n=='qalt/data/observed_manifest_v1.json' or str(p.parent) in ['qalt/experiments/innovation_response_pilot_v2','qalt/experiments/perceptual_appendix'] and p.suffix in ['.py','.md','.slurm','.json'] or n in ['qalt/experiments/observed_flow_pilot/run_shared.py','qalt/experiments/observed_flow_pilot/SHARED_PROTOCOL.md']
expected={n:hashlib.sha256(subprocess.check_output(['git','show',rev+':'+n],cwd=local)).hexdigest() for n in allnames if selected(n)}
remote=r'''import subprocess,json,hashlib,os,sys,time,selectors,re
from pathlib import Path
base=Path('/ocean/projects/mth250006p/ywang26');original=base/'diffusion-representation';repo=base/'diffusion-v2-interactive-preflight-20260909';out=base/'diffusion-results/20260909-response-v2-interactive-preflight';rev=REV;expected=EXPECTED
assert not repo.exists() and not out.exists()
before=subprocess.check_output(['git','-C',str(original),'status','--porcelain=v1','-z']);head=subprocess.check_output(['git','-C',str(original),'rev-parse','HEAD']);assert head.decode().strip()=='4553e04f71c545e401409d18d82731589974760b'
subprocess.run(['git','-C',str(original),'fetch','origin','agent/observation-transport-audit-20260908'],check=True)
subprocess.run(['git','-C',str(original),'worktree','add','--quiet','--detach',str(repo),rev],check=True)
assert subprocess.check_output(['git','-C',str(original),'status','--porcelain=v1','-z'])==before
assert subprocess.check_output(['git','-C',str(original),'rev-parse','HEAD'])==head
out.mkdir()
for name,h in expected.items():
 raw=(repo/name).read_bytes();assert hashlib.sha256(raw).hexdigest()==h and raw==subprocess.check_output(['git','-C',str(repo),'show',rev+':'+name]);dest=out/'sources'/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(raw)
identity={'commit':rev,'sha256':expected,'original_head':head.decode().strip(),'original_dirty_unchanged':True,'original_status_sha256':hashlib.sha256(before).hexdigest(),'checkout':str(repo)};(out/'source_identity.json').write_text(json.dumps(identity,indent=2))
print('FROZEN_SOURCE_VERIFIED '+rev+' '+str(len(expected))+' files; dirty checkout unchanged',flush=True)
worker=r"""import os,json,subprocess,hashlib,time,sys
from pathlib import Path
repo=Path(REPO);out=Path(OUTPUT);identity=json.loads((out/'source_identity.json').read_text())
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
""".replace('REPO',repr(str(repo))).replace('OUTPUT',repr(str(out)))
(out/'worker.py').write_text(worker)
env=os.environ.copy();env.update(PYTHONPATH=str(repo/'qalt/src'),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',SLURM_EXPORT_ENV='ALL')
command=['salloc','-A','cis260243p','-p','RM-shared','--qos=rminteract','-N','1','-n','1','-c','4','--mem=8000M','-t','01:00:00','--job-name=response-v2-interactive-preflight','srun','--export=ALL','/ocean/projects/mth250006p/ywang26/pytorch/bin/python','-u',str(out/'worker.py')]
(out/'allocation_command.json').write_text(json.dumps({'command':command,'attempts':1,'environment_overrides':{k:env[k] for k in ['PYTHONPATH','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','SLURM_EXPORT_ENV']}},indent=2))
start=time.monotonic();process=subprocess.Popen(command,cwd=repo,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,bufsize=1);selector=selectors.DefaultSelector();selector.register(process.stdout,selectors.EVENT_READ,'stdout');selector.register(process.stderr,selectors.EVENT_READ,'stderr');logs={kind:(out/('allocation.'+kind)).open('x') for kind in ['stdout','stderr']}
while selector.get_map():
 for key,_ in selector.select(timeout=30):
  line=key.fileobj.readline()
  if not line:selector.unregister(key.fileobj);continue
  logs[key.data].write(line);logs[key.data].flush();print(line,end='',flush=True)
code=process.wait()
for f in logs.values():f.close()
job=json.loads((out/'allocation_job.json').read_text())['job_id'] if (out/'allocation_job.json').exists() else None
record={'allocation_returncode':code,'allocation_wall_seconds':time.monotonic()-start,'job_id':job,'source_commit':rev,'release':'salloc command exited; allocation released automatically','attempts':1}
if job:
 r=subprocess.run(['sacct','-j',job,'--format=JobID,State,ExitCode,Elapsed,AllocCPUS,ReqMem,NodeList','-P'],capture_output=True,text=True);(out/'accounting.txt').write_text(r.stdout);record['accounting_returncode']=r.returncode
record['payload_sha256']={str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in out.rglob('*') if p.is_file() and p.name!='status.json'};(out/'status.json').write_text(json.dumps(record,indent=2));print('ALLOCATION_FINISHED '+json.dumps(record),flush=True)
'''.replace('REV',repr(rev)).replace('EXPECTED',repr(expected))
(out/'launch.remote.py').open('x').write(remote)
cmd=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=20','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=3','bridges2-codex','/ocean/projects/mth250006p/ywang26/pytorch/bin/python -u -c '+shlex.quote(remote)]
t=time.monotonic();p=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
with (out/'launch-transcript.txt').open('x') as f:
 for line in p.stdout:f.write(line);f.flush();print(line,end='',flush=True)
code=p.wait();(out/'transport-status.json').open('x').write(json.dumps({'returncode':code,'elapsed':time.monotonic()-t,'submission_attempts':1},indent=2));print('SSH_FINISHED',code,flush=True)
