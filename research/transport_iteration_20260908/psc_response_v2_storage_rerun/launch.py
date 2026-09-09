import subprocess,json,shlex,time
from pathlib import Path
rev='3b8c0f7bef604a90ff329fd9608639527b67a972';p=Path('work/psc-innovation-response-v2-attempt3');s=json.loads(Path('work/psc-v2-interactive-preflight/full/20260909-response-v2-interactive-preflight/source_identity.json').read_text());expected=s['sha256']
remote=r'''import subprocess,json,hashlib,os
from pathlib import Path
base=Path('/ocean/projects/mth250006p/ywang26');repo=base/'diffusion-v2-interactive-preflight-20260909';original=base/'diffusion-representation';newbase=Path('/ocean/projects/mth260022p/ywang26');out=newbase/'diffusion-results/20260909-innovation-response-v2-attempt3';revision=REV;expected=EXPECTED
assert not out.exists()
assert newbase.is_dir() and os.access(newbase,os.W_OK)
out.parent.mkdir(exist_ok=True)
project_metadata=subprocess.check_output(['lfs','project','-d',str(out.parent)],text=True)
assert project_metadata.split()[0]=='559736',project_metadata
assert os.statvfs(out.parent).f_bavail*os.statvfs(out.parent).f_frsize>10*1024**3
queue=subprocess.run(['squeue','-j','45619353','-h'],capture_output=True,text=True)
assert not queue.stdout.strip(),'failed prior study still queued/running'
assert queue.returncode==0 or 'Invalid job id' in queue.stderr
accounting=subprocess.check_output(['sacct','-j','45619353','--format=JobID,State,ExitCode,Elapsed,AllocCPUS,ReqMem,NodeList','-P'],text=True);print('PRIOR_FAILURE_TERMINAL_ACCOUNTING\n'+accounting,flush=True)
assert subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD']).decode().strip()==revision
assert subprocess.check_output(['git','-C',str(original),'rev-parse','HEAD']).decode().strip()=='4553e04f71c545e401409d18d82731589974760b'
dirty=subprocess.check_output(['git','-C',str(original),'status','--porcelain=v1','-z']);assert hashlib.sha256(dirty).hexdigest()=='568da4cced21bd99ae1c704911780750b2d6053c9e9f0e37a2dee25ef8ba10f0'
for name,h in expected.items():
 raw=(repo/name).read_bytes();assert hashlib.sha256(raw).hexdigest()==h and raw==subprocess.check_output(['git','-C',str(repo),'show',revision+':'+name])
source=base/'diffusion-evaluator-20260909/inception.py';weights=base/'diffusion-evaluator-20260909/pt_inception-2015-12-05-6726825d.pth'
assert hashlib.sha256(source.read_bytes()).hexdigest()=='c6183fff54dd240fe66d53d207f4bd28c06fde98c21b5525f10ca0cc5cef7780'
assert hashlib.sha256(weights.read_bytes()).hexdigest()=='6726825d0af5f729cebd5821db510b11b1cfad8faad88a03f1befd49fb9129b2'
exports={'SOURCE_COMMIT':revision,'RESULT_ROOT':str(out),'INCEPTION_SOURCE':str(source),'INCEPTION_WEIGHTS':str(weights)}
command=['sbatch','--parsable','--job-name=innovation-response-v2-storagefixed','--chdir',str(repo),'--export=ALL,'+','.join(k+'='+v for k,v in exports.items()),'--output',str(newbase/'diffusion-results/innovation-response-v2-storagefixed-%j.out'),'qalt/experiments/innovation_response_pilot_v2/run.slurm']
assert '|FAILED|1:0|' in accounting
print('VERIFIED_SUBMISSION_COMMAND '+json.dumps(command),flush=True)
r=subprocess.run(command,cwd=repo,capture_output=True,text=True)
d={'source_commit':revision,'source_sha256':expected,'original_dirty_unchanged':True,'original_status_sha256':hashlib.sha256(dirty).hexdigest(),'checkout':str(repo),'output':str(out),'command':command,'explicit_exports':exports,'prior_failed_job':'45619353','prior_failure_terminal_accounting':accounting,'submission_attempts':1,'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
if r.returncode==0:d['job_id']=r.stdout.strip().split(';')[0]
print(json.dumps(d),flush=True)
if r.returncode:raise SystemExit(r.returncode)
print(subprocess.check_output(['squeue','-j',d['job_id'],'-h','-o','%i|%T|%R|%M']).decode(),flush=True)
'''.replace('REV',repr(rev)).replace('EXPECTED',repr(expected))
(p/'launch.remote.py').open('x').write(remote);t=time.monotonic()
try:
 r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=20','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=3','bridges2-codex','/ocean/projects/mth250006p/ywang26/pytorch/bin/python -u -c '+shlex.quote(remote)],capture_output=True,text=True,timeout=600);d={'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
except subprocess.TimeoutExpired as e:d={'timeout_uncertain_no_retry':True,'stdout':str(e.stdout),'stderr':str(e.stderr)}
d['elapsed']=time.monotonic()-t;(p/'launch-record.json').open('x').write(json.dumps(d,indent=2));print(json.dumps(d,indent=2))
