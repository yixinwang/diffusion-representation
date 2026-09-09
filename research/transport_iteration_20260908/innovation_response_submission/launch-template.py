"""Prepared launcher; refuses to run without a separately supplied frozen revision.

Only instantiate after root freeze and concrete launch authorization. No action
is performed by importing this file. Exactly one sbatch call; uncertain output
must be investigated without retrying the command.
"""
import argparse,hashlib,json,shlex,subprocess,time
from pathlib import Path


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--commit',required=True)
    parser.add_argument('--repo',type=Path,required=True)
    parser.add_argument('--record',type=Path,required=True)
    args=parser.parse_args()
    if len(args.commit)!=40 or any(c not in '0123456789abcdef' for c in args.commit):raise ValueError('full frozen revision required')
    if args.record.exists():raise FileExistsError('preserve previous launch; do not overwrite')
    revision=args.commit
    names=subprocess.check_output(['git','ls-tree','-r','--name-only',revision],cwd=args.repo,text=True).splitlines()
    def selected(n):
        p=Path(n)
        return ((n.startswith('qalt/src/qalt/') and p.suffix=='.py')
          or (n.startswith('qalt/tests/') and len(p.parts)==3 and p.suffix=='.py')
          or n=='qalt/data/observed_manifest_v1.json'
          or (str(p.parent) in ('qalt/experiments/innovation_response_pilot','qalt/experiments/perceptual_appendix') and p.suffix in ('.py','.md','.slurm'))
          or n in ('qalt/experiments/observed_flow_pilot/run_shared.py','qalt/experiments/observed_flow_pilot/SHARED_PROTOCOL.md'))
    expected={n:hashlib.sha256(subprocess.check_output(['git','show',revision+':'+n],cwd=args.repo)).hexdigest() for n in names if selected(n)}
    if not {'qalt/experiments/innovation_response_pilot/run.py','qalt/experiments/innovation_response_pilot/run.slurm','qalt/src/qalt/innovation_response.py'}.issubset(expected):raise ValueError('frozen closure incomplete')
    remote='''import subprocess,hashlib,json,os
from pathlib import Path
base=Path('/ocean/projects/mth250006p/ywang26');safe=base/'diffusion-validation-20260908';repo=base/'diffusion-innovation-response-20260909';out=base/'diffusion-results/20260909-innovation-response';revision=REVISION;expected=EXPECTED
source=base/'diffusion-evaluator-20260909/inception.py';weights=base/'diffusion-evaluator-20260909/pt_inception-2015-12-05-6726825d.pth'
assert not repo.exists() and not out.exists()
subprocess.run(['git','-C',str(safe),'fetch','origin','agent/observation-transport-audit-20260908'],check=True)
subprocess.run(['git','-C',str(safe),'worktree','add','--quiet','--detach',str(repo),revision],check=True)
assert subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD']).decode().strip()==revision
for name,h in expected.items():
 raw=(repo/name).read_bytes();assert hashlib.sha256(raw).hexdigest()==h and subprocess.check_output(['git','-C',str(repo),'show',revision+':'+name])==raw
assert hashlib.sha256(source.read_bytes()).hexdigest()=='c6183fff54dd240fe66d53d207f4bd28c06fde98c21b5525f10ca0cc5cef7780'
assert hashlib.sha256(weights.read_bytes()).hexdigest()=='6726825d0af5f729cebd5821db510b11b1cfad8faad88a03f1befd49fb9129b2'
exports={'SOURCE_COMMIT':revision,'RESULT_ROOT':str(out),'INCEPTION_SOURCE':str(source),'INCEPTION_WEIGHTS':str(weights)}
export_argument='--export=ALL,'+','.join(k+'='+v for k,v in exports.items())
assert all(',' not in v and '\\n' not in v for v in exports.values())
command=['sbatch','--parsable','--job-name=innovation-response','--chdir',str(repo),export_argument,'--output',str(base/'diffusion-results/innovation-response-%j.out'),'qalt/experiments/innovation_response_pilot/run.slurm']
print('external closure/assets verified; explicit exports: '+export_argument,flush=True)
r=subprocess.run(command,cwd=repo,capture_output=True,text=True)
record={'source_commit':revision,'checkout':str(repo),'output':str(out),'command':command,'explicit_exports':exports,'source_sha256':expected,'submission_returncode':r.returncode,'submission_stdout':r.stdout,'submission_stderr':r.stderr,'submission_attempts':1,'login_SBATCH_EXPORT':os.environ.get('SBATCH_EXPORT')}
if r.returncode==0:
 job=r.stdout.strip().split(';')[0];assert job.isdigit();record['job_id']=job
print(json.dumps(record),flush=True)
if r.returncode:raise SystemExit(r.returncode)
print(subprocess.check_output(['squeue','-j',job,'-h','-o','%i|%T|%R|%M']).decode(),flush=True)
'''.replace('REVISION',repr(revision)).replace('EXPECTED',repr(expected))
    args.record.parent.mkdir(parents=True,exist_ok=True)
    script_record=args.record.with_suffix('.remote.py')
    with script_record.open('x') as f:f.write(remote)
    start=time.monotonic()
    try:
        r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=20','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=3','bridges2-codex','/ocean/projects/mth250006p/ywang26/pytorch/bin/python -u -c '+shlex.quote(remote)],capture_output=True,text=True,timeout=600)
        record={'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
    except subprocess.TimeoutExpired as e:
        record={'timeout_uncertain_no_retry':True,'stdout':str(e.stdout),'stderr':str(e.stderr)}
    record['elapsed_seconds']=time.monotonic()-start
    with args.record.open('x') as f:json.dump(record,f,indent=2)
    print(json.dumps(record),flush=True)

if __name__=='__main__':main()
