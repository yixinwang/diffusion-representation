import subprocess,json,time,shlex,hashlib
from pathlib import Path
rev='92702635134856f6a0b3a9f82186394b97a7d59e';base='/ocean/projects/mth250006p/ywang26';repo=base+'/diffusion-video-headers-20260909';rel='research/transport_iteration_20260908/video_archive_prerequisite';output=base+'/diffusion-results/20260909-video-headers';local=Path('work/diffusion-representation')
hashes={n:hashlib.sha256(subprocess.check_output(['git','show',rev+':'+rel+'/'+n],cwd=local)).hexdigest() for n in ['run_header_audit.py','archive_reader.py']}
remote='''import subprocess,hashlib,json,sys
from pathlib import Path
rev=REV;repo=Path(REPO);rel=REL;expected=HASHES
subprocess.run(['git','-C',BASE+'/diffusion-validation-20260908','fetch','origin','agent/observation-transport-audit-20260908'],check=True,timeout=90)
assert not repo.exists()
subprocess.run(['git','-C',BASE+'/diffusion-validation-20260908','worktree','add','--detach',str(repo),rev],check=True,timeout=60)
for name,h in expected.items():
 raw=(repo/rel/name).read_bytes();frozen=subprocess.check_output(['git','-C',str(repo),'show',rev+':'+rel+'/'+name]);assert raw==frozen and hashlib.sha256(raw).hexdigest()==h
print('external runner and reader Git/hash guards passed',flush=True)
p=subprocess.run([sys.executable,str(repo/rel/'run_header_audit.py'),'--expected-commit',rev,'--repo',str(repo),'--output',OUTPUT],timeout=120)
print('header_audit_exit',p.returncode,flush=True)
print((Path(OUTPUT)/'status.json').read_text(),flush=True)
'''
for k,v in [('REV',repr(rev)),('REPO',repr(repo)),('REL',repr(rel)),('HASHES',repr(hashes)),('BASE',repr(base)),('OUTPUT',repr(output))]:remote=remote.replace(k,v)
Path('work/psc-video-headers/remote-launch.py').write_text(remote)
cmd=base+'/pytorch/bin/python -u -c '+shlex.quote(remote);start=time.monotonic()
try:
 r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=20','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=3','bridges2-codex',cmd],capture_output=True,text=True,timeout=300);record={'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
except subprocess.TimeoutExpired as e:record={'timeout_uncertain_no_retry':True,'stdout':str(e.stdout),'stderr':str(e.stderr)}
record.update(elapsed_seconds=time.monotonic()-start,commit=rev,source_hashes=hashes,remote_output=output,checkout=repo);Path('work/psc-video-headers/launch-record.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))
