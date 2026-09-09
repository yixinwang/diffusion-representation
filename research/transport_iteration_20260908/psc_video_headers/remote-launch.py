import subprocess,hashlib,json,sys
from pathlib import Path
rev='92702635134856f6a0b3a9f82186394b97a7d59e';repo=Path('/ocean/projects/mth250006p/ywang26/diffusion-video-headers-20260909');rel='research/transport_iteration_20260908/video_archive_prerequisite';expected={'run_header_audit.py': '84eca6ae6e2abf3aeb906693992ae57e307e82914757b0fa23e428ab9e13a2c2', 'archive_reader.py': 'cd50355a6e9abfa872aa7eea0489a27df2e592ec94142645dad23eed96d7cf51'}
subprocess.run(['git','-C','/ocean/projects/mth250006p/ywang26'+'/diffusion-validation-20260908','fetch','origin','agent/observation-transport-audit-20260908'],check=True,timeout=90)
assert not repo.exists()
subprocess.run(['git','-C','/ocean/projects/mth250006p/ywang26'+'/diffusion-validation-20260908','worktree','add','--detach',str(repo),rev],check=True,timeout=60)
for name,h in expected.items():
 raw=(repo/rel/name).read_bytes();frozen=subprocess.check_output(['git','-C',str(repo),'show',rev+':'+rel+'/'+name]);assert raw==frozen and hashlib.sha256(raw).hexdigest()==h
print('external runner and reader Git/hash guards passed',flush=True)
p=subprocess.run([sys.executable,str(repo/rel/'run_header_audit.py'),'--expected-commit',rev,'--repo',str(repo),'--output','/ocean/projects/mth250006p/ywang26/diffusion-results/20260909-video-headers'],timeout=120)
print('header_audit_exit',p.returncode,flush=True)
print((Path('/ocean/projects/mth250006p/ywang26/diffusion-results/20260909-video-headers')/'status.json').read_text(),flush=True)
