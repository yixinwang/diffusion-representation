import subprocess,json,time,shlex,hashlib
from pathlib import Path
rev='92702635134856f6a0b3a9f82186394b97a7d59e';base='/ocean/projects/mth250006p/ywang26';repo=base+'/diffusion-video-headers-20260909';rel='research/transport_iteration_20260908/video_archive_prerequisite';output=base+'/diffusion-results/20260909-video-headers';local=Path('work/diffusion-representation')
hashes={n:hashlib.sha256(subprocess.check_output(['git','show',rev+':'+rel+'/'+n],cwd=local)).hexdigest() for n in ['run_header_audit.py','archive_reader.py']}
remote=Path('work/psc-video-headers/remote-resume.py').read_text()
cmd=base+'/pytorch/bin/python -u -c '+shlex.quote(remote);start=time.monotonic()
try:
 r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=20','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=3','bridges2-codex',cmd],capture_output=True,text=True,timeout=300);record={'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
except subprocess.TimeoutExpired as e:record={'timeout_uncertain_no_retry':True,'stdout':str(e.stdout),'stderr':str(e.stderr)}
record.update(elapsed_seconds=time.monotonic()-start,commit=rev,source_hashes=hashes,remote_output=output,checkout=repo);Path('work/psc-video-headers/resume-record.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))
