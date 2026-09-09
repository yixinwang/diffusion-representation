import subprocess,tarfile,json,time
from pathlib import Path
p=Path('work/video-dependency');archive=p/'transfer.tar'
with tarfile.open(archive,'w') as t:
 for n in ['av-16.1.0-cp310-cp310-manylinux_2_28_x86_64.whl','wheel-manifest.json','smoke.py']:t.add(p/n,arcname=n,recursive=False)
remote='/ocean/projects/mth250006p/ywang26/diffusion-video-deps-20260909'
command=f'mkdir {remote} && tar -xf - -C {remote}'
start=time.monotonic()
try:
 with archive.open('rb') as data:r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ServerAliveInterval=10','-o','ServerAliveCountMax=2','bridges2-codex',command],stdin=data,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=60)
 record={'returncode':r.returncode,'stdout':r.stdout.decode(),'stderr':r.stderr.decode()}
except subprocess.TimeoutExpired as e:record={'status':'timeout_uncertain_do_not_repeat','stdout':str(e.stdout),'stderr':str(e.stderr)}
record.update(remote=remote,elapsed_seconds=time.monotonic()-start);(p/'transfer.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))
