import subprocess,json,time,shlex,hashlib
from pathlib import Path
p=Path('work/video-dependency');remote='/ocean/projects/mth250006p/ywang26/diffusion-video-deps-20260909'
smoke=(p/'smoke.py').read_text().replace("root/'packages'","root/'dependencies-attempt2'").replace("root/'smoke.json'","root/'smoke-attempt2.json'")
installer=(p/'install-smoke-remote.py').read_text().replace("root/'packages'","root/'dependencies-attempt2'").replace("root/'smoke.py'","root/'smoke-attempt2.py'").replace("root/'installation-record.json'","root/'installation-record-attempt2.json'").replace('timeout=25','timeout=55').replace('timeout=20','timeout=45')
files={'wheel-manifest-attempt2.json':(p/'wheel-manifest.json').read_text(),'smoke-attempt2.py':smoke,'install-smoke-attempt2.py':installer}
for n,s in files.items():(p/n).write_text(s)
script='from pathlib import Path\nroot=Path('+repr(remote)+')\nfiles='+repr(files)+'\nfor name,data in files.items():\n with (root/name).open("x") as f:f.write(data)\nprint("attempt2 text bundle created",flush=True)\n'
cmd='/ocean/projects/mth250006p/ywang26/pytorch/bin/python -u -c '+shlex.quote(script)
start=time.monotonic()
try:
 r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=20','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=3','bridges2-codex',cmd],capture_output=True,text=True,timeout=180);record={'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
except subprocess.TimeoutExpired as e:record={'status':'timeout_uncertain_no_retry','stdout':str(e.stdout),'stderr':str(e.stderr)}
record['elapsed_seconds']=time.monotonic()-start;record['source_sha256']={n:hashlib.sha256(s.encode()).hexdigest() for n,s in files.items()};(p/'recovery-transfer.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))
