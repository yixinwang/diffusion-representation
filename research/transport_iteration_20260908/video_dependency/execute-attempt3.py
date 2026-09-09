import subprocess,time,json,shlex
from pathlib import Path
p=Path('work/video-dependency');source=(p/'recover-attempt3-remote.py').read_text();cmd='/ocean/projects/mth250006p/ywang26/pytorch/bin/python -u -c '+shlex.quote(source);start=time.monotonic()
try:
 r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=20','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=3','bridges2-codex',cmd],capture_output=True,text=True,timeout=300);record={'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
except subprocess.TimeoutExpired as e:record={'status':'timeout_uncertain_no_retry','stdout':str(e.stdout),'stderr':str(e.stderr)}
record['elapsed_seconds']=time.monotonic()-start;(p/'execution-attempt3.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))
