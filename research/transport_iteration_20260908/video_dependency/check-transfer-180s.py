import subprocess,json,time
from pathlib import Path
cmd='p=/ocean/projects/mth250006p/ywang26/diffusion-video-deps-20260909; if [ -d "$p" ]; then find "$p" -maxdepth 1 -type f -printf "%f %s bytes\\n"; if [ -f "$p/av-16.1.0-cp310-cp310-manylinux_2_28_x86_64.whl" ]; then sha256sum "$p/av-16.1.0-cp310-cp310-manylinux_2_28_x86_64.whl"; fi; else echo TARGET_ABSENT; fi'
start=time.monotonic()
try:
 r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ServerAliveInterval=10','-o','ServerAliveCountMax=2','bridges2-codex',cmd],capture_output=True,text=True,timeout=180);record={'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
except subprocess.TimeoutExpired as e:record={'timeout':True,'stdout':str(e.stdout),'stderr':str(e.stderr)}
record['elapsed_seconds']=time.monotonic()-start;Path('work/video-dependency/transfer-readonly-check-180s.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))
