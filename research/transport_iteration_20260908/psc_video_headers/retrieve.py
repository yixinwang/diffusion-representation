import subprocess,json,time,tarfile,io
from pathlib import Path
p=Path('work/psc-video-headers');start=time.monotonic()
cmd='tar -cf - -C /ocean/projects/mth250006p/ywang26/diffusion-results/20260909-video-headers status.json manifest.json archive_reader.py'
try:
 r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=20','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=3','bridges2-codex',cmd],capture_output=True,timeout=180);assert r.returncode==0,r.stderr
 target=p/'results';target.mkdir()
 with tarfile.open(fileobj=io.BytesIO(r.stdout)) as t:
  for member in t:
   assert member.name in ['status.json','manifest.json','archive_reader.py'] and member.isfile()
   (target/member.name).write_bytes(t.extractfile(member).read())
 record={'status':'retrieved','bytes':len(r.stdout),'elapsed_seconds':time.monotonic()-start}
except BaseException as e:record={'status':'failed','error':repr(e)}
(p/'retrieval.json').write_text(json.dumps(record,indent=2)+'\n');print(record)
