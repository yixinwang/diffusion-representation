import subprocess,json,shlex,time,zipfile,hashlib
from pathlib import Path
p=Path('work/video-dependency');script='from pathlib import Path; import json; r=Path("/ocean/projects/mth250006p/ywang26/diffusion-video-deps-20260909"); print(json.dumps({n:json.loads((r/n).read_text()) for n in ["extraction-attempt3.json","smoke-attempt3.json","recovery-record-attempt3.json"]}))'
start=time.monotonic()
try:
 r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=20','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=3','bridges2-codex','/ocean/projects/mth250006p/ywang26/pytorch/bin/python -u -c '+shlex.quote(script)],capture_output=True,text=True,timeout=180)
 assert r.returncode==0,r.stderr
 data=json.loads(r.stdout)
 for n,d in data.items():(p/n).write_text(json.dumps(d,indent=2)+'\n')
 with zipfile.ZipFile(next(p.glob('*.whl'))) as z:
  expected={i.filename:{'sha256':hashlib.sha256(z.read(i)).hexdigest(),'bytes':i.file_size} for i in z.infolist() if not i.is_dir()}
 assert expected==data['extraction-attempt3.json']['files']
 assert data['smoke-attempt3.json']['exact_rgb_roundtrip'] and data['recovery-record-attempt3.json']['status']=='passed'
 record={'status':'retrieved_and_inventory_matches_local_wheel','files_verified':len(expected),'elapsed_seconds':time.monotonic()-start}
except BaseException as e:record={'status':'retrieval_failed','error':repr(e),'elapsed_seconds':time.monotonic()-start}
(p/'retrieval-attempt3.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))
