import urllib.request,json,hashlib,zipfile,subprocess,time
from pathlib import Path
p=Path('work/video-dependency');j=json.load(urllib.request.urlopen('https://pypi.org/pypi/av/16.1.0/json',timeout=20));f=next(f for f in j['urls'] if f['filename']=='av-16.1.0-cp310-cp310-manylinux_2_28_x86_64.whl');target=p/f['filename']
with urllib.request.urlopen(f['url'],timeout=30) as response: data=response.read()
assert len(data)==f['size'] and hashlib.sha256(data).hexdigest()==f['digests']['sha256']
with target.open('xb') as out:out.write(data)
with zipfile.ZipFile(target) as z:
 for n in ['av-16.1.0.dist-info/METADATA','av-16.1.0.dist-info/WHEEL']:(p/Path(n).name).write_bytes(z.read(n))
(p/'wheel-manifest.json').write_text(json.dumps({'version':'16.1.0','pypi':'https://pypi.org/pypi/av/16.1.0/json',**{k:f[k] for k in ['filename','url','size','digests','requires_python']}},indent=2)+'\n')
cmd="uname -m; getconf GNU_LIBC_VERSION; /ocean/projects/mth250006p/ywang26/pytorch/bin/python -u -c 'import sys,sysconfig,json; print(json.dumps({\"version\":sys.version,\"implementation\":sys.implementation.name,\"SOABI\":sysconfig.get_config_var(\"SOABI\")}))'"
start=time.monotonic()
try:
 r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ServerAliveInterval=10','-o','ServerAliveCountMax=2','bridges2-codex',cmd],capture_output=True,text=True,timeout=60);record={'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
except subprocess.TimeoutExpired as e:record={'timeout':True,'stdout':str(e.stdout),'stderr':str(e.stderr)}
record['elapsed_seconds']=time.monotonic()-start;(p/'platform-probe.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))
