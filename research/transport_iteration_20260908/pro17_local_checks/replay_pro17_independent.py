import pathlib,subprocess,json,hashlib,tarfile,io,shutil,sys,time,os
w=pathlib.Path(__file__).resolve().parent;repo=w/'diffusion-representation';src=w/'pro17-verified/pro17_artifacts';out=w/'pro17-independent-replay';out.mkdir(exist_ok=False);run=out/'replay';run.mkdir()
base='research/transport_iteration_20260908/pro17_artifacts/';pub='0c47afe50df6e849b968a900073ac57698933281';cherry=subprocess.check_output(['git','-C',str(repo),'rev-parse','21fbaec'],text=True).strip()
def git(c,p):return subprocess.check_output(['git','-C',str(repo),'show',c+':'+p])
def sha(b):return hashlib.sha256(b).hexdigest()
m=json.loads(git(pub,base+'TRANSPORT_MANIFEST.json'));parts=[]
for r in m['parts']:
 b=git(pub,base+r['path']);assert b==git(cherry,base+r['path']);assert len(b)==r['size'] and sha(b)==r['sha256'];parts.append(b)
b=b''.join(parts);assert len(b)==m['archive_size'] and sha(b)==m['archive_sha256']
with tarfile.open(fileobj=io.BytesIO(b),mode='r:xz') as t:
 files=[f for f in t.getmembers() if f.isfile()];assert len(files)==23
 for f in files:assert t.extractfile(f).read()==(src/pathlib.PurePosixPath(f.name).name).read_bytes()
for r in m['payload_files']:
 b=(src/pathlib.PurePosixPath(r['path']).name).read_bytes();assert len(b)==r['size'] and sha(b)==r['sha256']
for r in json.loads((src/'INVENTORY.json').read_text())['files']:
 b=(src/r['path']).read_bytes();assert len(b)==r['size'] and sha(b)==r['sha256']
for line in (src/'SHA256SUMS').read_text().splitlines():
 h,n=line.split(maxsplit=1);assert sha((src/n.lstrip('*')).read_bytes())==h
(out/'authentication.json').write_text(json.dumps({'publication':pub,'cherry_pick':cherry,'all23_payloads_archive_and_original_inventory_verified':True,'transport':m},indent=2)+'\n')
for n in ('copula.py','test_copula.py','run_stress.py','FROZEN_PROTOCOL.json'):shutil.copy2(src/n,run/n)
receipts=[];env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
for name,args in [('tests',['-m','unittest','-v','test_copula.py']),('stress',['run_stress.py'])]:
 start=time.perf_counter()
 with (out/(name+'.stdout')).open('w') as so,(out/(name+'.stderr')).open('w') as se:
  try:p=subprocess.run([sys.executable]+args,cwd=run,env=env,stdout=so,stderr=se,timeout=120);code=p.returncode;error=None
  except Exception as e:code=None;error=repr(e)
 receipts.append({'name':name,'command':[sys.executable]+args,'exit':code,'error':error,'external_wall_seconds':time.perf_counter()-start});(out/'PROCESS.json').write_text(json.dumps({'python':sys.version,'runs':receipts},indent=2)+'\n')
print(receipts)
