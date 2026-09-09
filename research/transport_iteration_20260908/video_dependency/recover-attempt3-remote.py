import os,sys,stat,zipfile,hashlib,json,subprocess,time
from pathlib import Path
root=Path('/ocean/projects/mth250006p/ywang26/diffusion-video-deps-20260909');target=root/'dependencies-attempt3';wheel=root/'av-16.1.0-cp310-cp310-manylinux_2_28_x86_64.whl';start=time.monotonic();record={'status':'started','project_environment_mutated':False}
try:
 assert hashlib.sha256(wheel.read_bytes()).hexdigest()=='eb990672d97c18f99c02f31c8d5750236f770ffe354b5a52c5f4d16c5e65f619'
 with zipfile.ZipFile(wheel) as z:
  members=z.infolist();names=[i.filename for i in members];assert len(names)==len(set(names))
  assert not any(n.split('/')[0].endswith('.data') for n in names)
  for i in members:
   n=Path(i.filename);mode=i.external_attr>>16
   assert not n.is_absolute() and '..' not in n.parts and '\\' not in i.filename and not stat.S_ISLNK(mode) and stat.S_IFMT(mode) in (0,stat.S_IFREG,stat.S_IFDIR)
  target.mkdir();inventory={};print('verified wheel and exclusive target created',flush=True)
  for i in members:
   path=target/i.filename
   if i.is_dir():path.mkdir(parents=True,exist_ok=True);continue
   path.parent.mkdir(parents=True,exist_ok=True);data=z.read(i)
   with path.open('xb') as f:f.write(data)
   inventory[i.filename]={'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)}
 with (root/'extraction-attempt3.json').open('x') as f:json.dump({'wheel_sha256':hashlib.sha256(wheel.read_bytes()).hexdigest(),'files':inventory},f,indent=2)
 print('extraction complete: '+str(len(inventory))+' files',flush=True)
 source=(root/'smoke-attempt2.py').read_text().replace('dependencies-attempt2','dependencies-attempt3').replace('smoke-attempt2.json','smoke-attempt3.json')
 with (root/'smoke-attempt3.py').open('x') as f:f.write(source)
 env=os.environ.copy();env.update(PYTHONPATH=str(target),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
 p=subprocess.run([sys.executable,str(root/'smoke-attempt3.py'),str(root)],env=env,capture_output=True,text=True,timeout=180)
 record.update(status='passed' if p.returncode==0 else 'smoke_failed',smoke_returncode=p.returncode,smoke_stdout=p.stdout,smoke_stderr=p.stderr)
except BaseException as e:record.update(status='failed',error=repr(e))
record['elapsed_seconds']=time.monotonic()-start
with (root/'recovery-record-attempt3.json').open('x') as f:json.dump(record,f,indent=2)
print(json.dumps(record),flush=True)
