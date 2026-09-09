import hashlib,json,subprocess,sys,os
from pathlib import Path
root=Path('/ocean/projects/mth250006p/ywang26/diffusion-video-deps-20260909');wheel=root/'av-16.1.0-cp310-cp310-manylinux_2_28_x86_64.whl'
assert wheel.stat().st_size==40293006 and hashlib.sha256(wheel.read_bytes()).hexdigest()=='eb990672d97c18f99c02f31c8d5750236f770ffe354b5a52c5f4d16c5e65f619'
assert not (root/'dependencies-attempt2').exists()
record={'status':'started','project_environment_mutated':False}
try:
 p=subprocess.run([sys.executable,'-m','pip','install','--no-index','--no-deps','--no-cache-dir','--disable-pip-version-check','--target',str(root/'dependencies-attempt2'),str(wheel)],capture_output=True,text=True,timeout=55)
 record.update(install_returncode=p.returncode,install_stdout=p.stdout,install_stderr=p.stderr)
 assert p.returncode==0
 env=os.environ.copy();env['PYTHONPATH']=str(root/'dependencies-attempt2');env['PYTHONDONTWRITEBYTECODE']='1';env['OMP_NUM_THREADS']='1';env['OPENBLAS_NUM_THREADS']='1'
 p=subprocess.run([sys.executable,str(root/'smoke-attempt2.py'),str(root)],env=env,capture_output=True,text=True,timeout=45)
 record.update(smoke_returncode=p.returncode,smoke_stdout=p.stdout,smoke_stderr=p.stderr,status='passed' if p.returncode==0 else 'smoke_failed')
except BaseException as e:record.update(status='failed',error=repr(e))
with (root/'installation-record-attempt2.json').open('x') as f:json.dump(record,f,indent=2)
print(json.dumps(record))
