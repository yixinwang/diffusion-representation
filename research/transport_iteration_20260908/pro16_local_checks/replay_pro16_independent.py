import hashlib,json,subprocess,pathlib,os,time,tarfile,io,sys,platform
W=pathlib.Path(__file__).resolve().parent; repo=W/'diffusion-representation'; src=W/'pro16-local-audit/pro16_artifacts'; out=W/'pro16-independent-replay'; out.mkdir(exist_ok=False)
base='research/transport_iteration_20260908/pro16_artifacts/'; pub='1ed7ce2d7fc15edf5d73c7002c3c56b054a1d3a8'; cherry='eddae1ae7c28933f18593a214e6ab544da2c13fb'; frozen='168efc227e361a86a2a6aa6952786e5d0e13e30f'
def git(c,p): return subprocess.check_output(['git','-C',str(repo),'show',c+':'+p])
def sha(b):return hashlib.sha256(b).hexdigest()
def blob(b):return hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()
delivery=json.loads(git(pub,base+'DELIVERY.json')); parts=[]
for row in delivery['parts']:
 p=base+'archive_parts/'+row['path']; b=git(pub,p); assert b==git(cherry,p); assert (len(b),sha(b),blob(b))==(row['bytes'],row['sha256'],row['git_blob_sha1']);parts.append(b)
archive=b''.join(parts); assert sha(archive)==delivery['archive_sha256'] and len(archive)==delivery['archive_bytes']; assert sha((src/'MANIFEST.json').read_bytes())==delivery['manifest_sha256']
manifest=json.loads((src/'MANIFEST.json').read_text()); inventory=[]
for row in manifest['files']:
 b=(src/row['path']).read_bytes(); assert (len(b),sha(b),blob(b))==(row['bytes'],row['sha256'],row['git_blob_sha1']); inventory.append(row)
with tarfile.open(fileobj=io.BytesIO(archive),mode='r:xz') as t:
 files=[m for m in t.getmembers() if m.isfile()]; assert len(files)==26
 for m in files:
  rel='/'.join(pathlib.PurePosixPath(m.name).parts[1:]); assert t.extractfile(m).read()==(src/rel).read_bytes(),m.name
originals=[]
for name in ('dense_spline.py','dense_global_conditional_spline.py','spline.py'):
 b=(src/'original'/name).read_bytes(); assert b==git(frozen,'qalt/src/qalt/'+name)==git(pub,base+'original/'+name)==git(cherry,base+'original/'+name); originals.append({'name':name,'sha256':sha(b),'git_blob':blob(b)})
assert (src/'stable_spline.py').read_bytes()==git(pub,base+'stable_spline.py')==git(cherry,base+'stable_spline.py')
(out/'authentication.json').write_text(json.dumps({'publication':pub,'cherry_pick':cherry,'frozen_originals':frozen,'delivery':delivery,'manifest':manifest,'originals':originals,'all_26_archive_files_match_extraction':True},indent=2)+'\n')
env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
receipt={'python':sys.version,'executable':sys.executable,'platform':platform.platform(),'runs':[]}
for script,dest in [('run_audit.py','audit'),('run_supplement.py','supplement')]:
 cmd=[sys.executable,str(src/script),'--out',str(out/dest)]; start=time.perf_counter()
 with (out/(dest+'.stdout')).open('w') as so,(out/(dest+'.stderr')).open('w') as se:
  try:
   p=subprocess.run(cmd,stdout=so,stderr=se,env=env,timeout=180); code=p.returncode; error=None
  except Exception as e: code=None; error=repr(e)
 receipt['runs'].append({'command':cmd,'exit_code':code,'error':error,'external_wall_seconds':time.perf_counter()-start}); (out/'PROCESS.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2))
