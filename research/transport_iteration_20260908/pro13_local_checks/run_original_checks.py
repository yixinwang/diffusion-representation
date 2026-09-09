from pathlib import Path
import hashlib,json,shutil,subprocess,sys,time,platform
source=Path('work/pro13-staged-recovery/original_subset');root=Path('work/pro13-local-checks');root.mkdir(exist_ok=False)
files=[f for f in source.rglob('*') if f.is_file()];before={str(f.relative_to(source)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
receipt={'original_source_files':len(before),'original_source_sha256':before,'runs':[],'python':sys.version,'platform':platform.platform(),'execution_scope':'unchanged check_math and check_bijection only; no fitting or missing-state regeneration'}
for name in ['check_math','check_bijection']:
 d=root/name;d.mkdir()
 for module in ['pro13.py',name+'.py']:shutil.copyfile(source/module,d/module)
 started=time.monotonic()
 with (d/'stdout.txt').open('xb') as out,(d/'stderr.txt').open('xb') as err:
  try:
   run=subprocess.run([sys.executable,name+'.py'],cwd=d,stdout=out,stderr=err,timeout=120)
   result={'returncode':run.returncode}
  except BaseException as e:result={'exception':repr(e)}
 result.update(script=name+'.py',elapsed_seconds=time.monotonic()-started,sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in d.iterdir() if f.is_file()})
 receipt['runs'].append(result)
receipt['original52_unchanged']=before=={str(f.relative_to(source)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
report=root/'check_bijection/bijection_checks.json'
if report.exists():receipt['full_state_roundtrip_count']=len(json.loads(report.read_text())['full_state_roundtrips'])
receipt['qualification']='Full fitted-state loop is empty because missing original states were not regenerated; small Jacobian only for bijection script.'
(root/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({k:v for k,v in receipt.items() if k!='original_source_sha256'},indent=2))
