from pathlib import Path
import json,base64,hashlib
p=Path(__file__).resolve().parent;c=json.loads((p/'config.json').read_text());raw=(p/'submit-once.sh').read_bytes()
s='''import os,json,hashlib,subprocess,base64,time
from pathlib import Path
C=CONFIG
stage=Path(C['staging']);result=Path(C['result_root']);receipt={}
try:
 assert not result.exists()
 for target in (stage,result.parent):
  output=subprocess.check_output(['lfs','project','-d',str(target)],universal_newlines=True)
  assert output.split()[0]=='559736',output
  probe=target/('.rgb-fit-probe-'+str(os.getpid()));payload=b'bounded codec fit qualification probe\\n'
  with probe.open('xb') as f:f.write(payload);f.flush();os.fsync(f.fileno())
  assert probe.read_bytes()==payload;probe.unlink()
  receipt[str(target)]={'lfs_project':output.strip(),'project_id':559736,'write_fsync_read_remove':True}
 raw=base64.b64decode(PAYLOAD);assert hashlib.sha256(raw).hexdigest()==HASH
 launch=stage/'submit-once.sh'
 with launch.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
 assert hashlib.sha256(launch.read_bytes()).hexdigest()==HASH
 (stage/'startup_verified.json').write_text(json.dumps({'source_commit':C['source_commit'],'script_sha256':HASH,'storage_checks':receipt},indent=2)+'\\n')
 print(json.dumps({'status':'verified_before_single_allocation','script_sha256':HASH,'storage_checks':receipt}),flush=True)
 os.execvp('bash',['bash',str(launch)])
except BaseException as exc:
 (stage/'launch_failure.json').write_text(json.dumps({'error':repr(exc),'storage_checks':receipt},indent=2)+'\\n');raise
'''.replace('CONFIG',repr(c)).replace('PAYLOAD',repr(base64.b64encode(raw).decode())).replace('HASH',repr(hashlib.sha256(raw).hexdigest()))
(p/'upload_and_execute.remote.py').write_text(s)
