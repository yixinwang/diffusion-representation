import subprocess,tarfile,pathlib,json,datetime,hashlib
base=pathlib.Path(__file__).parent
out=base/'failure-full';out.mkdir(exist_ok=False)
command='tar -C /ocean/projects/mth250006p/ywang26/diffusion-results -cf - 20260909-innovation-response-v2-attempt2 innovation-response-v2-preflightfixed-45619353.out'
start=datetime.datetime.now(datetime.timezone.utc).isoformat()
with (base/'failure-transfer.stderr').open('wb') as err:
 p=subprocess.Popen(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=20','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=3','bridges2-codex',command],stdout=subprocess.PIPE,stderr=err)
 try:
  with tarfile.open(fileobj=p.stdout,mode='r|') as t:t.extractall(out,filter='data')
  rc=p.wait(timeout=1800)
 except BaseException:
  p.terminate();p.wait(timeout=30);raise
files={str(f.relative_to(out)):{'bytes':f.stat().st_size,'sha256':hashlib.file_digest(f.open('rb'),'sha256').hexdigest()} for f in out.rglob('*') if f.is_file()}
record={'start_utc':start,'end_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'returncode':rc,'command':command,'files':files,'bytes':sum(x['bytes'] for x in files.values())}
(base/'failure-transfer.json').write_text(json.dumps(record,indent=2));print(json.dumps({k:v for k,v in record.items() if k!='files'}));assert rc==0
