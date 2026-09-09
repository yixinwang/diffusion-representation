import json,hashlib,subprocess,sys,os,time,platform,traceback
from pathlib import Path
HERE=Path(__file__).resolve().parent;config=json.loads((HERE/'config.json').read_text());result=Path(config['result_root']);repo=Path(config['checkout']);started=time.perf_counter()
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
report={'status':'running','scope':'saved_codec_fit_audit_no_models_or_canonical_loader','source_commit':config['source_commit'],'job_id':os.environ.get('SLURM_JOB_ID')}
try:
 assert sha(result/'status.json')==config['expected_status_sha']
 status=json.loads((result/'status.json').read_text());identity=json.loads((result/'source_identity.json').read_text());assert identity['commit']==config['source_commit'] and identity['sha256']==config['source_sha256']
 assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()==config['source_commit']
 for rel,value in identity['sha256'].items():
  raw=(result/'source'/rel).read_bytes();assert hashlib.sha256(raw).hexdigest()==value and raw==subprocess.check_output(['git','show',config['source_commit']+':'+rel],cwd=repo)
 files={str(p.relative_to(result)) for p in result.rglob('*') if p.is_file() and p.name!='status.json'};assert files==set(status['payload_sha256'])
 for rel,value in status['payload_sha256'].items():assert sha(result/rel)==value
 for name,value in config['checker_sha256'].items():assert sha(HERE/name)==value
 (HERE/'authentication.json').write_text(json.dumps({'status':'all_payload_and11source_git_verified','expected_status_sha':config['expected_status_sha'],'source_count':11,'payload_count':len(files),'checker_sha256':config['checker_sha256']},indent=2)+'\n')
 import numpy,scipy,torch
 report['versions']={'python':platform.python_version(),'numpy':numpy.__version__,'scipy':scipy.__version__,'torch':torch.__version__};torch.set_num_threads(4)
 commands=[('metrics',[sys.executable,str(HERE/'metric_audit.py'),str(result),'--expected-status-sha',config['expected_status_sha'],'--expected-commit',config['source_commit'],'--repo',str(repo),'--output',str(HERE/'metrics')]),('ledger',[sys.executable,str(HERE/'ledger_audit.py'),str(result),str(HERE/'ledger.json')])]
 report['checks']={}
 for name,command in commands:
  start=time.perf_counter()
  with (HERE/(name+'.stdout')).open('w') as out,(HERE/(name+'.stderr')).open('w') as err:completed=subprocess.run(command,stdout=out,stderr=err,check=False)
  report['checks'][name]={'exit_code':completed.returncode,'elapsed_seconds':time.perf_counter()-start}
  if completed.returncode:raise RuntimeError(name+' audit failed without retry')
 report['status']='completed_audits'
except BaseException as error:report.update(status='failed',error=repr(error),traceback=traceback.format_exc());raise
finally:
 report['elapsed_seconds']=time.perf_counter()-started;(HERE/'audit_status.json').write_text(json.dumps(report,indent=2)+'\n')
