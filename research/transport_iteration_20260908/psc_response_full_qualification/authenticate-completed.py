import hashlib,json,subprocess,tarfile
from pathlib import Path
root=Path(__file__).resolve().parent;repo=root.parent/'diffusion-representation'
if not (root/'completed').exists():
 with tarfile.open(root/'completed-gpu-and-full.tar') as tar:tar.extractall(root/'completed',filter='data')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();records={}
for name in ['20260909-response-diagnostic-gpu','20260909-response-full-cpu','20260909-response-full-gpu']:
 p=root/'completed'/name;d=json.loads((p/'status.json').read_text());s=json.loads((p/'source_identity.json').read_text())
 for rel,h in d['payload_sha256'].items():assert sha(p/rel)==h,(name,rel)
 for rel,h in s['diagnostic_sha256'].items():
  f=p/'diagnostic_sources'/rel;assert sha(f)==h;assert f.read_bytes()==subprocess.check_output(['git','show',s['diagnostic_commit']+':'+rel],cwd=repo)
 for rel,h in s['reference_sha256'].items():assert sha(root/'full/20260909-innovation-response/sources'/rel)==h
 assert s['failure_status_sha256']==sha(root/'full/20260909-innovation-response/status.json')
 if (p/'full_model/status.json').exists():
  q=json.loads((p/'full_model/status.json').read_text());assert q==d['full_model_qualification']
  for rel,h in q['payload_sha256'].items():assert sha(p/'full_model'/rel)==h
  for field,rel in [('source_bank_sha256','source_bank.pt'),('numerical_bank_sha256','numerical_bank.pt')]:assert q[field]==sha(p/'full_model'/rel)
  assert q['checkpoint_sha256']==sha(root/'full/20260909-innovation-response/seed_78201/RQS_prefix_failed.pt')
 records[name]={'status_sha256':sha(p/'status.json'),'source_identity_sha256':sha(p/'source_identity.json'),'source_commit':s['diagnostic_commit'],'payload_count':len(d['payload_sha256']),'source_count':len(s['diagnostic_sha256']),'reference_source_count':len(s['reference_sha256']),'full_scope_payload_count':len(d.get('full_model_qualification',{}).get('payload_sha256',{})),'bytes':sum(f.stat().st_size for f in p.rglob('*') if f.is_file()),'status':d['status'],'decoder_outcome':d['decoder_outcome']}
(root/'completed-authentication.json').write_text(json.dumps(records,indent=2)+'\n');print(json.dumps(records,indent=2))
